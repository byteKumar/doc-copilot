import pytest
from fastapi.testclient import TestClient

from doc_copilot.api import create_app
from doc_copilot.config import Settings
from doc_copilot.embed import FakeEmbedder
from doc_copilot.llm import FakeLLM

pytestmark = pytest.mark.db


@pytest.fixture
def client(ingested, settings):
    s = Settings(**{**settings.__dict__, "min_similarity": 0.0})
    with TestClient(create_app(s, FakeEmbedder(384), FakeLLM())) as c:
        yield c


def test_health_and_filters(client):
    assert client.get("/health").json()["llm_provider"] == "fake"
    f = client.get("/filters").json()
    assert [c["ticker"] for c in f["companies"]] == ["ACME", "GLBX"]


def test_ask_cite_open_and_feedback(client, ingested):
    r = client.post("/ask", json={"question": "How many service centers does Globex operate?", "company": "GLBX"})
    assert r.status_code == 200
    body = r.json()
    assert not body["refused"] and "312" in body["answer"]
    cid = body["citations"][0]["id"]

    chunk = client.get(f"/chunks/{cid}").json()
    assert "312 service centers" in chunk["text"] and chunk["ticker"] == "GLBX"

    fb = client.post("/feedback", json={"query_id": body["query_id"], "rating": -1, "comment": "too short"})
    assert fb.json() == {"ok": True}
    row = ingested.execute("SELECT rating, comment FROM feedback").fetchone()
    assert row == (-1, "too short")


def test_errors(client):
    assert client.get("/chunks/NOPE-1").status_code == 404
    assert client.post("/feedback", json={"query_id": "00000000-0000-0000-0000-000000000000", "rating": 1}).status_code == 404
    assert client.post("/feedback", json={"query_id": "x", "rating": 5}).status_code == 422
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_usage_reports_real_counts(client):
    client.post("/ask", json={"question": "How many service centers does Globex operate?"})
    client.post("/ask", json={"question": "Where is Globex headquartered?"})
    u = client.get("/usage").json()
    assert u["queries_total"] == 2 and u["queries_24h"] == 2
    assert u["latency_p50_ms"] > 0 and u["cost_total_usd"] > 0
    assert len(u["recent"]) == 2 and u["by_day"][0]["queries"] == 2


def make_client(settings, **overrides):
    s = Settings(**{**settings.__dict__, "min_similarity": 0.0, **overrides})
    return TestClient(create_app(s, FakeEmbedder(384), FakeLLM()))


def test_access_code(ingested, settings):
    with make_client(settings, access_code="letmein") as c:
        assert c.get("/health").status_code == 200  # open for health checks
        assert c.get("/filters").status_code == 401
        assert c.get("/filters", headers={"X-Access-Code": "wrong"}).status_code == 401
        assert c.get("/filters", headers={"X-Access-Code": "letmein"}).status_code == 200
        pre = c.options("/ask", headers={"Origin": "http://localhost:3001", "Access-Control-Request-Method": "POST"})
        assert pre.status_code == 200
        r = c.get("/usage", headers={"Origin": "http://localhost:3001"})
        assert r.status_code == 401 and r.headers.get("access-control-allow-origin") == "http://localhost:3001"


def test_rate_limit(ingested, settings):
    with make_client(settings, rate_limit_per_minute=2) as c:
        q = {"question": "Where is Globex headquartered?"}
        assert c.post("/ask", json=q).status_code == 200
        assert c.post("/ask", json=q).status_code == 200
        r = c.post("/ask", json=q)
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
        # another client is not affected
        assert c.post("/ask", json=q, headers={"X-Forwarded-For": "10.0.0.9"}).status_code == 200


def test_daily_cap(ingested, settings):
    with make_client(settings, daily_query_cap=1) as c:
        q = {"question": "Where is Globex headquartered?"}
        assert c.post("/ask", json=q).status_code == 200
        assert c.post("/ask", json=q).status_code == 503


def test_rate_limiter_window():
    from doc_copilot.api import RateLimiter

    rl = RateLimiter(2, window_s=60)
    assert rl.check("a", now=0) == 0 and rl.check("a", now=1) == 0
    assert rl.check("a", now=2) == pytest.approx(58)
    assert rl.check("a", now=60.5) == 0  # first hit has aged out
