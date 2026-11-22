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
