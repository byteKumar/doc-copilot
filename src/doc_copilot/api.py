"""FastAPI backend for the chat UI.

Run: uv run uvicorn doc_copilot.api:app --port 8001
"""

from __future__ import annotations

import hmac
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from .answer import Answerer
from .config import Settings
from .db import init_db
from .embed import Embedder, make_embedder
from .llm import LLM, make_llm
from .retrieve import Filters, get_chunk, list_filters

log = logging.getLogger("doc_copilot.api")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    company: str | None = None
    year: int | None = None


class FeedbackRequest(BaseModel):
    query_id: str
    rating: int = Field(description="1 for thumbs up, -1 for thumbs down")
    comment: str | None = Field(default=None, max_length=2000)


class RateLimiter:
    """Sliding window per client: at most `limit` requests in any 60 seconds.

    In memory, so it resets on restart and is per process. That is fine for a single-instance
    demo; with several instances you would keep the counters in Postgres or Redis."""

    def __init__(self, limit: int, window_s: float = 60.0):
        self.limit, self.window = limit, window_s
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> float:
        """Return 0 if allowed, else seconds until the next request is allowed."""
        now = time.monotonic() if now is None else now
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return self.window - (now - q[0])
            q.append(now)
            return 0.0


def client_key(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class AccessCodeMiddleware(BaseHTTPMiddleware):
    """When ACCESS_CODE is set, every request except /health and CORS preflight must send
    it in the X-Access-Code header. Compared in constant time."""

    OPEN_PATHS = {"/health"}

    def __init__(self, app, code: str):
        super().__init__(app)
        self.code = code.encode()

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in self.OPEN_PATHS:
            return await call_next(request)
        given = request.headers.get("x-access-code", "").encode()
        if not hmac.compare_digest(given, self.code):
            return JSONResponse({"detail": "access code required"}, status_code=401)
        return await call_next(request)


def create_app(
    settings: Settings | None = None, embedder: Embedder | None = None, llm: LLM | None = None
) -> FastAPI:
    s = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(s.database_url, s.embedding_dim)
        app.state.pool = ConnectionPool(
            s.database_url, min_size=1, max_size=5, open=True,
            kwargs={"autocommit": True}, configure=register_vector,
        )
        app.state.answerer = Answerer(s, embedder or make_embedder(s), llm or make_llm(s))
        if s.llm_provider == "fake":
            log.warning("LLM_PROVIDER=fake: answers come from the offline stand-in, not Claude")
        yield
        app.state.pool.close()

    app = FastAPI(title="doc-copilot", lifespan=lifespan)
    app.state.settings = s
    app.state.limiter = RateLimiter(s.rate_limit_per_minute)
    if s.access_code:
        app.add_middleware(AccessCodeMiddleware, code=s.access_code)
    # Added last so it wraps everything: 401/429 responses still carry CORS headers.
    app.add_middleware(
        CORSMiddleware, allow_origins=list(s.cors_origins), allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"ok": True, "llm_provider": s.llm_provider, "chunking": s.chunking}

    @app.get("/auth/check")
    def auth_check():
        return {"ok": True}

    @app.get("/filters")
    def filters(request: Request):
        with request.app.state.pool.connection() as conn:
            return list_filters(conn)

    @app.post("/ask")
    def ask(body: AskRequest, request: Request):
        wait = request.app.state.limiter.check(client_key(request))
        if wait:
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(int(wait) + 1)})
        with request.app.state.pool.connection() as conn:
            used = conn.execute(
                "SELECT count(*) FROM query_log WHERE llm_called AND created_at > now() - interval '1 day'"
            ).fetchone()[0]
            if used >= s.daily_query_cap:
                raise HTTPException(503, "daily query budget reached; try again tomorrow")
            r = request.app.state.answerer.answer(conn, body.question.strip(), Filters(body.company, body.year))
        return r.to_dict()

    @app.get("/chunks/{chunk_id}")
    def chunk(chunk_id: str, request: Request):
        with request.app.state.pool.connection() as conn:
            h = get_chunk(conn, chunk_id)
            if not h:
                raise HTTPException(404, "chunk not found")
            src = conn.execute(
                "SELECT d.source_url FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=%s",
                (chunk_id,),
            ).fetchone()
        return {
            "id": h.id, "company": h.company, "ticker": h.ticker, "year": h.year,
            "section": h.section, "section_title": h.section_title, "anchor": h.anchor,
            "kind": h.kind, "text": h.text,
            "source_url": (src[0] + (f"#{h.anchor}" if h.anchor else "")) if src and src[0] else None,
        }

    @app.get("/usage")
    def usage(request: Request):
        with request.app.state.pool.connection() as conn:
            return usage_stats(conn, s.daily_query_cap)

    @app.post("/feedback")
    def feedback(body: FeedbackRequest, request: Request):
        if body.rating not in (1, -1):
            raise HTTPException(422, "rating must be 1 or -1")
        with request.app.state.pool.connection() as conn:
            exists = conn.execute("SELECT 1 FROM query_log WHERE id::text=%s", (body.query_id,)).fetchone()
            if not exists:
                raise HTTPException(404, "unknown query_id")
            conn.execute(
                "INSERT INTO feedback (query_id, rating, comment) VALUES (%s, %s, %s)",
                (body.query_id, body.rating, body.comment),
            )
        return {"ok": True}

    return app


def usage_stats(conn, daily_cap: int) -> dict:
    t = conn.execute(
        """SELECT count(*),
                  count(*) FILTER (WHERE created_at > now() - interval '1 day'),
                  count(*) FILTER (WHERE refused),
                  count(*) FILTER (WHERE llm_called),
                  count(*) FILTER (WHERE llm_called AND created_at > now() - interval '1 day'),
                  percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms),
                  percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms),
                  coalesce(sum(cost_usd), 0), coalesce(sum(input_tokens), 0),
                  coalesce(sum(output_tokens), 0)
           FROM query_log"""
    ).fetchone()
    fb = conn.execute(
        "SELECT count(*) FILTER (WHERE rating = 1), count(*) FILTER (WHERE rating = -1) FROM feedback"
    ).fetchone()
    days = conn.execute(
        """SELECT to_char(date_trunc('day', created_at), 'YYYY-MM-DD'), count(*), coalesce(sum(cost_usd), 0),
                  percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms)
           FROM query_log WHERE created_at > now() - interval '7 days'
           GROUP BY 1 ORDER BY 1 DESC"""
    ).fetchall()
    recent = conn.execute(
        """SELECT created_at, left(question, 120), refused, latency_ms, cost_usd
           FROM query_log ORDER BY created_at DESC LIMIT 20"""
    ).fetchall()
    total = t[0]
    return {
        "queries_total": total,
        "queries_24h": t[1],
        "refusal_rate": (t[2] / total) if total else None,
        "llm_calls_total": t[3],
        "llm_calls_24h": t[4],
        "daily_cap": daily_cap,
        "latency_p50_ms": t[5],
        "latency_p95_ms": t[6],
        "cost_total_usd": float(t[7]),
        "cost_per_1000_queries_usd": (float(t[7]) / total * 1000) if total else None,
        "input_tokens_total": int(t[8]),
        "output_tokens_total": int(t[9]),
        "feedback_up": fb[0],
        "feedback_down": fb[1],
        "by_day": [{"day": d[0], "queries": d[1], "cost_usd": float(d[2]), "latency_p50_ms": d[3]} for d in days],
        "recent": [
            {"at": r[0].isoformat(), "question": r[1], "refused": r[2], "latency_ms": r[3], "cost_usd": r[4]}
            for r in recent
        ],
    }


def __getattr__(name: str):
    # `uvicorn doc_copilot.api:app` builds the app lazily so importing this module in tests
    # doesn't read the environment or connect to anything.
    if name == "app":
        return create_app()
    raise AttributeError(name)
