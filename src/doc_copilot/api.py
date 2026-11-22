"""FastAPI backend for the chat UI.

Run: uv run uvicorn doc_copilot.api:app --port 8001
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

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
    app.add_middleware(
        CORSMiddleware, allow_origins=list(s.cors_origins), allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"ok": True, "llm_provider": s.llm_provider, "chunking": s.chunking}

    @app.get("/filters")
    def filters(request: Request):
        with request.app.state.pool.connection() as conn:
            return list_filters(conn)

    @app.post("/ask")
    def ask(body: AskRequest, request: Request):
        with request.app.state.pool.connection() as conn:
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


def __getattr__(name: str):
    # `uvicorn doc_copilot.api:app` builds the app lazily so importing this module in tests
    # doesn't read the environment or connect to anything.
    if name == "app":
        return create_app()
    raise AttributeError(name)
