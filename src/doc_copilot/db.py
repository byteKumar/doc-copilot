"""Postgres access: connection helper and schema.

One `chunks` table holds both chunking strategies side by side (column `chunking`),
so the eval harness can compare them on the same database without re-ingesting.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector


def schema_sql(dim: int) -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            SERIAL PRIMARY KEY,
    ticker        TEXT NOT NULL,
    company       TEXT NOT NULL,
    year          INT  NOT NULL,
    form          TEXT NOT NULL DEFAULT '10-K',
    source_path   TEXT NOT NULL,
    source_url    TEXT,
    content_hash  TEXT NOT NULL,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (ticker, year, form)
);

CREATE TABLE IF NOT EXISTS chunks (
    id             TEXT PRIMARY KEY,
    document_id    INT  NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunking       TEXT NOT NULL,
    ticker         TEXT NOT NULL,
    company        TEXT NOT NULL,
    year           INT  NOT NULL,
    section        TEXT NOT NULL,
    section_title  TEXT NOT NULL,
    anchor         TEXT,
    chunk_index    INT  NOT NULL,
    kind           TEXT NOT NULL,
    text           TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    embedding      vector({dim}) NOT NULL,
    tsv            tsvector GENERATED ALWAYS AS (
                       to_tsvector('english', section_title || ' ' || text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_filter_idx ON chunks (chunking, ticker, year);

CREATE TABLE IF NOT EXISTS query_log (
    id             UUID PRIMARY KEY,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    question       TEXT NOT NULL,
    filters        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    answer         TEXT NOT NULL,
    citations      JSONB NOT NULL DEFAULT '[]'::jsonb,
    refused        BOOLEAN NOT NULL,
    refusal_reason TEXT,
    latency_ms     DOUBLE PRECISION NOT NULL,
    input_tokens   INT NOT NULL DEFAULT 0,
    output_tokens  INT NOT NULL DEFAULT 0,
    cost_usd       DOUBLE PRECISION NOT NULL DEFAULT 0,
    llm_called     BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS query_log_created_idx ON query_log (created_at);

CREATE TABLE IF NOT EXISTS feedback (
    id          SERIAL PRIMARY KEY,
    query_id    UUID NOT NULL REFERENCES query_log(id) ON DELETE CASCADE,
    rating      SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    comment     TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def connect(database_url: str) -> psycopg.Connection:
    conn = psycopg.connect(database_url, autocommit=True)
    register_vector(conn)
    return conn


@contextmanager
def connection(database_url: str) -> Iterator[psycopg.Connection]:
    conn = connect(database_url)
    try:
        yield conn
    finally:
        conn.close()


def init_db(database_url: str, dim: int) -> None:
    # The extension must exist before register_vector can find the type.
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    with connection(database_url) as conn:
        conn.execute(schema_sql(dim))
