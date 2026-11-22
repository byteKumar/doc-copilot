"""Test setup. Runs before any project import.

Safety first: every ANTHROPIC_* variable is removed and LLM_PROVIDER/EMBEDDER are forced
to the offline fakes, so no test can ever reach a paid API.
"""

from __future__ import annotations

import os

for _k in list(os.environ):
    if _k.startswith("ANTHROPIC_"):
        del os.environ[_k]
os.environ["LLM_PROVIDER"] = "fake"
os.environ["EMBEDDER"] = "fake"
os.environ["DOC_COPILOT_TESTING"] = "1"
os.environ.pop("ACCESS_CODE", None)

from pathlib import Path  # noqa: E402

import psycopg  # noqa: E402
import pytest  # noqa: E402

from doc_copilot.config import Settings  # noqa: E402
from doc_copilot.db import connect, init_db  # noqa: E402
from doc_copilot.embed import FakeEmbedder  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
FILINGS = FIXTURES / "filings"
TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/doc_copilot_test"
)


def _ensure_database(url: str) -> None:
    base, _, name = url.rpartition("/")
    with psycopg.connect(f"{base}/postgres", autocommit=True, connect_timeout=3) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname=%s", (name,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{name}"')


@pytest.fixture(scope="session")
def db_url() -> str:
    try:
        _ensure_database(TEST_DB_URL)
    except psycopg.OperationalError as e:
        if os.environ.get("REQUIRE_DB") == "1":
            raise
        pytest.skip(f"Postgres not reachable at {TEST_DB_URL} ({e}). Run: docker compose up -d db")
    init_db(TEST_DB_URL, 384)
    return TEST_DB_URL


@pytest.fixture
def conn(db_url):
    c = connect(db_url)
    c.execute("TRUNCATE feedback, query_log, chunks, documents RESTART IDENTITY CASCADE")
    yield c
    c.close()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(384)


@pytest.fixture
def settings(db_url) -> Settings:
    return Settings(database_url=db_url, embedder="fake", llm_provider="fake")


@pytest.fixture
def ingested(conn, embedder):
    from doc_copilot.ingest.pipeline import ingest_dir

    ingest_dir(conn, FILINGS, embedder)
    return conn
