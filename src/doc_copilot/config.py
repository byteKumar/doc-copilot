"""All runtime settings come from environment variables, read once into a frozen dataclass.

Nothing here reads a `.env` file. Locally, export variables in your shell (or use
`uv run --env-file .env ...`, which you run yourself). Defaults are safe: the LLM
provider defaults to the offline FakeLLM, so no paid call happens unless
LLM_PROVIDER=anthropic is set explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql://postgres:postgres@localhost:5434/doc_copilot"

    # Embeddings: "local" = sentence-transformers all-MiniLM-L6-v2, "fake" = hashing embedder.
    embedder: str = "fake"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # LLM: "fake" (offline, deterministic) or "anthropic" (paid; must be set explicitly).
    llm_provider: str = "fake"
    llm_model: str = "claude-haiku-5-5"
    llm_max_tokens: int = 4000
    # USD per 1M tokens. Check current pricing before a paid run and update via env.
    price_input_per_mtok: float = 0.10
    price_output_per_mtok: float = 0.50

    # Chunking strategy used at query time (see ingest/chunk.py).
    chunking: str = "baseline"
    chunk_words: int = 220
    chunk_overlap_words: int = 40

    # Retrieval.
    top_k: int = 8
    candidates_per_retriever: int = 30
    rrf_k: int = 60
    # Refuse before calling the LLM if the best vector similarity is below this.
    # Tune it on the dev split only, never on the test split.
    min_similarity: float = 0.25

    # Deployment guards.
    access_code: str = ""
    rate_limit_per_minute: int = 10
    daily_query_cap: int = 500
    cors_origins: tuple[str, ...] = field(default=("http://localhost:3001",))

    # SEC EDGAR downloader. Must be set to "Your Name your-contact" before downloading.
    sec_user_agent: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            database_url=_env("DATABASE_URL", d.database_url),
            embedder=_env("EMBEDDER", d.embedder),
            embedding_model=_env("EMBEDDING_MODEL", d.embedding_model),
            embedding_dim=_env_int("EMBEDDING_DIM", d.embedding_dim),
            llm_provider=_env("LLM_PROVIDER", d.llm_provider),
            llm_model=_env("LLM_MODEL", d.llm_model),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", d.llm_max_tokens),
            price_input_per_mtok=_env_float("LLM_PRICE_INPUT_PER_MTOK", d.price_input_per_mtok),
            price_output_per_mtok=_env_float("LLM_PRICE_OUTPUT_PER_MTOK", d.price_output_per_mtok),
            chunking=_env("CHUNKING", d.chunking),
            chunk_words=_env_int("CHUNK_WORDS", d.chunk_words),
            chunk_overlap_words=_env_int("CHUNK_OVERLAP_WORDS", d.chunk_overlap_words),
            top_k=_env_int("TOP_K", d.top_k),
            candidates_per_retriever=_env_int("CANDIDATES_PER_RETRIEVER", d.candidates_per_retriever),
            rrf_k=_env_int("RRF_K", d.rrf_k),
            min_similarity=_env_float("MIN_SIMILARITY", d.min_similarity),
            access_code=_env("ACCESS_CODE", d.access_code),
            rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", d.rate_limit_per_minute),
            daily_query_cap=_env_int("DAILY_QUERY_CAP", d.daily_query_cap),
            cors_origins=tuple(
                o.strip() for o in _env("CORS_ORIGINS", ",".join(d.cors_origins)).split(",") if o.strip()
            ),
            sec_user_agent=_env("SEC_USER_AGENT", d.sec_user_agent),
        )

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.price_input_per_mtok + output_tokens * self.price_output_per_mtok
        ) / 1_000_000
