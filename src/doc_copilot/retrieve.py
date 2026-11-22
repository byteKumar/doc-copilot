"""Hybrid retrieval: Postgres full-text search + pgvector similarity, merged with RRF.

Why both: full-text search is precise on exact terms ("Item 1A", "lidar", "backlog"), and
vector search catches paraphrases ("how many people work there" vs "full-time employees").
Their scores are on different scales, so instead of adding scores we add reciprocal ranks:

    rrf(chunk) = sum over retrievers of 1 / (k + rank_in_that_retriever)

k=60 is the value from the original RRF paper; it damps the advantage of rank 1 so a chunk
that is decent in both lists beats one that is first in only one list.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import psycopg

from .embed import Embedder

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class Filters:
    company: str | None = None  # ticker or part of the company name
    year: int | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in {"company": self.company, "year": self.year}.items() if v}


@dataclass
class Hit:
    id: str
    ticker: str
    company: str
    year: int
    section: str
    section_title: str
    anchor: str | None
    kind: str
    text: str
    rrf_score: float = 0.0
    vector_rank: int | None = None
    vector_similarity: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None


@dataclass
class RetrievalResult:
    hits: list[Hit]
    best_similarity: float
    lexical_matches: int
    latency_ms: float
    debug: dict = field(default_factory=dict)


def or_tsquery(question: str) -> str:
    """Natural-language questions rarely contain every term of the answer, so OR the terms
    (to_tsquery drops English stop words and stems the rest)."""
    words = [w for w in _WORD.findall(question.lower()) if len(w) > 1]
    return " | ".join(dict.fromkeys(words))


def _where(filters: Filters, chunking: str) -> tuple[str, list]:
    clauses, params = ["chunking = %s"], [chunking]
    if filters.company:
        clauses.append("(ticker = upper(%s) OR company ILIKE %s)")
        params += [filters.company, f"%{filters.company}%"]
    if filters.year:
        clauses.append("year = %s")
        params.append(int(filters.year))
    return " AND ".join(clauses), params


_COLS = "id, ticker, company, year, section, section_title, anchor, kind, text"


def rrf_merge(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def retrieve(
    conn: psycopg.Connection,
    embedder: Embedder,
    question: str,
    filters: Filters | None = None,
    chunking: str = "baseline",
    top_k: int = 8,
    candidates: int = 30,
    rrf_k: int = 60,
) -> RetrievalResult:
    t0 = time.perf_counter()
    filters = filters or Filters()
    where, params = _where(filters, chunking)
    qvec = embedder.embed([question])[0]

    vec_rows = conn.execute(
        f"SELECT {_COLS}, 1 - (embedding <=> %s::vector) AS sim FROM chunks WHERE {where} "
        "ORDER BY embedding <=> %s::vector LIMIT %s",
        [str(qvec), *params, str(qvec), candidates],
    ).fetchall()

    tsq = or_tsquery(question)
    lex_rows = []
    if tsq:
        lex_rows = conn.execute(
            f"SELECT {_COLS}, ts_rank_cd(tsv, q) AS score "
            f"FROM chunks, to_tsquery('english', %s) q WHERE {where} AND tsv @@ q "
            "ORDER BY score DESC, id LIMIT %s",
            [tsq, *params, candidates],
        ).fetchall()

    hits: dict[str, Hit] = {}

    def hit(row) -> Hit:
        cid = row[0]
        if cid not in hits:
            hits[cid] = Hit(*row[:9])
        return hits[cid]

    for rank, row in enumerate(vec_rows, start=1):
        h = hit(row)
        h.vector_rank, h.vector_similarity = rank, float(row[9])
    for rank, row in enumerate(lex_rows, start=1):
        h = hit(row)
        h.lexical_rank, h.lexical_score = rank, float(row[9])

    scores = rrf_merge([[r[0] for r in vec_rows], [r[0] for r in lex_rows]], k=rrf_k)
    for cid, s in scores.items():
        hits[cid].rrf_score = s
    ranked = sorted(hits.values(), key=lambda h: (-h.rrf_score, h.id))[:top_k]

    return RetrievalResult(
        hits=ranked,
        best_similarity=max((float(r[9]) for r in vec_rows), default=0.0),
        lexical_matches=len(lex_rows),
        latency_ms=(time.perf_counter() - t0) * 1000,
        debug={"tsquery": tsq, "vector_candidates": len(vec_rows)},
    )


def get_chunk(conn: psycopg.Connection, chunk_id: str) -> Hit | None:
    row = conn.execute(f"SELECT {_COLS} FROM chunks WHERE id = %s", (chunk_id,)).fetchone()
    return Hit(*row) if row else None


def list_filters(conn: psycopg.Connection) -> dict:
    rows = conn.execute(
        "SELECT ticker, company, array_agg(year ORDER BY year) FROM documents GROUP BY ticker, company ORDER BY ticker"
    ).fetchall()
    years = sorted({y for r in rows for y in r[2]})
    return {
        "companies": [{"ticker": r[0], "name": r[1], "years": r[2]} for r in rows],
        "years": years,
    }
