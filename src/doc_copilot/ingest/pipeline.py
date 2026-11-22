"""Ingest a filing: parse -> chunk -> embed -> upsert into Postgres.

Idempotent: re-ingesting the same file changes nothing and embeds nothing. A chunk is
re-embedded only if its content hash changed; chunks that no longer exist are deleted.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

from ..embed import Embedder
from .chunk import STRATEGIES, Chunk, chunk_filing
from .parse import parse_filing

FILENAME_RE = re.compile(r"^(?P<ticker>[A-Z0-9.\-]+)_(?P<year>\d{4})_(?P<form>10-K)\.html?$")


@dataclass
class FilingMeta:
    ticker: str
    company: str
    year: int
    form: str = "10-K"
    source_url: str | None = None


@dataclass
class IngestStats:
    documents_seen: int = 0
    documents_unchanged: int = 0
    chunks_total: int = 0
    chunks_embedded: int = 0
    chunks_deleted: int = 0
    per_strategy: dict[str, int] = field(default_factory=dict)


def load_meta(path: Path) -> FilingMeta:
    """Metadata comes from a sidecar JSON written by the downloader (TICKER_YEAR_10-K.json)."""
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        d = json.loads(sidecar.read_text())
        return FilingMeta(
            ticker=d["ticker"].upper(),
            company=d["company"],
            year=int(d["year"]),
            form=d.get("form", "10-K"),
            source_url=d.get("source_url"),
        )
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"{path.name}: expected TICKER_YEAR_10-K.html or a .json sidecar")
    return FilingMeta(ticker=m["ticker"], company=m["ticker"], year=int(m["year"]))


def embedding_input(meta: FilingMeta, chunk: Chunk) -> str:
    # Company/year/section go into the embedded text so "Acme risk factors" can match a chunk
    # whose body never repeats the company name.
    return f"{meta.company} ({meta.ticker}) {meta.year} 10-K. {chunk.section_title}.\n{chunk.text}"


def _upsert_document(conn: psycopg.Connection, meta: FilingMeta, path: Path, content_hash: str) -> tuple[int, bool]:
    row = conn.execute(
        "SELECT id, content_hash FROM documents WHERE ticker=%s AND year=%s AND form=%s",
        (meta.ticker, meta.year, meta.form),
    ).fetchone()
    if row and row[1] == content_hash:
        return row[0], False
    if row:
        conn.execute(
            "UPDATE documents SET content_hash=%s, source_path=%s, source_url=%s, company=%s, ingested_at=now() WHERE id=%s",
            (content_hash, str(path), meta.source_url, meta.company, row[0]),
        )
        return row[0], True
    doc_id = conn.execute(
        "INSERT INTO documents (ticker, company, year, form, source_path, source_url, content_hash) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (meta.ticker, meta.company, meta.year, meta.form, str(path), meta.source_url, content_hash),
    ).fetchone()[0]
    return doc_id, True


def ingest_file(
    conn: psycopg.Connection,
    path: Path,
    embedder: Embedder,
    strategies: tuple[str, ...] = STRATEGIES,
    chunk_words: int = 220,
    overlap_words: int = 40,
    meta: FilingMeta | None = None,
    batch_size: int = 64,
) -> IngestStats:
    stats = IngestStats(documents_seen=1)
    meta = meta or load_meta(path)
    raw = path.read_bytes()
    content_hash = hashlib.sha256(
        raw + f"|{chunk_words}|{overlap_words}".encode()
    ).hexdigest()
    html = raw.decode("utf-8", errors="replace")

    with conn.transaction():
        doc_id, changed = _upsert_document(conn, meta, path, content_hash)
        have = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT chunking FROM chunks WHERE document_id=%s", (doc_id,)
            ).fetchall()
        }
        if not changed and set(strategies) <= have:
            stats.documents_unchanged = 1
            for s in strategies:
                n = conn.execute(
                    "SELECT count(*) FROM chunks WHERE document_id=%s AND chunking=%s", (doc_id, s)
                ).fetchone()[0]
                stats.per_strategy[s] = n
                stats.chunks_total += n
            return stats

        sections = parse_filing(html)
        for strategy in strategies:
            chunks = chunk_filing(sections, meta.ticker, meta.year, strategy, chunk_words, overlap_words)
            existing = dict(
                conn.execute(
                    "SELECT id, content_hash FROM chunks WHERE document_id=%s AND chunking=%s",
                    (doc_id, strategy),
                ).fetchall()
            )
            todo = [c for c in chunks if existing.get(c.id) != c.content_hash]
            for i in range(0, len(todo), batch_size):
                batch = todo[i : i + batch_size]
                vectors = embedder.embed([embedding_input(meta, c) for c in batch])
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO chunks (id, document_id, chunking, ticker, company, year, section,
                            section_title, anchor, chunk_index, kind, text, content_hash, embedding)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
                        ON CONFLICT (id) DO UPDATE SET
                            section_title=EXCLUDED.section_title, anchor=EXCLUDED.anchor,
                            chunk_index=EXCLUDED.chunk_index, kind=EXCLUDED.kind, text=EXCLUDED.text,
                            content_hash=EXCLUDED.content_hash, embedding=EXCLUDED.embedding,
                            company=EXCLUDED.company
                        """,
                        [
                            (c.id, doc_id, strategy, meta.ticker, meta.company, meta.year, c.section,
                             c.section_title, c.anchor, c.chunk_index, c.kind, c.text, c.content_hash,
                             str(v))
                            for c, v in zip(batch, vectors, strict=True)
                        ],
                    )
            stats.chunks_embedded += len(todo)
            stale = set(existing) - {c.id for c in chunks}
            if stale:
                conn.execute("DELETE FROM chunks WHERE id = ANY(%s)", (list(stale),))
                stats.chunks_deleted += len(stale)
            stats.per_strategy[strategy] = len(chunks)
            stats.chunks_total += len(chunks)
    return stats


def ingest_dir(conn: psycopg.Connection, directory: Path, embedder: Embedder, **kw) -> IngestStats:
    total = IngestStats()
    for path in sorted(directory.glob("*_10-K.htm*")):
        s = ingest_file(conn, path, embedder, **kw)
        total.documents_seen += 1
        total.documents_unchanged += s.documents_unchanged
        total.chunks_total += s.chunks_total
        total.chunks_embedded += s.chunks_embedded
        total.chunks_deleted += s.chunks_deleted
        for k, v in s.per_strategy.items():
            total.per_strategy[k] = total.per_strategy.get(k, 0) + v
    return total
