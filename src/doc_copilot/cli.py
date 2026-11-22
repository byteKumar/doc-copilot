"""Command line entry point: `uv run doc-copilot <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .db import connection, init_db


def cmd_init_db(args, s: Settings) -> int:
    init_db(s.database_url, s.embedding_dim)
    print("schema ready")
    return 0


def cmd_download(args, s: Settings) -> int:
    from .ingest.download import EdgarClient, UserAgentError, load_corpus

    try:
        client = EdgarClient(s.sec_user_agent)
    except UserAgentError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    pairs = load_corpus(Path(args.corpus))
    out = Path(args.out)
    for ticker, year in pairs:
        try:
            path = client.download(ticker, year, out)
            print(f"ok   {ticker} {year} -> {path}")
        except Exception as e:  # keep going; report every failure
            print(f"FAIL {ticker} {year}: {e}", file=sys.stderr)
    return 0


def cmd_ingest(args, s: Settings) -> int:
    from .embed import make_embedder
    from .ingest.chunk import STRATEGIES
    from .ingest.pipeline import ingest_dir

    strategies = STRATEGIES if args.chunking == "all" else (args.chunking,)
    init_db(s.database_url, s.embedding_dim)
    embedder = make_embedder(s)
    with connection(s.database_url) as conn:
        stats = ingest_dir(
            conn, Path(args.dir), embedder, strategies=strategies,
            chunk_words=s.chunk_words, overlap_words=s.chunk_overlap_words,
        )
    print(json.dumps(stats.__dict__, indent=2))
    return 0


def cmd_search(args, s: Settings) -> int:
    from .embed import make_embedder
    from .retrieve import Filters, retrieve

    with connection(s.database_url) as conn:
        r = retrieve(
            conn, make_embedder(s), args.question, Filters(args.company, args.year),
            chunking=args.chunking or s.chunking, top_k=args.k,
            candidates=s.candidates_per_retriever, rrf_k=s.rrf_k,
        )
    print(f"best_similarity={r.best_similarity:.3f} lexical_matches={r.lexical_matches} "
          f"latency_ms={r.latency_ms:.1f}")
    for h in r.hits:
        print(f"{h.rrf_score:.4f} vec#{h.vector_rank} lex#{h.lexical_rank} {h.id} "
              f"[{h.section_title[:40]}] {h.text[:100]!r}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="doc-copilot")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db", help="create tables and indexes").set_defaults(fn=cmd_init_db)

    d = sub.add_parser("download", help="download 10-Ks from SEC EDGAR (needs SEC_USER_AGENT)")
    d.add_argument("--corpus", default="corpus.toml")
    d.add_argument("--out", default="data/filings")
    d.set_defaults(fn=cmd_download)

    i = sub.add_parser("ingest", help="parse, chunk, embed and store filings")
    i.add_argument("--dir", default="data/filings")
    i.add_argument("--chunking", default="all", choices=["all", "baseline"])
    i.set_defaults(fn=cmd_ingest)

    q = sub.add_parser("search", help="hybrid retrieval only (no LLM)")
    q.add_argument("question")
    q.add_argument("--company")
    q.add_argument("--year", type=int)
    q.add_argument("--chunking")
    q.add_argument("-k", type=int, default=8)
    q.set_defaults(fn=cmd_search)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args, Settings.from_env())


if __name__ == "__main__":
    sys.exit(main())
