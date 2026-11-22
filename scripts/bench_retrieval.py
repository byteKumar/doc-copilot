"""Time hybrid retrieval (no LLM) over whatever is ingested in DATABASE_URL.

uv run python scripts/bench_retrieval.py --runs 200
Reports p50/p95 of retrieve() wall time, which includes embedding the question.
"""

import argparse
import statistics

from doc_copilot.config import Settings
from doc_copilot.db import connection
from doc_copilot.embed import make_embedder
from doc_copilot.retrieve import retrieve

QUESTIONS = [
    "What was total revenue?",
    "How many full-time employees does the company have?",
    "What are the main supplier risks?",
    "What was net income in the most recent year?",
    "Where is the company headquartered?",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--chunking", default="baseline")
    args = p.parse_args()
    s = Settings.from_env()
    emb = make_embedder(s)
    times = []
    with connection(s.database_url) as conn:
        n_chunks = conn.execute("SELECT count(*) FROM chunks WHERE chunking=%s", (args.chunking,)).fetchone()[0]
        for _ in range(5):  # warm-up
            retrieve(conn, emb, QUESTIONS[0], chunking=args.chunking)
        for i in range(args.runs):
            r = retrieve(conn, emb, QUESTIONS[i % len(QUESTIONS)], chunking=args.chunking, top_k=s.top_k)
            times.append(r.latency_ms)
    q = statistics.quantiles(times, n=100, method="inclusive")
    print(f"embedder={s.embedder} chunking={args.chunking} chunks={n_chunks} runs={args.runs} "
          f"p50={q[49]:.2f} ms p95={q[94]:.2f} ms")


if __name__ == "__main__":
    main()
