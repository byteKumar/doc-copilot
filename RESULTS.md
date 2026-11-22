# Results

Only measured numbers go here, each with the command that produced it. A blank means not
measured yet.

## Spec results block

```
CORPUS: companies=___ filings=___ chunks=___                        not measured yet (needs real filings)
EVAL SET (written by Chaman): answerable=___ unanswerable=___         not written yet
BEFORE improvement -> AFTER: answer accuracy ___% -> ___% ; citation accuracy ___% -> ___%
                                                                       not measured yet (needs eval set / API key)
REFUSALS: correct refusal on unanswerable ___% ; false refusal on answerable ___%
                                                                       not measured yet (needs eval set / API key)
JUDGE CHECK: Chaman spot-checked ___ answers, agreement with LLM judge ___%
                                                                       not measured yet (needs eval set / API key)
LATENCY p50/p95: ___ / ___ ms ; COST: $___ per 1,000 queries ; total API spend $___
                                                                       not measured yet (needs eval set / API key)
DEPLOYED URL: ___ ; CI passing? ___ ; demo video: ___                  not deployed / CI passes on GitHub (tests and web build; eval subset skips until a key and eval set exist) / not recorded
IMPROVEMENT MADE AND WHY: table-aware chunking (tables kept whole, caption + header rows repeated
  in every table chunk), switchable with CHUNKING=table_aware. Built because baseline windows can
  split a financial table so the rows lose their year headers. Whether it helps is
  not measured yet (needs eval set / API key).
ANYTHING THAT FAILED OR SURPRISED YOU: ___
REPO pushed? ___                                                       yes, https://github.com/byteKumar/doc-copilot
```

How to fill the eval lines:

```bash
uv run --env-file .env doc-copilot eval run --compare baseline,improved --judge --yes
uv run doc-copilot eval agreement evals/runs/<run>/baseline/judge_spotcheck.csv
```

Report the test split if you tuned anything on dev (`--split test`), and say which.

## Measured offline (no API calls, synthetic fixture corpus)

These check the machinery. They are not results about real filings or about Claude.

| What | Number | Command |
|---|---|---|
| Test suite | 72 passed, 0 failed (26 of them against real Postgres 16 + pgvector) | `docker compose up -d db && REQUIRE_DB=1 uv run pytest -q` |
| Fixture ingestion | 3 synthetic filings -> 18 baseline chunks, 25 table-aware chunks | `uv run doc-copilot ingest --dir tests/fixtures/filings` |
| Idempotent re-ingest | second run: 3 documents unchanged, 0 chunks embedded, 0 deleted | same command, run again |
| Retrieval latency, fixture, FakeEmbedder | baseline p50 0.87 ms / p95 1.03 ms; table-aware p50 0.92 ms / p95 1.71 ms (200 runs, 18 or 25 chunks) | `uv run python scripts/bench_retrieval.py --runs 200 [--chunking table_aware]` |
| Web build | `npm run build` passes (Next.js 16.4.6) | `cd web && npm run build` |

Measured on 2026-11-21, MacBook (Apple Silicon), Postgres in Docker via Colima. Retrieval
latency with the real MiniLM embedder and a real corpus will be higher (the question has to be
embedded by a neural model, and the index is hundreds of times larger): not measured yet.
