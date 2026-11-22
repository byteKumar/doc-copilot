# doc-copilot

Question answering over SEC 10-K filings that answers **only from the documents, with
citations**, and says "I don't know" when the filings don't support an answer. It comes with an
eval harness that measures answer accuracy, citation accuracy, refusals, latency and cost.

Status: code, tests, CI and deployment config are done. **Not done yet:** real filings
downloaded, the eval set, any run against Claude, the deployed URL and the demo video. See
"What Chaman does next". Every number in [RESULTS.md](RESULTS.md) is blank until measured.

Demo video: _not recorded yet_ · Live demo: _not deployed yet_

## How it works

```
 SEC EDGAR ──download──> data/filings/*.html (+ .json metadata)
                              │
                  parse: Items (1, 1A, 7, 8 ...) -> text + table blocks
                  chunk: baseline windows | table-aware (tables whole, headers repeated)
                  embed: all-MiniLM-L6-v2 (local, 384-d)
                              ▼
                 Postgres: chunks(text, tsvector, vector(384), company, year, section)
                              │
 question ──> hybrid retrieval: full-text (ts_rank_cd) + vector (cosine), merged by RRF,
              optional company/year filter
                              │
              weak evidence? ──yes──> refuse (no LLM call)
                              │ no
              Claude (structured JSON: answer, citations, refused, reason)
                              │
              grounding check: citations must be retrieved chunks; numbers and key terms
              in the answer must appear in the cited chunks ──fail──> refuse
                              ▼
              answer + citations ──> FastAPI ──> Next.js chat (click a citation to read it)
```

Code map:

| Path | What |
|---|---|
| `src/doc_copilot/ingest/` | EDGAR downloader, HTML parser, chunkers, idempotent ingest |
| `src/doc_copilot/retrieve.py` | hybrid search + reciprocal-rank fusion |
| `src/doc_copilot/llm.py` | `AnthropicLLM` (paid) and `FakeLLM` (offline) behind one interface |
| `src/doc_copilot/answer.py`, `grounding.py` | prompt, refusal logic, grounding check |
| `src/doc_copilot/evals/` | eval format + validator, scoring, LLM judge, harness |
| `src/doc_copilot/api.py` | FastAPI: `/ask`, `/chunks/{id}`, `/feedback`, `/usage`, access code, rate limit |
| `web/` | Next.js chat UI and usage page |
| `evals/` | `HOW_TO_WRITE_THE_EVAL_SET.md`, `FORMAT_EXAMPLES.jsonl` (your `questions.jsonl` goes here) |
| `docs/INTERVIEW_NOTES.md` | per-milestone decisions and trade-offs |

## Safety rails on spending

- `LLM_PROVIDER` defaults to `fake` in code. Nothing paid runs unless you set `LLM_PROVIDER=anthropic`.
- The test suite deletes every `ANTHROPIC_*` variable and the paid client refuses to start inside tests.
- `eval run --dry-run` prints projected tokens and dollars; a paid eval also needs `--yes`.
- The API has a daily cap on LLM calls (`DAILY_QUERY_CAP`).

## Run it locally (offline, free)

Needs: uv, Python 3.12, Node 24, Docker.

```bash
docker compose up -d db                       # pgvector on localhost:5434
uv sync
uv run pytest                                 # 72 tests; DB tests skip if Postgres is down
uv run doc-copilot ingest --dir tests/fixtures/filings   # synthetic fixture filings
uv run doc-copilot search "net income 2024" --company ACME
uv run doc-copilot ask "How many full-time employees did Acme have?"   # FakeLLM

uv run uvicorn doc_copilot.api:app --port 8001          # API
cd web && npm install && npm run dev -- --port 3001     # UI at http://localhost:3001
```

With the fake LLM the "answers" are extractive guesses: they exercise the pipeline, they are
not Claude's answers.

## Run it on real filings

```bash
cp .env.example .env                         # fill ANTHROPIC_API_KEY and SEC_USER_AGENT yourself
cp corpus.example.toml corpus.toml           # choose 10-20 companies and the years
uv sync --extra local-embeddings             # sentence-transformers + torch (~1 GB)
uv run --env-file .env doc-copilot download  # refuses to run without SEC_USER_AGENT
uv run --env-file .env doc-copilot ingest    # both chunking strategies, idempotent
uv run --env-file .env doc-copilot ask "What was Apple's total net sales in fiscal 2024?"
```

## Evaluate

Write `evals/questions.jsonl` first (see [evals/HOW_TO_WRITE_THE_EVAL_SET.md](evals/HOW_TO_WRITE_THE_EVAL_SET.md)).

```bash
uv run doc-copilot eval validate
uv run --env-file .env doc-copilot eval run --dry-run --judge          # free estimate
uv run --env-file .env doc-copilot eval run --judge --yes              # paid
uv run --env-file .env doc-copilot eval run --compare baseline,improved --judge --yes
uv run doc-copilot eval agreement evals/runs/<run>/<label>/judge_spotcheck.csv
```

The harness refuses to report metrics from `FORMAT_EXAMPLES.jsonl` and warns when the set has
fewer than 120 questions.

## What Chaman does next

1. Put your Anthropic key and `SEC_USER_AGENT` ("Your Name your-contact") in `.env`. Confirm the model id and per-token prices in `.env` against Anthropic's pricing page.
2. Create `corpus.toml` (10-20 companies), run `download` and `ingest` (needs `--extra local-embeddings`).
3. Spot-check parsing on 2-3 real filings: `uv run doc-copilot search "risk factors" --company AAPL`. Real 10-K HTML varies; fix any parser misses before writing questions.
4. Write the eval set (~120 questions, ~90/~30, dev/test split) yourself.
5. Tune `MIN_SIMILARITY` and `TOP_K` on the **dev** split only, with `--dry-run` before each paid run.
6. Run baseline vs improved on the full set with the judge; spot-check 30 judge answers; fill RESULTS.md.
7. Write the results into CUSTOMER_BRIEF.md.
8. Push to GitHub, add the `ANTHROPIC_API_KEY` and `SEC_USER_AGENT` secrets if you want the CI eval subset, commit `corpus.toml` and `evals/questions.jsonl`.
9. Deploy (DEPLOY.md), record the 3-minute demo, link both here.
