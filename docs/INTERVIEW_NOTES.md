# Interview notes

For each milestone: what was built, why, the trade-offs, and one thing you should be able to
explain or change yourself without help. Read the code next to each section; every claim here
points at a file.

---

## 1. Filing ingestion (`src/doc_copilot/ingest/`)

**Built.** An EDGAR downloader (`download.py`), a 10-K HTML parser (`parse.py`), a fixed-window
chunker (`chunk.py`) and an idempotent ingest into Postgres (`pipeline.py`, schema in `db.py`).

**Why this way.**
- *Downloader:* SEC requires a User-Agent with your name and contact and at most 10 requests/s.
  The client refuses to run without `SEC_USER_AGENT` and sleeps between requests. "Year" is the
  fiscal year (`reportDate`), not the filing date: Apple's FY2024 10-K is filed in Nov 2024 but a
  retailer's FY2024 10-K is filed in March 2025. Amendments (10-K/A) are skipped.
- *Parser:* walks the DOM in order and emits text blocks and table blocks. A short block that
  matches "Item 1A. Risk Factors" starts a section. The table of contents repeats every heading,
  so for duplicates we keep the occurrence with the most content. Tables get their split
  `$`, `%` and `)` cells merged so "$ | 1,184.2" becomes "$ 1,184.2".
- *Chunking:* windows of 220 words with 40 overlapping. Overlap means a sentence cut at a
  boundary still appears whole in one chunk.
- *Idempotency:* a document hash decides whether to re-parse at all; a per-chunk hash decides
  whether to re-embed. Chunk ids are stable (`AAPL-2024-7-b012`), so re-ingest is an upsert plus
  deleting ids that disappeared.
- *Embeddings:* local MiniLM (free, 384-d, runs on CPU). Each chunk is embedded with a header
  "Company (TICKER) YEAR 10-K. Section." so a chunk that never names the company still matches.

**Trade-offs.**
- Regex headings are brittle: some filings use images, odd spacing or "ITEM 7 -". Tested on
  synthetic fixtures only; real filings need a spot check.
- Words are a proxy for tokens (about 1.3 tokens per English word). Good enough for sizing.
- MiniLM is weaker than paid embedding models on financial jargon; hybrid search compensates.
- Tests use a hashing `FakeEmbedder`: deterministic and fast, but it only measures word overlap.

**Be able to:** add a new Item pattern to `ITEM_RE` (e.g. "ITEM 7 —") and write the failing
test first in `tests/test_ingest.py`.

---

## 2. Hybrid retrieval (`src/doc_copilot/retrieve.py`)

**Built.** Two queries per question: Postgres full-text (`tsvector`, `ts_rank_cd`) and pgvector
cosine similarity, each returning 30 candidates, merged with reciprocal-rank fusion
(`1/(60 + rank)` summed over both lists), plus optional company and year filters.

**Why.** Keywords win on exact terms ("Item 1A", "lidar", a segment name); vectors win on
paraphrase ("how many people work there" vs "full-time employees"). Scores from the two are on
different scales, so adding them is meaningless; adding reciprocal ranks is scale-free.
Full-text uses OR of the question's words because natural questions rarely contain every term
of the answer (`websearch_to_tsquery` would AND them and often match nothing).

**Trade-offs.**
- RRF ignores how confident each retriever is. A re-ranker (cross-encoder) would be the next step
  but costs latency.
- k=60 is the paper's default, not tuned.
- HNSW is approximate; at this corpus size (thousands of chunks) exact search would also be fast.
- Postgres FTS is not BM25 (no document-length normalization by default). Close enough here and
  it avoids running a second search engine.

**Be able to:** explain why the chunk ranked first by only one retriever can lose to one ranked
second by both (see `test_rrf_rewards_agreement`), and change `candidates_per_retriever`.

---

## 3. Grounded answering (`answer.py`, `grounding.py`, `llm.py`)

**Built.** Retrieval -> Claude with a JSON schema (answer, citations, refused, reason) ->
deterministic grounding check -> answer or refusal. Three refusal paths: weak retrieval
(no LLM call, saves money), the model declines, or the grounding check fails.

**Why.**
- *Structured output* makes citations machine-checkable; free text "[1]" markers would need parsing.
- *Grounding check* catches the expensive failure (a confident wrong number): every cited id
  must be one we sent, every non-trivial number in the answer must appear in a cited chunk
  (allowing unit scale and rounding: "$1.6 billion" matches "1,587.1" in millions), and at least
  40% of the answer's key words must appear in the cited text.
- *One LLM interface* (`complete_json`) with a real and a fake implementation: tests and CI never
  touch the paid API, and you can swap models by changing an env var.
- Errors map to specific types (rate limit, unavailable, bad response) and become a refusal
  with a reason instead of a 500.

**Trade-offs.**
- The number check rejects correct derived numbers (e.g. a computed 23.5% growth rate that is not
  printed in the filing). That raises false refusals on comparison questions. You can measure it.
- The similarity floor (`MIN_SIMILARITY`) trades false refusals against wasted LLM calls; it must
  be tuned on the dev split.
- The term-coverage check is crude and could be gamed by echoing the passage.

**Be able to:** relax the number check to allow percentages computed from two cited numbers, and
show the change in false-refusal rate on dev questions.

---

## 4. Eval harness (`src/doc_copilot/evals/`)

**Built.** JSONL format + validator, a deterministic subset for CI, scoring (automatic number/text
match, LLM judge, citation-contains-answer, section match), refusal rates, p50/p95 latency, cost
per 1,000 queries, a dry-run cost estimate, a `--yes` gate for paid runs, a spot-check CSV and an
agreement + Cohen's kappa calculator.

**Why.**
- Accuracy is computed over *all* answerable questions, so a refusal counts as wrong. Otherwise a
  system that refuses everything hard would look accurate.
- Two accuracy numbers: automatic match is strict and reproducible; the judge handles paraphrase.
  The judge is itself a model, so you check it by hand on 30 answers and report agreement.
- Kappa matters because raw agreement is inflated when most answers are correct.
- The dry run builds the real prompts (retrieval is free) and counts characters, so the estimate
  reflects actual context sizes.
- The harness refuses to report metrics from FORMAT_EXAMPLES and warns under 120 questions, so a
  smoke run can't be mistaken for a result.

**Trade-offs.** Character-based token estimates are approximate (deliberately pessimistic at 3.5
chars/token). Citation accuracy checks the gold number appears in a cited chunk, which can pass if
the same number appears elsewhere for a different reason.

**Be able to:** add a new metric (e.g. accuracy by `answer_type`) to `summarize()` and the table.

---

## 5. Table-aware chunking (`chunk.py`, `--compare baseline,improved`)

**Built.** A second chunking strategy stored side by side with the baseline (`chunks.chunking`).
Prose is windowed as before but never across a table; each table becomes its own chunk(s), and
every table chunk repeats the caption line before it and the column header rows.

**Why.** In the baseline, a window boundary can fall inside a financial table, and the second half
loses the "2024 | 2023" header row. The model then sees "Net income | $ 182.4 | $ 77.0" without
knowing which column is which year. `test_baseline_can_split_table_from_its_header` shows the
failure on the fixture. Storing both strategies lets one eval run compare them on identical
questions, and `flips()` lists which questions changed.

**Not measured yet.** Whether it improves accuracy on real filings is unknown until you run
`--compare baseline,improved`. If it doesn't help, report that; a null result is still a finding.

**Trade-offs.** More chunks (25 vs 18 on the fixture), duplicated headers use more tokens per
prompt, and very wide tables still overflow.

**Be able to:** decide from the eval breakdown whether the next improvement should be re-ranking
or query rewriting, and justify it with numbers.

---

## 6. Chat UI (`web/`, `api.py`)

**Built.** FastAPI endpoints (`/ask`, `/chunks/{id}`, `/filters`, `/feedback`) and a Next.js App
Router client: filters, chat, citation chips that open the source passage in a side panel,
refusals shown with their reason, thumbs up/down stored in Postgres against the logged query id.

**Why.** Showing the source passage is the trust feature: a user can verify an answer in one
click. Feedback rows link to `query_log`, so thumbs-down answers can be turned into new eval
questions.

**Trade-offs.** Client-side fetching (simple, no server state) means the API URL is public and the
access code sits in the browser's localStorage. Fine for a demo.

**Be able to:** add a "copy answer with citations" button, or show the answer's cost and latency
differently.

---

## 7. Deployment prep and CI (`Dockerfile`, `web/Dockerfile`, `render.yaml`, `.github/workflows/ci.yml`)

**Built.** Access-code middleware (constant-time compare, `/health` open, CORS headers still sent
on 401), an in-memory sliding-window rate limiter per client IP, a daily cap on LLM calls read from
`query_log`, a `/usage` endpoint and page, Dockerfiles, a compose `app` profile, a Render
blueprint and DEPLOY.md. CI runs ruff + pytest against a pgvector service container + the web
build, and a separate eval-subset job that runs only when the API key and SEC user-agent secrets
exist and the eval set is committed.

**Trade-offs.**
- The rate limiter is per process and forgets on restart; with two instances, a client gets twice
  the limit. Postgres or Redis would fix it.
- The daily cap is the real cost control. Worst-case daily spend = cap x cost per query.
- CPU torch makes the API image large and needs ~1 GB RAM; an ONNX embedder would be lighter.

**Be able to:** move the rate limiter into Postgres (one table, one upsert per request) and say
what that costs in latency.

---

## 8. Results and brief

RESULTS.md and CUSTOMER_BRIEF.md have the structure; the numbers are yours to measure. In an
interview, the strongest answers come from your own failure analysis: open
`evals/runs/<run>/<label>/results.jsonl`, sort by wrong answers, and group them (retrieval missed,
parsing broke a table, model refused wrongly, grounding rejected a right answer). Have two
concrete examples ready.
