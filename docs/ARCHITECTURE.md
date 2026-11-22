# doc-copilot architecture

A study guide to how the code works. It goes deeper than the [README](../README.md): what each component is responsible for, what each module does and its key functions, the full data flow for ingestion and for a query, the exact prompt and JSON schema, how every eval metric is computed, and the trade-offs.

Every claim here points at a file in this repo. If the code and this document ever disagree, the code wins: fix this document.

Contents:

1. [Components](#1-components)
2. [Configuration](#2-configuration)
3. [Module walkthrough](#3-module-walkthrough)
4. [Ingestion data flow](#4-ingestion-data-flow)
5. [Query data flow](#5-query-data-flow)
6. [Prompt and JSON schema](#6-prompt-and-json-schema)
7. [The grounding check in detail](#7-the-grounding-check-in-detail)
8. [Eval harness: how each metric is computed](#8-eval-harness-how-each-metric-is-computed)
9. [API and web UI](#9-api-and-web-ui)
10. [Testing strategy](#10-testing-strategy)
11. [Trade-offs and known gaps](#11-trade-offs-and-known-gaps)

---

## 1. Components

| Component | Responsibility | Code |
|---|---|---|
| Downloader | Fetch 10-K HTML from SEC EDGAR by ticker and fiscal year, write a JSON sidecar with metadata | `src/doc_copilot/ingest/download.py` |
| Parser | Turn 10-K HTML into Items (sections) made of text and table blocks | `src/doc_copilot/ingest/parse.py` |
| Chunker | Cut sections into chunks with two strategies, `baseline` and `table_aware` | `src/doc_copilot/ingest/chunk.py` |
| Embedder | Turn text into 384-dimension vectors (real MiniLM or an offline hashing fake) | `src/doc_copilot/embed.py` |
| Ingest pipeline | Parse, chunk, embed and upsert, idempotently | `src/doc_copilot/ingest/pipeline.py` |
| Database | Postgres 16 with pgvector: documents, chunks, query log, feedback | `src/doc_copilot/db.py` |
| Retriever | Full-text search plus vector search, merged with reciprocal rank fusion | `src/doc_copilot/retrieve.py` |
| LLM client | One `complete_json` interface over Claude (paid) and a fake (offline) | `src/doc_copilot/llm.py` |
| Answer service | Retrieve, decide whether to call the LLM, check grounding, build the response, log it | `src/doc_copilot/answer.py` |
| Grounding check | Deterministic checks that an answer is supported by the chunks it cites | `src/doc_copilot/grounding.py` |
| API | HTTP endpoints, access code, rate limit, daily cap, usage stats | `src/doc_copilot/api.py` |
| CLI | `doc-copilot` command for every offline task | `src/doc_copilot/cli.py` |
| Eval harness | Load and validate the eval set, run it, score it, estimate cost, compare configs | `src/doc_copilot/evals/` |
| Web UI | Chat with filters, citation side panel, feedback, usage page | `web/` |

```mermaid
flowchart LR
    subgraph ING["Ingestion (CLI)"]
        EDGAR["SEC EDGAR"] --> DL["download.py<br/>10-K HTML + JSON sidecar"]
        DL --> PARSE["parse.py<br/>Items, text and table blocks"]
        PARSE --> CHUNK["chunk.py<br/>baseline and table_aware"]
        CHUNK --> EMB["embed.py<br/>MiniLM 384-d or FakeEmbedder"]
    end
    EMB --> PG[("Postgres 16 + pgvector<br/>documents, chunks")]
    subgraph QRY["Query path"]
        API["api.py (FastAPI)<br/>access code, rate limit, daily cap"] --> ANS["answer.py<br/>Answerer"]
        ANS --> RET["retrieve.py<br/>full-text + vector, RRF"]
        ANS --> LLM["llm.py<br/>AnthropicLLM or FakeLLM"]
        ANS --> GR["grounding.py<br/>citation, number, term checks"]
        ANS --> LOG[("query_log, feedback")]
    end
    RET --> PG
    LLM --> CLAUDE["Claude API"]
    USER["User"] --> WEB["Next.js web UI"] --> API
    EVAL["evals/harness.py<br/>eval set, scoring, judge"] --> ANS
    EVAL --> LLM
```

Two design rules run through the whole codebase:

- **Offline by default.** `Settings` defaults to `llm_provider="fake"` and `embedder="fake"`. Real Claude runs only when `LLM_PROVIDER=anthropic` is set explicitly.
- **One code path.** The API, the CLI `ask` command and the eval harness all call the same `Answerer.answer()`, so eval numbers describe the code users hit.

## 2. Configuration

All runtime settings come from environment variables, read once into the frozen dataclass `Settings` (`config.py`, `Settings.from_env()`). Nothing in the Python code reads `.env`; you pass it with `uv run --env-file .env ...`. An empty variable counts as unset.

| Variable | Default in code | What it controls |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5434/doc_copilot` | Postgres connection |
| `EMBEDDER` | `fake` | `local` (MiniLM) or `fake` (hashing) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model for `local` |
| `EMBEDDING_DIM` | `384` | Vector column size at schema creation |
| `LLM_PROVIDER` | `fake` | `anthropic` (paid) or `fake` |
| `LLM_MODEL` | `claude-haiku-5-5` | Claude model id |
| `LLM_MAX_TOKENS` | `4000` | Output token limit per call |
| `LLM_PRICE_INPUT_PER_MTOK` | `0.10` | USD per 1M input tokens, for cost math |
| `LLM_PRICE_OUTPUT_PER_MTOK` | `0.50` | USD per 1M output tokens |
| `CHUNKING` | `baseline` | Strategy queried at runtime (`baseline` or `table_aware`) |
| `CHUNK_WORDS` / `CHUNK_OVERLAP_WORDS` | `220` / `40` | Window size and overlap, in words |
| `TOP_K` | `8` | Chunks sent to the model |
| `CANDIDATES_PER_RETRIEVER` | `30` | Candidates from each of full-text and vector search |
| `RRF_K` | `60` | The k in 1 / (k + rank) |
| `MIN_SIMILARITY` | `0.25`, or `0.0` when `EMBEDDER=fake` | Refuse without calling the LLM below this best cosine similarity |
| `ACCESS_CODE` | empty (off) | Shared code required in `X-Access-Code` |
| `RATE_LIMIT_PER_MINUTE` | `10` | Per client IP, on `/ask` |
| `DAILY_QUERY_CAP` | `500` | LLM calls per rolling 24 hours, all users |
| `CORS_ORIGINS` | `http://localhost:3001` | Comma-separated allowed origins |
| `SEC_USER_AGENT` | empty | "Name contact" string EDGAR requires |

Other variables: `ANTHROPIC_API_KEY` (read by `AnthropicLLM`), `DOC_COPILOT_TESTING` (set by the tests to block the paid client), `TEST_DATABASE_URL` and `REQUIRE_DB` (tests), `NEXT_PUBLIC_API_URL` (web, read at build time), `PORT` (API Docker image).

`Settings.cost_usd(input_tokens, output_tokens)` is the single place dollar cost is computed: `(input * price_in + output * price_out) / 1,000,000`.

## 3. Module walkthrough

### `config.py`
- `Settings`: frozen dataclass of every setting, with safe defaults.
- `Settings.from_env()`: reads the variables in section 2. Turns the similarity floor off by default for the fake embedder, because its similarities mean nothing.
- `Settings.cost_usd()`: tokens to dollars.

### `db.py`
- `schema_sql(dim)`: the DDL for `documents`, `chunks`, `query_log`, `feedback` and their indexes. `chunks.tsv` is a generated column: `to_tsvector('english', section_title || ' ' || text)`.
- `init_db(url, dim)`: creates the `vector` extension first (so `register_vector` can find the type), then runs the schema. Safe to run repeatedly (`IF NOT EXISTS`).
- `connect()` / `connection()`: autocommit connection with pgvector types registered.

### `ingest/download.py`
- `validate_user_agent(ua)`: rejects empty values, single words and placeholders like "your name" or "example.com".
- `EdgarClient`: an `httpx` client that sleeps so requests are at least 0.15 s apart and sends the User-Agent on every call.
  - `lookup(ticker)`: ticker to (CIK, company name) via `company_tickers.json`.
  - `list_10k(ticker)`: reads `data.sec.gov/submissions/CIK##########.json` and keeps form `10-K` only (amendments skipped).
  - `download(ticker, year, out_dir)`: picks the latest-filed 10-K whose `reportDate` year equals `year`, writes `TICKER_YEAR_10-K.html` (skipped if present) and a `.json` sidecar with `ticker`, `company`, `year`, `cik`, `accession`, `report_date`, `filing_date`, `source_url`.
- `load_corpus(path)`: reads `corpus.toml` (`tickers`, `years`) into every (ticker, year) pair.

### `ingest/parse.py`
- `clean_text()`: normalises non-breaking spaces, curly quotes and whitespace.
- `extract_blocks(html)`: removes `script`, `style`, `head`, `noscript`, `ix:header` and `display:none` elements, then walks the DOM in document order. Leaf block elements become text blocks; each `<table>` becomes one table block. Tracks the latest `id` or `name` attribute as the block's anchor (used for "open on sec.gov" links).
- `_table_rows()`: merges cells that hold only `$`, `(`, `)`, `%` into their neighbours, so "$ | 1,184.2" becomes "$ 1,184.2".
- `table_block()`: rows joined with ` | `. Leading rows are headers until the first data row (a row with a digit in a non-first cell that isn't a year), at most 3.
- `_heading(block, next_block)`: is this block an Item heading? Uses `ITEM_RE` on short blocks (up to 220 characters), also accepts a one-row table heading, takes the title from the next block when the heading line has none, and rejects lines whose title starts lowercase ("Item 7 of this report...").
- `split_sections(blocks)`: starts with a "Cover page" section (Item `0`, tables dropped), opens a new section at each heading, and for repeated Items (the table of contents) keeps the occurrence with the most words.
- `parse_filing(html)`: `split_sections(extract_blocks(html))`.

### `ingest/chunk.py`
- `STRATEGIES = ("baseline", "table_aware")`.
- `_windows(text, size, overlap)`: fixed windows of `size` words, stepping `size - overlap`.
- `chunk_section(section, strategy, size, overlap)`: returns `(kind, text, anchor)` tuples.
  - `baseline`: all blocks joined into one stream and windowed. A window is `kind="mixed"` if the section has a table and the window contains `|`, else `"text"`.
  - `table_aware`: prose collected and windowed between tables (`kind="text"`); each table goes through `_table_chunks` (`kind="table"`).
- `_caption_for(blocks, i)`: the text block right before a table, if it is 60 words or fewer.
- `_table_chunks(block, caption, size)`: splits table body rows into pieces that fit `size` minus the prefix (minimum budget 20 words), and prefixes every piece with the caption and header rows.
- `chunk_filing(sections, ticker, year, strategy, size, overlap)`: builds `Chunk` objects with ids `{TICKER}-{YEAR}-{ITEM}-{b|t}{index:03d}`. `Chunk.content_hash` is SHA-256 of `section_title + "\n" + text`.

### `embed.py`
- `Embedder` protocol: `dim` and `embed(texts) -> list[list[float]]`.
- `FakeEmbedder`: feature hashing of words and word bigrams into 384 buckets (BLAKE2b), signed, L2-normalised. Deterministic, no download, measures word overlap only.
- `LocalEmbedder`: `SentenceTransformer(model, device="cpu")`, batch size 32, normalised embeddings. Needs `uv sync --extra local-embeddings`.
- `make_embedder(settings)`: picks one by `EMBEDDER`.

### `ingest/pipeline.py`
- `load_meta(path)`: metadata from the `.json` sidecar, or from the file name `TICKER_YEAR_10-K.html` as a fallback (company = ticker).
- `embedding_input(meta, chunk)`: `"{company} ({ticker}) {year} 10-K. {section_title}.\n{text}"`. Only the vector sees this header; the stored `text` is the plain chunk.
- `ingest_file(conn, path, embedder, strategies, chunk_words, overlap_words)`: the idempotent ingest described in section 4. Returns `IngestStats`.
- `ingest_dir(conn, directory, embedder, **kw)`: runs `ingest_file` on every `*_10-K.htm*` in sorted order and sums the stats.

### `retrieve.py`
- `Filters(company, year)`: `company` matches the ticker exactly (upper-cased) or any part of the company name (`ILIKE`).
- `or_tsquery(question)`: lowercase `[a-z0-9]+` words longer than one character, de-duplicated, joined with ` | `. Postgres `to_tsquery('english', ...)` then drops stop words and stems.
- `rrf_merge(ranked_lists, k)`: `score[id] += 1 / (k + rank)` over each list.
- `retrieve(conn, embedder, question, filters, chunking, top_k, candidates, rrf_k)`: embeds the question, runs the vector query and the full-text query (section 5), merges, returns a `RetrievalResult` with `hits` (each `Hit` keeps its vector rank and similarity, lexical rank and score, and RRF score), `best_similarity`, `lexical_matches`, `latency_ms`.
- `get_chunk(conn, id)`, `list_filters(conn)`: used by `/chunks/{id}` and `/filters`.

### `llm.py`
- `LLM` protocol: `model` and `complete_json(system, user, schema, task) -> LLMResult`.
- `LLMResult`: `data`, `input_tokens`, `output_tokens`, `model`, `stop_reason`.
- Errors: `LLMError` (base), `LLMRateLimited` and `LLMUnavailable` (marked `retryable = True`), `LLMBadResponse`.
- `AnthropicLLM`: refuses to build if `DOC_COPILOT_TESTING=1` or `ANTHROPIC_API_KEY` is missing. Calls `client.messages.create(model, max_tokens, system, messages, output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})` with `max_retries=2`. Maps `RateLimitError` to `LLMRateLimited`, status 5xx or 529 and connection errors to `LLMUnavailable`, other status errors to `LLMError`. Raises `LLMBadResponse` on `stop_reason` `refusal` or `max_tokens`, a missing text block, or invalid JSON. Input tokens include cache-creation and cache-read tokens.
- `FakeLLM`: for task `answer`, splits each `<chunk>` in the prompt into sentences and returns the sentence with the highest share of the question's content words (refuses below 0.5). For task `judge`, marks correct if at least 60% of the gold answer's content words appear in the model answer. Tokens are estimated at 0.25 per character. `script(task, *responses)` lets a test queue exact replies.
- `make_llm(settings)`: picks one by `LLM_PROVIDER`.

### `answer.py`
- `SYSTEM_PROMPT`, `ANSWER_SCHEMA`, `build_prompt()`: section 6.
- `REFUSAL_TEXT`: "I don't know. The filings I searched don't support an answer to that."
- `Answerer(settings, embedder, llm, chunking)`: holds the dependencies. `answer(conn, question, filters, log=True)` runs the query flow in section 5 and returns an `AnswerResponse`.
- `AnswerResponse` fields: `query_id`, `question`, `answer`, `citations` (each with id, company, ticker, year, section, section_title and a 280-character snippet), `refused`, `refusal_reason`, `latency_ms`, `retrieval_ms`, `input_tokens`, `output_tokens`, `cost_usd`, `llm_called`, `grounding_issues`, `retrieved_ids`, `best_similarity`, `model_answer`.
- `log_query(conn, response, filters)`: one `query_log` row per answer.

### `grounding.py`
- `NUM_RE`, `extract_numbers(text) -> list[Number]`: finds numbers with optional `$`, parentheses, commas, decimals and a unit (`%`, `percent`, `billion`, `bn`, `million`, `mm`, `m`, `thousand`, `k`). Each `Number` keeps value, decimal places, scale and whether it is a percentage.
- `_trivial(n)`: whole numbers under 10 and years 1990 to 2035 (no unit, no decimals) are skipped.
- `number_supported(n, evidence)`: section 7.
- `check_grounding(answer, citations, retrieved, question, min_term_coverage=0.4) -> GroundingResult`: section 7.

### `api.py`
- `create_app(settings, embedder, llm)`: builds the FastAPI app. The lifespan runs `init_db`, opens a `psycopg_pool.ConnectionPool` (1 to 5 connections, autocommit, pgvector registered) and builds one `Answerer`. Injectable dependencies make the API testable with fakes.
- `RateLimiter(limit, window_s=60)`: a sliding window of timestamps per key in a `deque`, guarded by a lock. `check()` returns 0 or the seconds to wait.
- `client_key(request)`: first `X-Forwarded-For` entry, else the socket address.
- `AccessCodeMiddleware`: section 9.
- `usage_stats(conn, daily_cap)`: totals, refusal rate, LLM calls in 24 hours, p50/p95 latency (`percentile_cont`), cost totals and cost per 1,000 queries, token totals, feedback counts, a 7-day breakdown and the 20 most recent questions.
- Module-level `__getattr__("app")`: `uvicorn doc_copilot.api:app` builds the app lazily, so importing the module in tests reads no environment and opens no connection.

### `cli.py`
Entry point `doc-copilot = doc_copilot.cli:main` (from `pyproject.toml`). Subcommands:

| Command | What it does |
|---|---|
| `init-db` | Create tables and indexes |
| `download [--corpus corpus.toml] [--out data/filings]` | Download every (ticker, year) pair; exits 2 without a valid `SEC_USER_AGENT` |
| `ingest [--dir data/filings] [--chunking all\|baseline\|table_aware]` | Parse, chunk, embed, store; prints `IngestStats` as JSON |
| `search QUESTION [--company] [--year] [--chunking] [-k 8]` | Retrieval only, prints RRF score and both ranks per hit |
| `ask QUESTION [--company] [--year] [--chunking] [--json]` | Full answer with citations, latency, tokens, cost |
| `eval validate [--questions]` | Check the eval JSONL; exits 2 on errors |
| `eval run [...]` | Run and score the eval set (section 8) |
| `eval agreement CSV` | Human vs. judge agreement and Cohen's kappa |

### `evals/dataset.py`, `evals/scoring.py`, `evals/harness.py`
Covered in section 8.

## 4. Ingestion data flow

```mermaid
flowchart TD
    CORPUS["corpus.toml<br/>tickers x fiscal years"] --> LOOKUP["EdgarClient.lookup<br/>ticker to CIK"]
    LOOKUP --> LIST["list_10k<br/>submissions JSON, form 10-K only"]
    LIST --> PICK["Pick reportDate year = requested year"]
    PICK --> SAVE["Save TICKER_YEAR_10-K.html<br/>and .json sidecar"]
    SAVE --> HASH{"File hash changed, or a<br/>strategy missing?"}
    HASH -->|no| SKIP["Skip, report unchanged"]
    HASH -->|yes| BLOCKS["extract_blocks<br/>text and table blocks"]
    BLOCKS --> SECTIONS["split_sections<br/>Items, TOC duplicates resolved"]
    SECTIONS --> CHUNKS["chunk_filing<br/>baseline and table_aware"]
    CHUNKS --> DIFF{"Chunk hash differs<br/>from stored?"}
    DIFF -->|yes| EMBED["Embed with header<br/>batches of 64"]
    EMBED --> UPSERT["INSERT ... ON CONFLICT (id) DO UPDATE"]
    DIFF -->|no| KEEP["Keep stored row and vector"]
    UPSERT --> STALE["Delete chunk ids that no longer exist"]
    KEEP --> STALE
```

Step by step, for `doc-copilot ingest --dir data/filings`:

1. `cmd_ingest` runs `init_db`, builds the embedder, and calls `ingest_dir` with both strategies (unless `--chunking` picks one).
2. For each `*_10-K.htm*` file, `ingest_file`:
   1. Loads metadata from the sidecar.
   2. Computes the document hash: SHA-256 of the raw bytes plus `|chunk_words|overlap_words`. Changing the window size therefore re-processes every file.
   3. Opens a transaction. `_upsert_document` inserts the `documents` row, or updates it if the hash changed, keyed by `(ticker, year, form)`.
   4. If the hash is unchanged and every requested strategy already has chunks, it counts the stored chunks and returns. Nothing is parsed or embedded.
   5. Otherwise it parses the file once and, for each strategy, builds the chunks, loads the stored `(id, content_hash)` pairs, embeds only chunks that are new or whose hash changed, upserts them in batches of 64, and deletes ids that disappeared.
3. The CLI prints `documents_seen`, `documents_unchanged`, `chunks_total`, `chunks_embedded`, `chunks_deleted` and `per_strategy`.

Notes for interviews:

- **Why stable ids.** `ACME-2024-7-t003` stays the same across re-ingests, so re-ingest is an upsert, `query_log.citations` stay meaningful, and eval gold data can refer to chunks.
- **Why a per-chunk hash.** Fixing a parser bug in one section re-embeds only the chunks that changed. Embedding is the slow step on CPU.
- **Gap.** Neither hash includes the embedder name. Switching `EMBEDDER` on a populated database does not re-embed; start from an empty database instead.

## 5. Query data flow

```mermaid
sequenceDiagram
    actor U as User
    participant W as Web UI
    participant A as FastAPI /ask
    participant S as Answerer
    participant DB as Postgres
    participant C as Claude
    participant G as Grounding check
    U->>W: Question, optional company and year
    W->>A: POST /ask with X-Access-Code header
    opt Over the per-IP rate limit or the daily LLM cap
        A-->>W: 429 with Retry-After, or 503
    end
    A->>S: answer(question, filters)
    S->>DB: Vector search, 30 candidates by cosine similarity
    S->>DB: Full-text search, 30 candidates by ts_rank_cd
    S->>S: Merge with RRF, keep top 8
    alt Best similarity below MIN_SIMILARITY
        S->>S: Refuse with weak evidence, no LLM call
    else Evidence strong enough
        S->>C: System prompt, chunks, question, JSON schema
        alt API error, rate limit or unusable reply
            C-->>S: Error
            S->>S: Refuse with llm error reason
        else Model sets refused to true
            C-->>S: refused and reason
            S->>S: Refuse with the model's reason
        else Model answers
            C-->>S: answer and citations
            S->>G: check_grounding(answer, citations, chunks)
            alt Check fails
                G-->>S: Issues found
                S->>S: Refuse with grounding check failed
            else Check passes
                G-->>S: OK
                S->>S: Build answer with citations and cost
            end
        end
    end
    S->>DB: Insert row into query_log
    S-->>A: AnswerResponse
    A-->>W: JSON
    W-->>U: Answer with citation chips, or the refusal reason
```

Step by step:

1. **Guards (`api.py`, `ask`).** Pydantic validates the body (`question` 3 to 1,000 characters, optional `company` and `year`). The rate limiter checks the client key; over the limit returns 429 with `Retry-After`. Then one SQL count of `query_log` rows with `llm_called` in the last day; at or over `DAILY_QUERY_CAP` returns 503.
2. **Retrieval (`retrieve.py`).** Both queries share a `WHERE` clause: `chunking = %s` plus the optional filters.

   ```sql
   -- vector
   SELECT ..., 1 - (embedding <=> %s::vector) AS sim FROM chunks WHERE ...
   ORDER BY embedding <=> %s::vector LIMIT 30
   -- full-text
   SELECT ..., ts_rank_cd(tsv, q) AS score
   FROM chunks, to_tsquery('english', %s) q WHERE ... AND tsv @@ q
   ORDER BY score DESC, id LIMIT 30
   ```

   `<=>` is pgvector's cosine distance, so `1 - distance` is cosine similarity. `rrf_merge` combines the two id lists; ties break by id; the top `TOP_K` (8) are kept. `best_similarity` is the highest cosine similarity among the vector candidates.
3. **Weak evidence (`answer.py`).** No hits, or `best_similarity < MIN_SIMILARITY`: refuse with `"weak evidence: best similarity X < Y"`. `llm_called` stays false, so this costs nothing and doesn't count against the daily cap.
4. **LLM call.** `llm.complete_json(SYSTEM_PROMPT, build_prompt(question, hits), ANSWER_SCHEMA, "answer")`. Any `LLMError` becomes a refusal with `"llm error: ..."`. On success, tokens and `cost_usd` are recorded and `llm_called` is set.
5. **Model refusal.** If the JSON has `refused: true`, refuse with `"model: <reason>"`.
6. **Grounding.** Citations are de-duplicated in order. `evidence` maps each retrieved id to `"{company} {ticker} {section_title}\n{text}"`. `check_grounding` runs; on failure the response is a refusal with `"grounding check failed: ..."` and `grounding_issues` set.
7. **Success.** `answer` is the model's text, and each citation is expanded with company, ticker, year, section, title and a 280-character snippet.
8. **Logging.** `_finish` sets `latency_ms` (whole request inside the Answerer, retrieval included) and writes the `query_log` row, for answers and refusals alike. The eval harness passes `log=False`, so eval runs don't touch `query_log`.

In every refusal the user sees `REFUSAL_TEXT` and the reason; the model's own text stays in `model_answer`.

## 6. Prompt and JSON schema

### System prompt (`answer.py`, `SYSTEM_PROMPT`)

```text
You answer questions about companies using ONLY the 10-K excerpts provided.

Rules:
- Use only facts stated in the <chunk> passages. Do not use outside knowledge, even if you are sure.
- Cite every chunk you used by its id in "citations". Cite only ids that appear in the passages.
- Copy numbers exactly as written in the passages, with their units (for example "$1,587.1 million").
  Say which fiscal year a number belongs to.
- If the passages do not contain the answer, set "refused" to true, leave "citations" empty and
  explain in "reason" what is missing. A partial answer is better than a guess only if every part
  you state is supported; otherwise refuse.
- For comparisons, cite a chunk for each side of the comparison.
- Keep the answer to one to three sentences.
```

Why these rules: "copy numbers exactly" makes the grounding number check pass on correct answers; "cite only ids that appear" matches the citation check; "cite a chunk for each side" matches how comparison questions are scored.

### User message (`build_prompt`)

```text
<passages>
<chunk id="ACME-2024-8-t000" company="Acme Robotics, Inc." year="2024" section="Item 8. Financial Statements and Supplementary Data">
...chunk text...
</chunk>
...up to TOP_K chunks...
</passages>

<question>
What was Acme Robotics' total revenue in fiscal 2024?
</question>
```

Company names and section titles are HTML-escaped inside the attributes. The chunk ids in the prompt are the same ids the model must cite.

### Answer schema (`ANSWER_SCHEMA`)

```json
{
  "type": "object",
  "properties": {
    "answer":    {"type": "string", "description": "The answer, or a short refusal."},
    "citations": {"type": "array", "items": {"type": "string"},
                  "description": "Chunk ids that support the answer."},
    "refused":   {"type": "boolean"},
    "reason":    {"type": "string", "description": "Why refused, or which passages were used."}
  },
  "required": ["answer", "citations", "refused", "reason"],
  "additionalProperties": false
}
```

It is sent as `output_config.format` with `type: "json_schema"`, so the reply is JSON that matches it. `AnthropicLLM` still checks `stop_reason` first, because a `refusal` or `max_tokens` stop can leave the text incomplete.

### Judge prompt and schema (`evals/scoring.py`)

`JUDGE_SYSTEM` tells the model to compare the model answer with the gold answer and mark it correct if it states the same facts (same numbers allowing units and rounding, right company and year). Extra correct detail is fine; a wrong number, year or company, a missing part, or a refusal is incorrect. The user message wraps `<question>`, `<gold_answer>` and `<model_answer>`. The schema is `{"correct": boolean, "explanation": string}`, both required, no extra fields. The judge uses the same `LLM` client and model as answering.

## 7. The grounding check in detail

```mermaid
flowchart TD
    START["Model answer + cited ids"] --> C1{"Any citations?"}
    C1 -->|no| R1["Reject: no citations"]
    C1 -->|yes| C2{"All cited ids were retrieved?"}
    C2 -->|no| R2["Reject: cites chunks not retrieved"]
    C2 -->|yes| C3{"Every non-trivial number found<br/>in cited chunks or question?"}
    C3 -->|no| I1["Issue: numbers not found"]
    C3 -->|yes| C4
    I1 --> C4{"At least 40 percent of key terms<br/>in cited chunks or question?"}
    C4 -->|no| I2["Issue: low key-term coverage"]
    C4 -->|yes| END{"Any issues?"}
    I2 --> END
    END -->|yes| REJ["Refuse, log issues"]
    END -->|no| OK["Accept answer"]
```

**Number matching (`number_supported`).** The answer number's target value is `value x scale`. Its tolerance is half a unit in its last written decimal place, scaled: `0.5 x 10^(-decimals) x scale`. Each evidence number is tried at its own scale and at 1, 1e3, 1e6 and 1e9, because tables state "in millions" once at the top. Percentages only match percentages.

Worked examples (from `tests/test_answer.py`, cited chunk contains `Revenue | $ 1,587.1 | $ 1,285.3`):

- "$1,587.1 million": target 1,587,100,000, tolerance 50,000. Evidence 1,587.1 at scale 1e6 matches exactly. **Supported.**
- "about $1.6 billion": target 1,600,000,000, tolerance 50,000,000. Evidence 1,587.1 at scale 1e6 is 12.9 million away. **Supported.**
- "$1,612.0 million": target 1,612,000,000, tolerance 50,000. The nearest evidence is 24.9 million away. **Rejected.**

**Key-term coverage.** Terms are words of 4 or more letters (letters, hyphens, apostrophes), lower-cased, with a trailing "'s" removed, minus a stop list that includes filler like "approximately", "reported", "total" and the scale words. Coverage is the share of terms that appear as substrings of the cited text plus the question. Below 0.4 is an issue.

**Why numbers in the question count as evidence.** "Did revenue exceed $1 billion?" can repeat "$1 billion" in its answer without the filing printing that number.

**What it misses.** It does not check that a number is attached to the right label or year (a correct number from the wrong row of a cited table passes). It rejects correct derived numbers. Term coverage can be gamed by echoing passage wording.

## 8. Eval harness: how each metric is computed

### The eval set format (`evals/dataset.py`)

One JSON object per line with `id`, `question`, `answerable`, `answer_type` (`number`, `text`, `comparison`, `unanswerable`), `gold_answer`, `gold_sections` (list of `{ticker, year, section}`), and optional `gold_evidence`, `filters`, `split` (`dev` or `test`, default `test`), `notes`, `format_example`.

`load_questions(path)` returns a `ValidationReport`:

- **Errors** (the run stops): missing or unknown fields, bad types, an answerable question without a gold answer or gold section, an unanswerable question with a gold answer, gold sections, or a type other than `unanswerable`, duplicate ids.
- **Warnings**: fewer than 120 real questions, an unanswerable share outside 15% to 35%, no dev questions, duplicate question text, a comparison with fewer than two gold sections.

`is_format_examples()` is true for any file named like `FORMAT_EXAMPLES` or containing `format_example: true`; the harness refuses to score it without `--smoke`, and with `--smoke` prints answers but no metrics.

`select_subset(questions, n)`: sorts answerable and unanswerable questions separately by SHA-256 of their id, takes a share of unanswerable questions proportional to the full set (at least one if any exist), and returns them in file order. Deterministic, so CI always runs the same 20.

### Running (`evals/harness.py`, `run_questions`)

For each question: build `Filters` from the question's `filters`, call `answerer.answer(conn, question, filters, log=False)`, and record answer, refusal, citations, latency, tokens and cost. For answerable questions it also computes:

- `auto_correct = not refused and auto_match(gold_answer, answer)`
- if answered: `citation_contains_answer` and `citation_section_match` from the cited chunks (fetched with `get_chunk`)
- with `--judge`: `judge_correct` from the judge, or `False` without a call if the system refused. A judge error leaves it `None`.

Unanswerable questions are not judged.

### Scoring functions (`evals/scoring.py`)

- `normalize(text)`: lower-case, drop commas, replace anything outside `[a-z0-9.%$ ]` with a space, collapse spaces, strip leading and trailing spaces and dots.
- `auto_match(gold, answer)`: if the gold answer has non-trivial numbers, **every** one must be supported by a number in the answer (same `number_supported` rule as grounding, with scale and rounding). Otherwise the normalised gold text must be a substring of the normalised answer.
- `chunk_contains_answer(q, chunk_text)`: true if the normalised `gold_evidence` is in the chunk; else, if the gold answer has numbers, true if **any** of them is supported by a number in the chunk; else true if the normalised gold answer is in the chunk.
- `section_match(q, cited)`: true if any cited (ticker, year, section) is one of the gold sections.

### Metrics (`summarize`)

Let A = answerable results, U = unanswerable results, and D = answerable results that were not refused.

| Metric | Formula |
|---|---|
| `answer_accuracy_auto` | count(A where `auto_correct`) / count(A) |
| `answer_accuracy_judge` | count(A where `judge_correct`) / count(A), only with `--judge` |
| `citation_accuracy` | count(D where `citation_contains_answer`) / count(D) |
| `citation_section_match` | count(D where `citation_section_match`) / count(D) |
| `correct_refusal_rate` | count(U refused) / count(U) |
| `false_refusal_rate` | count(A refused) / count(A) |
| `latency_p50_ms`, `latency_p95_ms` | `statistics.quantiles(latencies, n=100, method="inclusive")`, entries 50 and 95, over all questions |
| `cost_per_1000_queries_usd` | sum of answering cost / number of questions x 1,000 (judge cost excluded) |
| `answer_cost_usd`, `judge_cost_usd` | totals for the run |

A rate is `None` (printed `n/a`) when its denominator is 0.

Two choices to be able to defend:

- **Refusals count as wrong in accuracy.** Otherwise a system that refuses every hard question would look accurate. The false refusal rate shows the same cost from the other side.
- **Citation accuracy uses only answered questions.** It measures whether citations are right when the system does answer. It can pass if the same number appears in a cited chunk for a different reason.

### Comparing configurations

`--compare baseline,improved` runs the questions once per label (`improved` is an alias for `table_aware`), with the same LLM and embedder. `flips(a, b)` lists answerable questions whose `auto_correct` changed, as `wrong->right` or `right->wrong`.

### Judge spot check and agreement

With `--judge`, `spotcheck_rows` shuffles (seed 0) the answerable results that have a judge verdict, which includes refused ones recorded as `False`, and writes the first 30 to `judge_spotcheck.csv` with empty `human_correct` and `human_notes` columns. After a human fills `human_correct`, `eval agreement CSV` computes:

- raw agreement: p_o = agreeing rows / n
- chance agreement: p_e = p_h x p_j + (1 - p_h) x (1 - p_j), where p_h and p_j are the shares marked correct by the human and the judge
- Cohen's kappa = (p_o - p_e) / (1 - p_e), or n/a when p_e = 1

Rows without a yes/no in both columns are skipped. It warns when fewer than 30 rows were checked. Kappa matters because raw agreement looks high whenever most answers are correct.

### Cost estimate (`estimate_cost`, `--dry-run`)

For each configuration and question it runs the real retrieval (local and free). Questions below the similarity floor count as "refused before the LLM". For the rest it builds the real prompt and estimates input tokens as characters / 3.5 (deliberately pessimistic; English is about 4 characters per token), plus 250 output tokens. With `--judge`, each answerable question adds a judge call estimated at 450 input tokens plus the judge system prompt, and 80 output tokens. Dollars come from `Settings.cost_usd`. No API call is made.

### Run gates and outputs (`cli.py`, `cmd_eval_run`)

1. Validation errors: exit 2.
2. Format examples without `--smoke`: exit 2.
3. The plan and the cost estimate are always printed. `--dry-run` stops here.
4. A non-fake provider without `--yes`: exit 3.
5. With the fake provider it prints a note that the numbers test the plumbing only.

Each run writes `evals/runs/<YYYYMMDD-HHMMSS>/<label>/summary.json`, `results.jsonl` (one `QuestionResult` per line) and, with `--judge`, `judge_spotcheck.csv`. `evals/runs/` is git-ignored.

### In CI

The `eval-subset` job in `.github/workflows/ci.yml` runs only on push, and only does real work when the `ANTHROPIC_API_KEY` and `SEC_USER_AGENT` secrets exist and `evals/questions.jsonl` is committed. It downloads and ingests the corpus (filings cached by the hash of `corpus.toml`), runs `eval run --subset 20 --dry-run`, then `eval run --subset 20 --yes`, and uploads `evals/runs/` as an artifact.

## 9. API and web UI

### Endpoints (`api.py`)

| Method and path | Purpose | Notes |
|---|---|---|
| `GET /health` | Liveness, shows `llm_provider` and `chunking` | Open without the access code |
| `GET /auth/check` | Lets the UI test the access code | Returns `{"ok": true}` |
| `GET /filters` | Companies (ticker, name, years) and all years | From `documents` |
| `POST /ask` | Answer a question | Rate limit, daily cap, then `Answerer.answer` |
| `GET /chunks/{chunk_id}` | Full chunk text and source link | `source_url` gets `#anchor` appended when known |
| `GET /usage` | Usage, latency and cost stats | `usage_stats()` |
| `POST /feedback` | Thumbs up (1) or down (-1) for a `query_id` | 422 for other ratings, 404 for unknown ids |

**Middleware order.** The access-code middleware is added first and CORS last, so CORS wraps everything and 401 and 429 responses still carry CORS headers (the UI can read them). `AccessCodeMiddleware` lets `OPTIONS` (CORS preflight) and `/health` through and compares `X-Access-Code` with `hmac.compare_digest`.

Route handlers are plain `def` functions, so FastAPI runs them in its thread pool; each borrows a connection from the pool for the request.

### Web UI (`web/`)

- Next.js 16 App Router, React 19, TypeScript, `output: "standalone"` for the Docker image.
- `lib/api.ts`: typed client. Every request sends `X-Access-Code` from localStorage key `doc-copilot-access-code`. `API_URL` is `NEXT_PUBLIC_API_URL` or `http://localhost:8001`, fixed at build time.
- `components/AccessGate.tsx`: calls `/auth/check`; shows the child page, a code prompt on 401, or "API not reachable" with a retry button.
- `components/Chat.tsx`: company and fiscal-year selects from `/filters` (years narrow to the chosen company), three example questions, the conversation, citation chips (`[n] TICKER YEAR · Item X`) that open the source passage in a side panel via `/chunks/{id}` with an "open on sec.gov" link, refusals shown with "Why: reason", latency and cost per answer (or "no LLM call"), and one Helpful / Not helpful vote per answer.
- `components/Usage.tsx` at `/usage`: the `/usage` stats, a 7-day table and the 20 most recent questions.

All fetching is client-side. That keeps the frontend stateless, but the API URL is public and the access code lives in the browser.

## 10. Testing strategy

- **No paid calls, ever.** `tests/conftest.py` runs before any project import: it deletes every `ANTHROPIC_*` variable, forces `LLM_PROVIDER=fake` and `EMBEDDER=fake`, sets `DOC_COPILOT_TESTING=1` (so `AnthropicLLM` refuses to build) and clears `ACCESS_CODE`.
- **Real Postgres for DB tests.** Tests that need the database use `TEST_DATABASE_URL` (default `localhost:5434/doc_copilot_test`, created if missing) and truncate all tables per test. Without Postgres they skip, unless `REQUIRE_DB=1` (as in CI), which makes them fail.
- **Synthetic fixtures.** `tests/fixtures/filings/` holds three made-up 10-Ks (ACME 2023, ACME 2024, GLBX 2024) with a table of contents, hidden XBRL, split `$` and `%` cells and multi-row table headers, so parser edge cases are tested without network access. The downloader is tested with `httpx.MockTransport`.
- **Scripted fakes.** `FakeLLM.script()` queues exact replies, which is how tests produce a hallucinated citation, an invented number, a model refusal and an LLM error.

| File | Covers |
|---|---|
| `tests/test_ingest.py` | Section detection, TOC duplicates, hidden XBRL, cell merging, heading edge cases, chunk ids, overlap, table headers and captions, the baseline table-split failure, User-Agent checks, fiscal-year selection, idempotent re-ingest, partial re-embed |
| `tests/test_retrieve.py` | RRF rewards agreement, tsquery sanitising, hybrid hits, filters, strategy separation, chunk lookup |
| `tests/test_answer.py` | Grounding accept and reject cases, number extraction, strict schema, prompt contents, fake provider default, each refusal path |
| `tests/test_api.py` | Health, filters, ask, citation open, feedback, errors, usage counts, access code (including CORS on 401), rate limit, daily cap, limiter window |
| `tests/test_evals.py` | Validator errors and warnings, subset stratification, auto match, judge, agreement and kappa, percentiles, the format-example refusal, end-to-end harness, dry run, the `--yes` gate, smoke run, compare |

72 tests in total (some parametrised).

## 11. Trade-offs and known gaps

**Retrieval**

- RRF ignores score magnitude: a near-perfect vector match and a barely relevant one at the same rank count the same. A cross-encoder re-ranker on the top 30 would fix ordering at the cost of latency.
- `RRF_K=60` and `CANDIDATES_PER_RETRIEVER=30` are defaults, not tuned.
- Postgres `ts_rank_cd` is not BM25 (no length normalisation by default). It avoids running a second search system.
- HNSW is approximate, and filters are applied to its candidates, so a narrow company and year filter on a large index can return fewer than 30 vector hits. At the current corpus size this is not expected to matter; it is not measured.
- The OR query raises recall but lets common words match many chunks; RRF and the vector list counter that.

**Chunking**

- Words stand in for tokens (about 1.3 tokens per English word).
- `table_aware` produces more chunks (25 vs. 18 on the fixture) and repeats headers, so prompts use more tokens. Very wide tables still overflow.
- Whether `table_aware` improves accuracy on real filings is not measured yet.

**Answering and grounding**

- The number check rejects correct derived numbers, so comparison and growth questions will see more false refusals.
- The check does not tie a number to its row or year label.
- `MIN_SIMILARITY` trades wasted LLM calls against false refusals and must be tuned on the dev split only.
- The answer service does not retry by itself; it relies on the SDK's two retries, then refuses. The `retryable` flag on error types is available for a future retry policy.

**Operations**

- The rate limiter is per process and resets on restart; two instances double the effective limit. A Postgres or Redis counter would fix it.
- The rate-limit key trusts the first `X-Forwarded-For` entry, which is right behind a proxy that sets it (Render and Fly.io do) but spoofable if the API is exposed directly.
- The daily cap is the real spend ceiling: worst-case daily cost = cap x cost per query.
- CPU torch makes the API image large and needs about 1 GB of RAM. An ONNX embedder would be lighter.
- Ingest change detection ignores the embedder, so switching `EMBEDDER` needs an empty database.

**Evaluation**

- Character-based token estimates are approximate on purpose (pessimistic).
- The LLM judge is itself a model; it is checked by a 30-answer human spot check, with agreement and kappa reported.
- Citation accuracy can pass when the gold number appears in a cited chunk for an unrelated reason.

For per-milestone decisions and practice exercises, see [INTERVIEW_NOTES.md](INTERVIEW_NOTES.md).
