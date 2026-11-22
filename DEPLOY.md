# Deploying doc-copilot

Nothing here has been deployed yet. These are the steps for you to run. Two options: Render
(simplest, uses `render.yaml`) or Fly.io.

## What gets deployed

| Piece | Image | Notes |
|---|---|---|
| Postgres 16 + pgvector | managed | needs `CREATE EXTENSION vector` (the API runs it on start) |
| API (FastAPI) | `Dockerfile` | CPU torch + MiniLM baked in; needs ~1 GB RAM |
| Web (Next.js) | `web/Dockerfile` | `NEXT_PUBLIC_API_URL` is read at **build** time |

Guards already in the code, all set by environment variables:
- `ACCESS_CODE`: every request except `/health` must send it (the UI asks once and stores it in the browser).
- `RATE_LIMIT_PER_MINUTE` (default 10): per client IP on `/ask`, in memory. Resets on restart; one instance only.
- `DAILY_QUERY_CAP` (default 500): LLM calls per rolling 24 h across everyone. This is the spend ceiling:
  cap x cost per query (from your eval) = worst-case daily cost. Set it from the measured number.
- `/usage` and the Usage page: query counts, refusal rate, p50/p95 latency, cost.

## Before you deploy

1. Run the eval and fill RESULTS.md, so you know the cost per query and the right `MIN_SIMILARITY`.
2. Pick an access code (a long random string) and a daily cap.
3. Push the repo to GitHub.

## Option A: Render

1. In Render: New > Blueprint > pick the repo. It reads `render.yaml` and proposes a Postgres
   database and two web services. Check the plans and monthly price it shows before you accept.
2. Fill the `sync: false` values when prompted:
   - API: `ANTHROPIC_API_KEY`, `ACCESS_CODE`, `CORS_ORIGINS` (the web service URL, e.g. `https://doc-copilot-web.onrender.com`).
   - Web: `NEXT_PUBLIC_API_URL` (the API URL, e.g. `https://doc-copilot-api.onrender.com`).
     Render passes service env vars to Docker builds as build args; if the UI calls
     `localhost:8001`, set it again and trigger a manual redeploy.
3. Load the corpus into the managed database. From your machine, with the external database URL
   from the Render dashboard:
   ```bash
   export DATABASE_URL='<render external connection string>'
   export EMBEDDER=local
   uv sync --extra local-embeddings
   uv run doc-copilot ingest --dir data/filings
   ```
   Embeddings are computed locally and only the vectors are uploaded. Re-running is safe (idempotent).
4. Open the web URL, enter the access code, ask a question, then check `/usage`.

## Option B: Fly.io

```bash
fly launch --no-deploy --name doc-copilot-api        # uses ./Dockerfile; say no to its Postgres offer
fly postgres create --name doc-copilot-db             # or use any Postgres 16 with pgvector (e.g. Neon, Supabase)
fly postgres attach doc-copilot-db -a doc-copilot-api # sets DATABASE_URL
fly scale memory 1024 -a doc-copilot-api
fly secrets set -a doc-copilot-api LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... ACCESS_CODE=... \
  CORS_ORIGINS=https://doc-copilot-web.fly.dev
fly deploy -a doc-copilot-api

cd web
fly launch --no-deploy --name doc-copilot-web
fly deploy -a doc-copilot-web --build-arg NEXT_PUBLIC_API_URL=https://doc-copilot-api.fly.dev
```

Check that the Postgres image has pgvector (`CREATE EXTENSION vector;` must succeed). Then ingest
from your machine with `DATABASE_URL` pointing at it (use `fly proxy 5432 -a doc-copilot-db`).

## After deploying

- Put the URL and access code in the README (the code only if you are happy for anyone reading the README to use it; otherwise share it per person).
- Watch `/usage` for the first days. If cost per query is far from the eval number, the questions people ask differ from the eval set; that is worth a line in the customer brief.

## Known limits

- The rate limiter is per process and forgets on restart. For more than one instance, move it to Postgres or Redis.
- The access code is a shared secret, not user accounts. Fine for a demo, not for a customer.
- No TLS termination or WAF of our own; both platforms provide HTTPS.
