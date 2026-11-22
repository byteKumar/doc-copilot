# API image: FastAPI + local MiniLM embeddings (CPU torch).
FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra local-embeddings --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --extra local-embeddings

# Bake the embedding model into the image so containers start without downloading it.
RUN uv run --no-sync python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

RUN useradd --create-home app && chown -R app /app
USER app
ENV EMBEDDER=local PORT=8001
EXPOSE 8001
CMD ["sh", "-c", "uv run --no-sync uvicorn doc_copilot.api:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
