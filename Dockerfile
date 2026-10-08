# syntax=docker/dockerfile:1

# ---- Builder: resolve and install dependencies with uv ----
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install third-party dependencies first for better layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Then add the project source and install it into the same venv.
COPY rag/ ./rag/
COPY README.md LICENSE ./
RUN uv sync --frozen --no-dev

# ---- Runtime: minimal image with the prebuilt virtualenv ----
FROM python:3.12-slim-bookworm

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app rag/ ./rag/
COPY --chown=app:app docs/ ./docs/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    RAG_HOST=0.0.0.0 \
    RAG_PORT=8000 \
    RAG_DB_PATH=/data/index.sqlite

RUN mkdir -p /data && chown app:app /data
VOLUME /data

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"]

CMD ["rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
