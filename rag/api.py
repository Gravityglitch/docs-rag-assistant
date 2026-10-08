"""FastAPI backend.

Routes: ``GET /health``, ``GET /ready``, ``POST /ingest``, ``POST /ask``, ``POST /ask/stream``
(SSE) and, when ``RAG_METRICS_ENABLED=true``, ``GET /metrics``. ``/ask*`` and ``/ingest``
require ``X-API-Key`` when ``RAG_API_KEY`` is set. Every request gets a request id
(``X-Request-ID`` in and out) bound into the structlog context.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, Literal

import structlog
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from rag.config import Settings
from rag.logging import configure_logging, get_logger
from rag.service import RagService, build_index

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_service() -> RagService:
    return RagService(get_settings())


def _rate_limit_key(request: Request) -> str:
    return request.headers.get("x-api-key") or get_remote_address(request)


def _rate_limit_value() -> str:
    return get_settings().rate_limit


limiter = Limiter(key_func=_rate_limit_key)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(json_logs=settings.log_json, level=settings.log_level)
    if not settings.auth_enabled:
        log.warning("api_auth_disabled", hint="set RAG_API_KEY to require X-API-Key")
    if settings.metrics_enabled:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    log.info(
        "startup",
        store="pgvector" if settings.use_postgres else "sqlite",
        embedding_provider=settings.resolved_embedding_provider,
        rerank=settings.rerank,
        metrics=settings.metrics_enabled,
    )
    yield


app = FastAPI(title="docs-rag-assistant", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limited(request: Request, exc: RateLimitExceeded) -> Response:
    return Response(
        content=json.dumps({"detail": f"Rate limit exceeded: {exc.detail}"}),
        status_code=429,
        media_type="application/json",
        headers={"Retry-After": "60"},
    )


@app.middleware("http")
async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id, path=request.url.path, method=request.method
    )
    limit = get_settings().max_request_bytes
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > limit:
        response = Response(
            content=json.dumps({"detail": f"Request body exceeds {limit} bytes"}),
            status_code=413,
            media_type="application/json",
        )
    else:
        response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    log.info("request", status=response.status_code)
    return response


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: Annotated[str | None, Depends(_api_key_header)]) -> None:
    settings = get_settings()
    if not settings.auth_enabled:
        return
    expected = settings.api_key.get_secret_value() if settings.api_key else ""
    if key is None or key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


Protected = Depends(require_api_key)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    history: list[Turn] = Field(default_factory=list, max_length=100)


class CitationOut(BaseModel):
    doc: str
    page: int
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    confidence: float
    abstained: bool
    diagnostics: dict[str, object] | None = None


class IngestResponse(BaseModel):
    chunks: int
    provider: str
    indexed: list[str]
    skipped: list[str]
    pruned: list[str]
    failed: dict[str, str]


@app.get("/health")
def health() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ok",
        "version": app.version,
        "store": "pgvector" if settings.use_postgres else "sqlite",
        "embedding_provider": settings.resolved_embedding_provider,
        "rerank": settings.rerank,
        "auth": settings.auth_enabled,
    }


@app.get("/ready")
def ready(response: Response) -> dict[str, object]:
    try:
        service = get_service()
        store_ok = service.ready()
        chunks = service.chunk_count() if store_ok else 0
    except Exception as exc:  # noqa: BLE001 - readiness must report, not crash
        log.error("ready_check_failed", error=repr(exc))
        store_ok, chunks = False, 0
    if not store_ok:
        response.status_code = 503
    return {
        "status": "ready" if store_ok else "unavailable",
        "store_ok": store_ok,
        "chunks": chunks,
        "index_empty": chunks == 0,
        "llm": type(get_service().llm).__name__ if store_ok else None,
    }


@app.post("/ingest", response_model=IngestResponse, dependencies=[Protected])
@limiter.limit(_rate_limit_value)
def ingest(request: Request, full: bool = False) -> IngestResponse:
    service = get_service()
    result = build_index(service.settings, full=full)
    service.reload()
    return IngestResponse(**result.to_dict())


def _history(body: AskRequest) -> list[dict[str, str]]:
    return [turn.model_dump() for turn in body.history]


def _require_index(service: RagService) -> None:
    if service.chunk_count() == 0:
        raise HTTPException(status_code=409, detail="Index is empty; call POST /ingest first.")


@app.post(
    "/ask", response_model=AskResponse, response_model_exclude_none=True, dependencies=[Protected]
)
@limiter.limit(_rate_limit_value)
def ask(
    request: Request,
    body: AskRequest,
    debug: Annotated[bool, Query(description="Include retrieval diagnostics")] = False,
) -> AskResponse:
    service = get_service()
    _require_index(service)
    result = service.ask(body.question, k=body.top_k, history=_history(body))
    log.info(
        "ask",
        abstained=result.abstained,
        confidence=result.confidence,
        citations=[(c.doc, c.page) for c in result.citations],
        chunk_ids=[h["chunk_id"] for h in result.diagnostics.hits] if result.diagnostics else [],
    )
    return AskResponse(**result.to_dict(debug=debug))


@app.post("/ask/stream", dependencies=[Protected])
@limiter.limit(_rate_limit_value)
async def ask_stream(
    request: Request,
    body: AskRequest,
    debug: Annotated[bool, Query(description="Include retrieval diagnostics")] = False,
) -> EventSourceResponse:
    """Server-sent events: ``meta`` -> ``delta``* -> ``done`` (final AskResponse JSON)."""
    service = get_service()
    _require_index(service)
    stream = service.ask_stream(body.question, k=body.top_k, history=_history(body))

    async def events() -> AsyncIterator[dict[str, str]]:
        async for event, data in iterate_in_threadpool(stream):
            if event == "delta":
                yield {"event": "delta", "data": json.dumps({"text": data})}
            elif event == "done":
                yield {"event": "done", "data": json.dumps(data.to_dict(debug=debug))}
            else:
                yield {"event": event, "data": json.dumps(data)}

    return EventSourceResponse(events())
