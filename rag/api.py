"""FastAPI backend: POST /ingest, POST /ask, GET /health."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.config import Settings
from rag.service import RagService, build_index

app = FastAPI(title="docs-rag-assistant", version="0.1.0")


@lru_cache(maxsize=1)
def get_service() -> RagService:
    return RagService(Settings.from_env())


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    doc: str
    page: int
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    confidence: float
    abstained: bool


class IngestResponse(BaseModel):
    chunks: int
    provider: str


@app.get("/health")
def health() -> dict[str, object]:
    service = get_service()
    return {
        "status": "ok",
        "chunks": service.chunk_count(),
        "embedding_provider": service.embedder.name,
        "llm": type(service.llm).__name__,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    service = get_service()
    count = build_index(service.settings)
    service.reload()
    return IngestResponse(chunks=count, provider=service.embedder.name)


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    service = get_service()
    if service.chunk_count() == 0:
        raise HTTPException(status_code=409, detail="Index is empty; call POST /ingest first.")
    result = service.ask(body.question, k=body.top_k)
    return AskResponse(**result.to_dict())
