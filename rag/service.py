"""Glue between config, store, retriever and LLM. Used by the API, CLI and eval."""

from __future__ import annotations

import os

import numpy as np

from rag.answer import LLM, Answer, ClaudeLLM, ExtractiveLLM, answer_question
from rag.config import Settings
from rag.embeddings import get_embedder
from rag.ingest import ingest_directory
from rag.retrieval import HybridRetriever
from rag.store import VectorStore


def build_index(settings: Settings) -> int:
    """Ingest every supported file under docs_dir and rebuild the index. Returns chunk count."""
    chunks = ingest_directory(settings.docs_dir, settings.chunk_size, settings.chunk_overlap)
    embedder = get_embedder(settings.embedding_provider)
    store = VectorStore(settings.db_path)
    try:
        if chunks:
            vectors = embedder.embed([c.text for c in chunks])
        else:
            vectors = np.zeros((0, embedder.dim), dtype=np.float32)
        store.replace_all(chunks, vectors, embedder.name)
    finally:
        store.close()
    return len(chunks)


def default_llm(settings: Settings) -> LLM:
    if os.getenv("ANTHROPIC_API_KEY"):
        return ClaudeLLM(model=settings.model)
    return ExtractiveLLM()


class RagService:
    def __init__(self, settings: Settings, llm: LLM | None = None) -> None:
        self.settings = settings
        self.llm = llm or default_llm(settings)
        self.store = VectorStore(settings.db_path)
        indexed_provider = self.store.provider()
        if indexed_provider and indexed_provider != settings.embedding_provider:
            raise RuntimeError(
                f"Index was built with provider {indexed_provider!r} but "
                f"RAG_EMBEDDING_PROVIDER={settings.embedding_provider!r}; re-run ingest."
            )
        self.embedder = get_embedder(settings.embedding_provider)
        self.retriever = HybridRetriever(self.store, self.embedder)

    def reload(self) -> None:
        self.retriever = HybridRetriever(self.store, self.embedder)

    def chunk_count(self) -> int:
        return self.store.count()

    def ask(self, question: str, k: int | None = None) -> Answer:
        hits = self.retriever.search(question, k=k or self.settings.top_k)
        score = self.retriever.confidence(question, hits)
        return answer_question(
            question, hits, self.llm, self.settings.confidence_threshold, score=score
        )
