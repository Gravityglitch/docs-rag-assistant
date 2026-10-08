"""Glue between config, store, embedder, retriever, reranker and LLM.

Used by the API, CLI and eval. ``build_index`` performs an incremental sync: unchanged
documents (same content hash) are skipped, missing ones are pruned, and a document that
fails to load is logged and skipped rather than aborting the run.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

from rag.answer import (
    ABSTAIN_PHRASE,
    LLM,
    SYSTEM_PROMPT,
    Answer,
    ClaudeLLM,
    ExtractiveLLM,
    Message,
    build_user_prompt,
    finalize_answer,
)
from rag.config import Settings
from rag.embeddings import EmbeddingProvider, get_embedder
from rag.ingest import chunk_pages, content_hash, iter_documents, load_document
from rag.logging import get_logger
from rag.rerank import CrossEncoderReranker, LLMReranker, Reranker
from rag.retrieval import Hit, HybridRetriever, RetrievalDiagnostics
from rag.store import ChunkStore, get_store

log = get_logger(__name__)


@dataclass
class IngestResult:
    chunks: int
    provider: str
    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def make_embedder(settings: Settings) -> EmbeddingProvider:
    provider = settings.resolved_embedding_provider
    if provider == "hash":
        log.warning(
            "hash_embeddings_in_use",
            hint="hash embeddings are for tests/local only; set VOYAGE_API_KEY or OPENAI_API_KEY",
        )
    key = settings.voyage_api_key if provider == "voyage" else settings.openai_api_key
    return get_embedder(
        provider,
        api_key=key.get_secret_value() if key else None,
        timeout=settings.embed_timeout_seconds,
        max_retries=settings.embed_max_retries,
        batch_size=settings.embed_batch_size,
    )


def open_store(settings: Settings, embedder: EmbeddingProvider) -> ChunkStore:
    return get_store(settings, dim=embedder.dim)


def build_index(settings: Settings, full: bool = False) -> IngestResult:
    """Sync the index with ``docs_dir``. ``full=True`` discards and rebuilds everything."""
    embedder = make_embedder(settings)
    store = open_store(settings, embedder)
    try:
        return sync_index(settings, store, embedder, full=full)
    finally:
        store.close()


def sync_index(
    settings: Settings, store: ChunkStore, embedder: EmbeddingProvider, full: bool = False
) -> IngestResult:
    indexed_provider = store.provider()
    if full or (indexed_provider and indexed_provider != embedder.name):
        if indexed_provider and indexed_provider != embedder.name:
            log.warning("provider_changed", old=indexed_provider, new=embedder.name)
        store.clear()
    store.set_provider(embedder.name)

    known = store.doc_hashes()
    result = IngestResult(chunks=0, provider=embedder.name)
    present: set[str] = set()
    for path in iter_documents(settings.docs_dir):
        doc = path.name
        present.add(doc)
        try:
            digest = content_hash(path)
            if known.get(doc) == digest:
                result.skipped.append(doc)
                continue
            chunks = chunk_pages(load_document(path), settings.chunk_size, settings.chunk_overlap)
            vectors = embedder.embed([c.text for c in chunks])
            store.upsert_document(doc, digest, chunks, vectors)
            result.indexed.append(doc)
            log.info("document_indexed", doc=doc, chunks=len(chunks))
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the run
            result.failed[doc] = f"{type(exc).__name__}: {exc}"
            log.error("document_failed", doc=doc, error=result.failed[doc])

    for doc in set(known) - present:
        store.delete_document(doc)
        result.pruned.append(doc)
        log.info("document_pruned", doc=doc)

    result.chunks = store.count()
    log.info(
        "index_synced",
        chunks=result.chunks,
        indexed=len(result.indexed),
        skipped=len(result.skipped),
        pruned=len(result.pruned),
        failed=len(result.failed),
    )
    return result


def default_llm(settings: Settings) -> LLM:
    if settings.anthropic_api_key is not None:
        return ClaudeLLM(
            model=settings.model,
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            max_tokens=settings.llm_max_tokens,
        )
    log.warning("extractive_llm_in_use", hint="set ANTHROPIC_API_KEY for Claude answers")
    return ExtractiveLLM()


def default_reranker(settings: Settings, llm: LLM) -> Reranker | None:
    if not settings.rerank:
        return None
    if settings.reranker == "cross-encoder":
        return CrossEncoderReranker(settings.cross_encoder_model)
    return LLMReranker(llm)


class RagService:
    def __init__(
        self,
        settings: Settings,
        llm: LLM | None = None,
        reranker: Reranker | None = None,
        store: ChunkStore | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm or default_llm(settings)
        self.embedder = make_embedder(settings)
        self.store = store or open_store(settings, self.embedder)
        indexed_provider = self.store.provider()
        if indexed_provider and indexed_provider != self.embedder.name:
            raise RuntimeError(
                f"Index was built with provider {indexed_provider!r} but the configured "
                f"provider is {self.embedder.name!r}; re-run ingest."
            )
        self.reranker = reranker or default_reranker(settings, self.llm)
        self.retriever = HybridRetriever(self.store, self.embedder)

    def reload(self) -> None:
        self.retriever = HybridRetriever(self.store, self.embedder)

    def chunk_count(self) -> int:
        return self.store.count()

    def ready(self) -> bool:
        return self.store.ping()

    def trim_history(self, history: list[Message] | None) -> list[Message]:
        """Keep the last N user/assistant turns (2 messages per turn), oldest first."""
        if not history or self.settings.history_turns == 0:
            return []
        return history[-2 * self.settings.history_turns :]

    def retrieve(
        self, question: str, k: int | None = None
    ) -> tuple[list[Hit], RetrievalDiagnostics]:
        top_k = k or self.settings.top_k
        diag = RetrievalDiagnostics(
            chunks_indexed=len(self.retriever.chunks),
            threshold=self.settings.confidence_threshold,
            reranked=self.reranker is not None,
            reranker=self.reranker.name if self.reranker else None,
        )
        started = time.perf_counter()
        if self.reranker is None:
            hits = self.retriever.search(question, k=top_k, candidates=self.settings.candidates)
            diag.candidates = len(hits)
            diag.retrieval_ms = round((time.perf_counter() - started) * 1000, 2)
        else:
            pool = max(top_k, self.settings.rerank_candidates)
            candidates = self.retriever.search(
                question, k=pool, candidates=self.settings.candidates
            )
            diag.candidates = len(candidates)
            diag.retrieval_ms = round((time.perf_counter() - started) * 1000, 2)
            started = time.perf_counter()
            hits = self.reranker.rerank(question, candidates, top_k)
            diag.rerank_ms = round((time.perf_counter() - started) * 1000, 2)
        diag.confidence = self.retriever.confidence(question, hits)
        diag.hits = [h.diagnostic() for h in hits]
        return hits, diag

    def ask(
        self, question: str, k: int | None = None, history: list[Message] | None = None
    ) -> Answer:
        hits, diag = self.retrieve(question, k)
        if diag.confidence < self.settings.confidence_threshold:
            return Answer(
                ABSTAIN_PHRASE, confidence=diag.confidence, abstained=True, diagnostics=diag
            )
        started = time.perf_counter()
        text = self.llm.complete(
            SYSTEM_PROMPT, build_user_prompt(question, hits), self.trim_history(history)
        )
        diag.llm_ms = round((time.perf_counter() - started) * 1000, 2)
        answer = finalize_answer(text, hits, diag.confidence)
        answer.diagnostics = diag
        return answer

    def ask_stream(
        self, question: str, k: int | None = None, history: list[Message] | None = None
    ) -> Iterator[tuple[str, object]]:
        """Yield ``(event, data)`` pairs: ``meta`` once, ``delta`` per text piece, ``done`` last.

        The ``done`` payload is authoritative: if the streamed text turns out to be uncited
        the final answer is the abstain phrase even though deltas were already sent.
        """
        hits, diag = self.retrieve(question, k)
        yield "meta", {"confidence": diag.confidence, "threshold": diag.threshold}
        if diag.confidence < self.settings.confidence_threshold:
            answer = Answer(ABSTAIN_PHRASE, confidence=diag.confidence, abstained=True)
            answer.diagnostics = diag
            yield "done", answer
            return
        started = time.perf_counter()
        pieces: list[str] = []
        for piece in self.llm.stream(
            SYSTEM_PROMPT, build_user_prompt(question, hits), self.trim_history(history)
        ):
            pieces.append(piece)
            yield "delta", piece
        diag.llm_ms = round((time.perf_counter() - started) * 1000, 2)
        answer = finalize_answer("".join(pieces), hits, diag.confidence)
        answer.diagnostics = diag
        yield "done", answer
