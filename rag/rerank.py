"""Optional second-stage rerankers over the fused hybrid candidates.

- ``llm``           : one Claude call scores every candidate 0-10 (no extra dependency)
- ``cross-encoder`` : sentence-transformers cross-encoder (install the ``rerank`` extra)

Both implement :class:`Reranker`; the service applies one only when ``RAG_RERANK=true``.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from rag.logging import get_logger
from rag.retrieval import Hit

if TYPE_CHECKING:
    from rag.answer import LLM

log = get_logger(__name__)

RERANK_SYSTEM = (
    "You grade how useful each passage is for answering the question. "
    "Reply with ONLY a JSON object mapping passage index (as a string) to an integer score "
    "0-10, where 10 means the passage directly answers the question."
)


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]: ...


def _apply_scores(hits: list[Hit], scores: list[float], top_k: int) -> list[Hit]:
    rescored = [replace(h, rerank_score=s) for h, s in zip(hits, scores, strict=True)]
    rescored.sort(key=lambda h: (h.rerank_score or 0.0, h.score), reverse=True)
    return rescored[:top_k]


class LLMReranker:
    name = "llm"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        passages = "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits))
        reply = self.llm.complete(RERANK_SYSTEM, f"Question: {query}\n\nPassages:\n\n{passages}")
        scores = parse_scores(reply, len(hits))
        return _apply_scores(hits, scores, top_k)


def parse_scores(reply: str, n: int) -> list[float]:
    """Pull a ``{"0": 7, "1": 2}`` object out of the reply; missing indexes score 0."""
    match = re.search(r"\{.*\}", reply, re.S)
    scores = [0.0] * n
    if not match:
        log.warning("rerank_parse_failed", reply=reply[:200])
        return scores
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        log.warning("rerank_parse_failed", reply=reply[:200])
        return scores
    for key, value in raw.items():
        try:
            idx = int(key)
            if 0 <= idx < n:
                scores[idx] = max(0.0, min(10.0, float(value)))
        except (TypeError, ValueError):
            continue
    return scores


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "RAG_RERANKER=cross-encoder needs sentence-transformers: uv sync --extra rerank"
            ) from exc
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(query, h.chunk.text) for h in hits])
        return _apply_scores(hits, [float(s) for s in scores], top_k)
