from __future__ import annotations

import json

from rag.config import Settings
from rag.ingest import Chunk
from rag.rerank import LLMReranker, parse_scores
from rag.retrieval import Hit
from rag.service import RagService
from tests.conftest import FakeLLM


def _hits(n: int) -> list[Hit]:
    return [
        Hit(
            Chunk(f"d#p1#c{i}", "d.md", 1, f"passage {i}"),
            score=1.0 / (i + 1),
            vector_score=0.1,
            bm25_score=0.1,
        )
        for i in range(n)
    ]


def test_parse_scores_handles_noise_and_bounds() -> None:
    assert parse_scores('Sure: {"0": 3, "1": "9", "7": 1, "x": 2}', 3) == [3.0, 9.0, 0.0]
    assert parse_scores("no json here", 2) == [0.0, 0.0]
    assert parse_scores("{broken", 1) == [0.0]
    assert parse_scores('{"0": 99, "1": -4}', 2) == [10.0, 0.0]


def test_llm_reranker_reorders_and_truncates() -> None:
    llm = FakeLLM(json.dumps({"0": 1, "1": 2, "2": 9}))
    reranked = LLMReranker(llm).rerank("q", _hits(3), top_k=2)
    assert [h.chunk.chunk_id for h in reranked] == ["d#p1#c2", "d#p1#c1"]
    assert reranked[0].rerank_score == 9.0
    assert LLMReranker(llm).rerank("q", [], top_k=2) == []


class ReverseReranker:
    name = "reverse"

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        return list(reversed(hits))[:top_k]


def test_rerank_flag_controls_stage_and_diagnostics(settings: Settings) -> None:
    llm = FakeLLM("x [employee_handbook.md p.2].")
    question = "How many PTO days per year?"

    off = RagService(settings, llm=llm)
    assert off.reranker is None
    _, diag_off = off.retrieve(question)
    assert diag_off.reranked is False and diag_off.candidates == 5

    on_settings = settings.model_copy(update={"rerank": True, "rerank_candidates": 8, "top_k": 3})
    on = RagService(on_settings, llm=llm, reranker=ReverseReranker())
    hits, diag_on = on.retrieve(question)
    assert diag_on.reranked is True and diag_on.reranker == "reverse"
    assert diag_on.candidates == 8 and len(hits) == 3
    assert diag_on.rerank_ms >= 0


def test_default_reranker_is_llm_when_enabled(settings: Settings) -> None:
    service = RagService(settings.model_copy(update={"rerank": True}), llm=FakeLLM("{}"))
    assert isinstance(service.reranker, LLMReranker)
