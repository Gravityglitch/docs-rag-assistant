from __future__ import annotations

from pathlib import Path

from rag.config import Settings
from rag.eval import load_questions, run_eval
from rag.service import RagService

ROOT = Path(__file__).resolve().parent.parent


def test_eval_harness_runs_offline(settings: Settings) -> None:
    items = load_questions(ROOT / "evals" / "questions.jsonl")
    report = run_eval(RagService(settings), items, k=5)
    assert report.total == len(items)
    assert report.hit_at_k == 1.0
    assert report.unanswerable_abstained == 1.0
    assert report.citation_accuracy >= 0.9
