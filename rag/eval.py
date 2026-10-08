"""Offline evaluation against a labelled question set.

Usage: `uv run python -m rag.eval [--questions evals/questions.jsonl] [--k 5]`

Metrics
- retrieval hit@k   : expected (doc, page) appears in the top-k retrieved chunks
- citation accuracy : answered questions whose citations include the expected (doc, page)
- keyword accuracy  : answered questions whose text contains every expected keyword
- abstention        : unanswerable questions correctly refused / answerable wrongly refused

Runs with the extractive mock LLM when ANTHROPIC_API_KEY is unset, so it works offline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from rag.config import Settings
from rag.service import RagService, build_index


@dataclass(frozen=True)
class EvalItem:
    id: str
    question: str
    answerable: bool
    expected_doc: str | None = None
    expected_page: int | None = None
    answer_keywords: tuple[str, ...] = ()


@dataclass
class EvalReport:
    total: int
    hit_at_k: float
    citation_accuracy: float
    keyword_accuracy: float
    unanswerable_abstained: float
    answerable_false_abstain: float
    rows: list[dict[str, object]]


def load_questions(path: Path) -> list[EvalItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        items.append(
            EvalItem(
                id=raw["id"],
                question=raw["question"],
                answerable=raw.get("answerable", True),
                expected_doc=raw.get("expected_doc"),
                expected_page=raw.get("expected_page"),
                answer_keywords=tuple(raw.get("answer_keywords", [])),
            )
        )
    return items


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0


def run_eval(service: RagService, items: list[EvalItem], k: int) -> EvalReport:
    rows: list[dict[str, object]] = []
    hits = cited = keywords_ok = answered = 0
    unanswerable = abstained_correctly = answerable = false_abstain = 0

    for item in items:
        retrieved = service.retriever.search(item.question, k=k)
        result = service.ask(item.question, k=k)
        expected = (item.expected_doc, item.expected_page)
        retrieved_ok = any((h.chunk.doc, h.chunk.page) == expected for h in retrieved)
        cited_ok = any((c.doc, c.page) == expected for c in result.citations)
        text = result.answer.lower()
        keywords_hit = all(kw.lower() in text for kw in item.answer_keywords)

        if item.answerable:
            answerable += 1
            hits += retrieved_ok
            if result.abstained:
                false_abstain += 1
            else:
                answered += 1
                cited += cited_ok
                keywords_ok += keywords_hit
        else:
            unanswerable += 1
            abstained_correctly += result.abstained

        rows.append(
            {
                "id": item.id,
                "answerable": item.answerable,
                "retrieved": retrieved_ok,
                "abstained": result.abstained,
                "cited": cited_ok,
                "keywords": keywords_hit if item.answerable else None,
                "confidence": result.confidence,
            }
        )

    return EvalReport(
        total=len(items),
        hit_at_k=_ratio(hits, answerable),
        citation_accuracy=_ratio(cited, answered),
        keyword_accuracy=_ratio(keywords_ok, answered),
        unanswerable_abstained=_ratio(abstained_correctly, unanswerable),
        answerable_false_abstain=_ratio(false_abstain, answerable),
        rows=rows,
    )


def print_report(report: EvalReport, k: int, llm_name: str) -> None:
    print(f"\nEval over {report.total} questions (LLM: {llm_name}, k={k})\n")
    print(f"{'id':<22}{'retrieved':<11}{'abstained':<11}{'cited':<8}{'kw':<7}conf")
    for row in report.rows:
        kw = "-" if row["keywords"] is None else str(row["keywords"])
        print(
            f"{row['id']:<22}{str(row['retrieved']):<11}{str(row['abstained']):<11}"
            f"{str(row['cited']):<8}{kw:<7}{row['confidence']:.2f}"
        )
    print()
    print(f"retrieval hit@{k}:          {report.hit_at_k:.0%}")
    print(f"citation accuracy:         {report.citation_accuracy:.0%}")
    print(f"keyword accuracy:          {report.keyword_accuracy:.0%}")
    print(f"unanswerable -> abstained: {report.unanswerable_abstained:.0%}")
    print(f"answerable -> abstained:   {report.answerable_false_abstain:.0%} (lower is better)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--questions", type=Path, default=Path("evals/questions.jsonl"))
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--no-ingest", action="store_true", help="Reuse the existing index")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if not args.no_ingest:
        build_index(settings)
    service = RagService(settings)
    k = args.k or settings.top_k
    report = run_eval(service, load_questions(args.questions), k)
    print_report(report, k, type(service.llm).__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
