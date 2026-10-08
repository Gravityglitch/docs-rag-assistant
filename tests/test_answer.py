from __future__ import annotations

from rag.answer import ABSTAIN_PHRASE, ExtractiveLLM, answer_question, parse_citations
from rag.ingest import Chunk
from rag.retrieval import Hit


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


def _hits() -> list[Hit]:
    chunks = [
        Chunk("h#p2#c0", "handbook.md", 2, "Employees accrue 18 days of PTO per year."),
        Chunk("f#p3#c0", "faq.docx", 3, "Rate limit is 600 requests per minute on Team."),
    ]
    return [Hit(chunk=c, score=0.03, vector_score=0.4, bm25_score=2.0) for c in chunks]


def test_cited_answer_is_parsed() -> None:
    llm = FakeLLM("Employees get 18 days of PTO per year [handbook.md p.2].")
    result = answer_question("How much PTO?", _hits(), llm, threshold=0.0, score=0.9)
    assert not result.abstained
    assert [(c.doc, c.page) for c in result.citations] == [("handbook.md", 2)]
    assert result.confidence == 0.9


def test_low_confidence_abstains_without_calling_llm() -> None:
    llm = FakeLLM("anything")
    result = answer_question("Capital of Australia?", _hits(), llm, threshold=0.35, score=0.1)
    assert result.abstained and result.answer == ABSTAIN_PHRASE and llm.calls == 0


def test_model_abstention_is_respected() -> None:
    result = answer_question("Who is the CEO?", _hits(), FakeLLM(ABSTAIN_PHRASE), 0.0, 0.9)
    assert result.abstained and result.citations == []


def test_uncited_answer_is_treated_as_unsupported() -> None:
    result = answer_question("How much PTO?", _hits(), FakeLLM("18 days."), 0.0, 0.9)
    assert result.abstained


def test_parse_citations_ignores_unknown_sources() -> None:
    cites = parse_citations("x [handbook.md p.2] y [ghost.pdf p.9] z [handbook.md p.2]", _hits())
    assert len(cites) == 1 and cites[0].page == 2


def test_extractive_llm_cites_best_sentence() -> None:
    from rag.answer import SYSTEM_PROMPT, build_user_prompt

    reply = ExtractiveLLM().complete(SYSTEM_PROMPT, build_user_prompt("PTO days per year", _hits()))
    assert "18 days" in reply and reply.endswith("[handbook.md p.2].")
