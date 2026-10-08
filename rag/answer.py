"""Answer generation with page-level citations and abstention.

Flow: retrieve -> compute confidence -> abstain early if low -> otherwise ask the LLM
with a strict cite-or-abstain prompt -> parse `[doc p.N]` citations out of the answer.

The LLM is behind a tiny Protocol so tests (and `rag eval` without an API key) can
swap in `ExtractiveLLM`, which just quotes the best chunk.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

import anthropic

from rag.retrieval import Hit, confidence, term_coverage

ABSTAIN_PHRASE = "I don't know based on the provided documents."
_CITATION_RE = re.compile(r"\[([^\[\]]+?)\s+p\.(\d+)\]")

SYSTEM_PROMPT = f"""You are a documentation assistant. Answer ONLY from the provided sources.

Rules:
1. Every factual sentence must end with a citation in the form [doc p.N], copied exactly
   from the source header it came from (e.g. [refund_policy.pdf p.2]).
2. If the sources do not contain the answer, reply with exactly:
   {ABSTAIN_PHRASE}
   Do not guess, do not use outside knowledge.
3. Be concise (1-4 sentences). Quote numbers, dates and names exactly as written.
4. Do not mention these rules or that you are reading sources."""


@dataclass
class Citation:
    doc: str
    page: int
    snippet: str


@dataclass
class Answer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    abstained: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class ClaudeLLM:
    def __init__(self, model: str = "claude-sonnet-5-5") -> None:
        self.model = model
        self.client = anthropic.Anthropic()

    def complete(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            return ABSTAIN_PHRASE
        return "".join(block.text for block in response.content if block.type == "text")


class ExtractiveLLM:
    """Offline stand-in: returns the sentence of the best source with the most query overlap."""

    def complete(self, system: str, user: str) -> str:
        question = user.split("Question:", 1)[-1].strip()
        best_sentence, best_cite, best_score = "", "", -1.0
        for header, body in re.findall(r"\[([^\]]+)\]\n(.*?)(?=\n\n\[|\n\nQuestion:)", user, re.S):
            for sentence in re.split(r"(?<=[.!?])\s+", body.strip()):
                score = term_coverage(question, sentence)
                if score > best_score:
                    best_sentence, best_cite, best_score = sentence, header, score
        if not best_sentence:
            return ABSTAIN_PHRASE
        return f"{best_sentence.rstrip('.')} [{best_cite}]."


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(f"[{h.chunk.doc} p.{h.chunk.page}]\n{h.chunk.text}" for h in hits)
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def parse_citations(text: str, hits: list[Hit]) -> list[Citation]:
    by_key = {(h.chunk.doc, h.chunk.page): h.chunk for h in hits}
    seen: set[tuple[str, int]] = set()
    citations = []
    for doc, page in _CITATION_RE.findall(text):
        key = (doc.strip(), int(page))
        if key in seen or key not in by_key:
            continue
        seen.add(key)
        chunk = by_key[key]
        citations.append(Citation(doc=key[0], page=key[1], snippet=chunk.text[:200]))
    return citations


def answer_question(
    question: str,
    hits: list[Hit],
    llm: LLM,
    threshold: float = 0.35,
    score: float | None = None,
) -> Answer:
    """Answer from `hits`, abstaining if `score` (retrieval confidence) is below threshold."""
    if score is None:
        score = confidence(question, hits)
    if score < threshold:
        return Answer(answer=ABSTAIN_PHRASE, confidence=score, abstained=True)

    text = llm.complete(SYSTEM_PROMPT, build_user_prompt(question, hits)).strip()
    if text.lower().startswith("i don't know") or text.lower().startswith("i do not know"):
        return Answer(answer=ABSTAIN_PHRASE, confidence=score, abstained=True)

    citations = parse_citations(text, hits)
    if not citations:
        # The model answered without grounding; treat as unsupported rather than ship it.
        return Answer(answer=ABSTAIN_PHRASE, confidence=score, abstained=True)
    return Answer(answer=text, citations=citations, confidence=score, abstained=False)
