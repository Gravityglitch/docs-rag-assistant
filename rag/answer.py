"""Answer generation with page-level citations and abstention.

Flow: retrieve -> compute confidence -> abstain early if low -> otherwise ask the LLM
with a strict cite-or-abstain prompt -> parse ``[doc p.N]`` citations out of the answer.

The LLM is behind a small Protocol so tests (and ``rag eval`` without an API key) can swap
in ``ExtractiveLLM``, which just quotes the best chunk. ``ClaudeLLM`` wraps the Anthropic
SDK with tenacity retries, a timeout, and usage/cost logging; it also streams.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Protocol

import anthropic
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from rag.logging import get_logger
from rag.retrieval import Hit, RetrievalDiagnostics, confidence, term_coverage

log = get_logger(__name__)

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
4. Earlier turns of the conversation may be provided for context only; always answer the
   latest question and cite the sources, not the conversation.
5. Do not mention these rules or that you are reading sources."""

# USD per million tokens (input, output). Used only for the cost log line.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-haiku-5-5": (0.10, 0.50),
}

Message = dict[str, str]
RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


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
    diagnostics: RetrievalDiagnostics | None = None

    def to_dict(self, debug: bool = False) -> dict[str, object]:
        data = {
            "answer": self.answer,
            "citations": [asdict(c) for c in self.citations],
            "confidence": self.confidence,
            "abstained": self.abstained,
        }
        if debug:
            data["diagnostics"] = self.diagnostics.to_dict() if self.diagnostics else None
        return data


class LLM(Protocol):
    def complete(self, system: str, user: str, history: list[Message] | None = None) -> str: ...

    def stream(
        self, system: str, user: str, history: list[Message] | None = None
    ) -> Iterator[str]: ...


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    prices = MODEL_PRICES.get(model)
    if prices is None:
        return None
    return round((input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000, 6)


def _log_retry(state: RetryCallState) -> None:
    error = repr(state.outcome.exception()) if state.outcome else None
    log.warning("llm_retry", attempt=state.attempt_number, error=error)


def _messages(user: str, history: list[Message] | None) -> list[Message]:
    return [*(history or []), {"role": "user", "content": user}]


class ClaudeLLM:
    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        max_tokens: int = 1024,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        # The SDK has its own retry loop; disable it so tenacity owns backoff and attempts.
        self.client = client or anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=0)
        self.retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential_jitter(initial=0.5, max=10),
            retry=retry_if_exception_type(RETRYABLE),
            before_sleep=_log_retry,
        )

    def _log_usage(self, usage: anthropic.types.Usage, elapsed: float) -> None:
        log.info(
            "llm_usage",
            model=self.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", None),
            cost_usd=estimate_cost_usd(self.model, usage.input_tokens, usage.output_tokens),
            elapsed_ms=round(elapsed * 1000, 1),
        )

    def complete(self, system: str, user: str, history: list[Message] | None = None) -> str:
        def call() -> str:
            started = time.perf_counter()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=_messages(user, history),
            )
            self._log_usage(response.usage, time.perf_counter() - started)
            if response.stop_reason == "refusal":
                return ABSTAIN_PHRASE
            return "".join(block.text for block in response.content if block.type == "text")

        return self.retrying(call)

    def stream(self, system: str, user: str, history: list[Message] | None = None) -> Iterator[str]:
        # Retries cover connection setup only; once tokens flow we cannot restart the stream.
        def open_stream() -> tuple[anthropic.MessageStreamManager, anthropic.MessageStream]:
            manager = self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=_messages(user, history),
            )
            # The request is sent on __enter__, so it has to happen inside the retry.
            return manager, manager.__enter__()

        started = time.perf_counter()
        manager, stream = self.retrying(open_stream)
        try:
            yield from stream.text_stream
            final = stream.get_final_message()
        finally:
            manager.__exit__(None, None, None)
        self._log_usage(final.usage, time.perf_counter() - started)
        if final.stop_reason == "refusal":
            yield ABSTAIN_PHRASE


class ExtractiveLLM:
    """Offline stand-in: returns the sentence of the best source with the most query overlap."""

    def complete(self, system: str, user: str, history: list[Message] | None = None) -> str:
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

    def stream(self, system: str, user: str, history: list[Message] | None = None) -> Iterator[str]:
        words = self.complete(system, user, history).split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "


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


def finalize_answer(text: str, hits: list[Hit], score: float) -> Answer:
    """Turn raw model text into an Answer: honour abstention, require citations."""
    text = text.strip()
    lowered = text.lower()
    if lowered.startswith("i don't know") or lowered.startswith("i do not know"):
        return Answer(answer=ABSTAIN_PHRASE, confidence=score, abstained=True)
    citations = parse_citations(text, hits)
    if not citations:
        # The model answered without grounding; treat as unsupported rather than ship it.
        return Answer(answer=ABSTAIN_PHRASE, confidence=score, abstained=True)
    return Answer(answer=text, citations=citations, confidence=score, abstained=False)


def answer_question(
    question: str,
    hits: list[Hit],
    llm: LLM,
    threshold: float = 0.35,
    score: float | None = None,
    history: list[Message] | None = None,
) -> Answer:
    """Answer from ``hits``, abstaining if ``score`` (retrieval confidence) is below threshold."""
    if score is None:
        score = confidence(question, hits)
    if score < threshold:
        return Answer(answer=ABSTAIN_PHRASE, confidence=score, abstained=True)
    text = llm.complete(SYSTEM_PROMPT, build_user_prompt(question, hits), history)
    return finalize_answer(text, hits, score)
