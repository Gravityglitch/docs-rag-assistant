from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from tenacity import wait_none

from rag.answer import ABSTAIN_PHRASE, ClaudeLLM

_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _rate_limited() -> anthropic.RateLimitError:
    return anthropic.RateLimitError(
        "rate limited", response=httpx2.Response(429, request=_REQUEST), body=None
    )


def _bad_request() -> anthropic.BadRequestError:
    return anthropic.BadRequestError(
        "bad", response=httpx2.Response(400, request=_REQUEST), body=None
    )


def _message(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=120, output_tokens=30),
    )


class FakeMessages:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def _next(self, kwargs: dict[str, Any]) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def create(self, **kwargs: Any) -> Any:
        return self._next(kwargs)

    @contextmanager
    def stream(self, **kwargs: Any) -> Any:
        message = self._next(kwargs)
        yield SimpleNamespace(
            text_stream=iter(message.content[0].text.split(" ")),
            get_final_message=lambda: message,
        )


def _llm(outcomes: list[Any], max_retries: int = 3) -> tuple[ClaudeLLM, FakeMessages]:
    messages = FakeMessages(outcomes)
    client = SimpleNamespace(messages=messages)
    llm = ClaudeLLM(model="claude-sonnet-5-5", client=client, max_retries=max_retries)  # type: ignore[arg-type]
    llm.retrying.wait = wait_none()
    return llm, messages


def test_retries_rate_limit_then_succeeds() -> None:
    llm, messages = _llm([_rate_limited(), _rate_limited(), _message("ok [a.md p.1].")])
    assert llm.complete("sys", "user") == "ok [a.md p.1]."
    assert len(messages.calls) == 3
    assert messages.calls[0]["messages"] == [{"role": "user", "content": "user"}]


def test_gives_up_after_max_retries() -> None:
    llm, messages = _llm([_rate_limited()] * 5, max_retries=1)
    with pytest.raises(anthropic.RateLimitError):
        llm.complete("sys", "user")
    assert len(messages.calls) == 2


def test_non_retryable_error_is_raised_immediately() -> None:
    llm, messages = _llm([_bad_request(), _message("never")])
    with pytest.raises(anthropic.BadRequestError):
        llm.complete("sys", "user")
    assert len(messages.calls) == 1


def test_refusal_becomes_abstention_and_history_is_forwarded() -> None:
    llm, messages = _llm([_message("", stop_reason="refusal")])
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}]
    assert llm.complete("sys", "user", history) == ABSTAIN_PHRASE
    assert messages.calls[0]["messages"] == [*history, {"role": "user", "content": "user"}]


def test_stream_retries_connection_setup() -> None:
    llm, messages = _llm([anthropic.APIConnectionError(request=_REQUEST), _message("a b c")])
    assert list(llm.stream("sys", "user")) == ["a", "b", "c"]
    assert len(messages.calls) == 2
