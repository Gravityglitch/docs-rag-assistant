from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from typing import Any

import numpy as np
import pytest
from tenacity import wait_none

from rag import embeddings
from rag.embeddings import EmbeddingError, VoyageEmbedder, get_embedder, is_retryable


class _Response(BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _ok(n: int, dim: int) -> _Response:
    body = {"data": [{"embedding": [1.0] * dim} for _ in range(n)], "usage": {"total_tokens": 3}}
    return _Response(json.dumps(body).encode())


def test_remote_embedder_batches_retries_and_normalises(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes: list[Any] = [
        urllib.error.HTTPError("u", 503, "down", {}, None),  # type: ignore[arg-type]
        _ok(2, 4),
        _ok(1, 4),
    ]
    requests: list[dict[str, Any]] = []

    def fake_urlopen(request: Any, timeout: float) -> Any:
        requests.append(json.loads(request.data))
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", fake_urlopen)
    embedder = VoyageEmbedder(api_key="k", batch_size=2, max_retries=2)
    embedder._post_with_retry.retry.wait = wait_none()  # type: ignore[attr-defined]
    embedder.dim = 4
    vectors = embedder.embed(["a", "b", "c"])
    assert vectors.shape == (3, 4) and np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert [len(r["input"]) for r in requests] == [2, 2, 1]


def test_remote_embedder_raises_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_fail(request: Any, timeout: float) -> Any:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", always_fail)
    embedder = VoyageEmbedder(api_key="k", max_retries=1)
    embedder._post_with_retry.retry.wait = wait_none()  # type: ignore[attr-defined]
    with pytest.raises(EmbeddingError):
        embedder.embed(["a"])


def test_is_retryable_classification() -> None:
    assert is_retryable(urllib.error.HTTPError("u", 429, "x", {}, None))  # type: ignore[arg-type]
    assert not is_retryable(urllib.error.HTTPError("u", 401, "x", {}, None))  # type: ignore[arg-type]
    assert is_retryable(TimeoutError())
    assert not is_retryable(ValueError())


def test_remote_embedder_requires_key() -> None:
    with pytest.raises(ValueError):
        get_embedder("openai", api_key=None)
    assert get_embedder("openai", api_key="k").name == "openai"
    assert get_embedder("voyage", api_key="k").embed([]).shape == (0, 512)
