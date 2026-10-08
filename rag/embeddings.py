"""Pluggable embedding providers.

- ``hash``   : deterministic feature-hashing of unigrams+bigrams (offline; tests only)
- ``voyage`` : Voyage AI ``voyage-3-lite`` (Anthropic's recommended embedding partner)
- ``openai`` : OpenAI ``text-embedding-3-small``

Remote providers use urllib (no extra HTTP dependency) wrapped in tenacity retries with
exponential backoff and a per-request timeout; inputs are sent in bounded batches.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from typing import Protocol

import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from rag.logging import get_logger

log = get_logger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def stem(token: str) -> str:
    """Tiny suffix stripper (Porter-lite) so 'limits'/'limit', 'retried'/'retries' agree."""
    if len(token) <= 3:
        return token
    for suffix, replacement in (
        ("ies", "i"),
        ("ied", "i"),
        ("ing", ""),
        ("ed", ""),
        ("es", ""),
        ("s", ""),
    ):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)] + replacement
            break
    return token[:-1] + "i" if token.endswith("y") and len(token) > 3 else token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN_RE.findall(text.lower())]


class EmbeddingProvider(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Feature-hashed bag of unigrams and bigrams, L2-normalised. Deterministic, no network."""

    name = "hash"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        tokens = tokenize(text)
        return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                sign = 1.0 if value & 1 else -1.0
                out[row, (value >> 1) % self.dim] += sign
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class EmbeddingError(RuntimeError):
    """Raised when a remote embedding call fails after retries."""


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 409, 429} or exc.code >= 500
    return isinstance(exc, urllib.error.URLError | TimeoutError | ConnectionError)


class _JsonEmbedder:
    name = "remote"
    dim = 0
    url = ""
    model = ""

    def __init__(
        self,
        api_key: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        batch_size: int = 64,
    ) -> None:
        if not api_key:
            raise ValueError(f"{self.name} embeddings need an API key")
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = batch_size
        self._post_with_retry = retry(
            reraise=True,
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential_jitter(initial=0.5, max=10),
            retry=retry_if_exception(is_retryable),
        )(self._post)

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        try:
            body = self._post_with_retry({"model": self.model, "input": texts})
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise EmbeddingError(f"{self.name} embedding request failed: {exc}") from exc
        data = body["data"]
        if not isinstance(data, list) or len(data) != len(texts):
            raise EmbeddingError(f"{self.name} returned {len(data)} vectors for {len(texts)} texts")
        usage = body.get("usage", {})
        log.info("embedding_usage", provider=self.name, texts=len(texts), usage=usage)
        return np.array([item["embedding"] for item in data], dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        parts = [
            self._embed_batch(texts[i : i + self.batch_size])
            for i in range(0, len(texts), self.batch_size)
        ]
        vectors = np.concatenate(parts)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)


class VoyageEmbedder(_JsonEmbedder):
    name = "voyage"
    dim = 512
    url = "https://api.voyageai.com/v1/embeddings"
    model = "voyage-3-lite"


class OpenAIEmbedder(_JsonEmbedder):
    name = "openai"
    dim = 1536
    url = "https://api.openai.com/v1/embeddings"
    model = "text-embedding-3-small"


def get_embedder(
    provider: str,
    api_key: str | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    batch_size: int = 64,
) -> EmbeddingProvider:
    if provider == "hash":
        return HashEmbedder()
    remote: dict[str, type[_JsonEmbedder]] = {"voyage": VoyageEmbedder, "openai": OpenAIEmbedder}
    if provider not in remote:
        raise ValueError(f"Unknown embedding provider: {provider!r}")
    return remote[provider](
        api_key=api_key or "", timeout=timeout, max_retries=max_retries, batch_size=batch_size
    )
