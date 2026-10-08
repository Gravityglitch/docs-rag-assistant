"""Pluggable embedding providers.

- `hash`   : deterministic feature-hashing of unigrams+bigrams (offline; used in tests)
- `voyage` : Voyage AI `voyage-3-lite` (Anthropic's recommended embedding partner)
- `openai` : OpenAI `text-embedding-3-small`

Remote providers use urllib so the project has no extra HTTP dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from typing import Protocol

import numpy as np

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


class _JsonEmbedder:
    name = "remote"
    dim = 0
    url = ""
    model = ""
    api_key_env = ""

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set")
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    def embed(self, texts: list[str]) -> np.ndarray:
        body = self._post({"model": self.model, "input": texts})
        data = body["data"]
        assert isinstance(data, list)
        vectors = np.array([item["embedding"] for item in data], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)


class VoyageEmbedder(_JsonEmbedder):
    name = "voyage"
    dim = 512
    url = "https://api.voyageai.com/v1/embeddings"
    model = "voyage-3-lite"
    api_key_env = "VOYAGE_API_KEY"


class OpenAIEmbedder(_JsonEmbedder):
    name = "openai"
    dim = 1536
    url = "https://api.openai.com/v1/embeddings"
    model = "text-embedding-3-small"
    api_key_env = "OPENAI_API_KEY"


def get_embedder(provider: str) -> EmbeddingProvider:
    providers: dict[str, type[EmbeddingProvider]] = {
        "hash": HashEmbedder,
        "voyage": VoyageEmbedder,
        "openai": OpenAIEmbedder,
    }
    try:
        return providers[provider]()
    except KeyError as exc:
        raise ValueError(f"Unknown embedding provider: {provider!r}") from exc
