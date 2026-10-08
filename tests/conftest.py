from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag import api
from rag.config import Settings
from rag.service import build_index

ROOT = Path(__file__).resolve().parent.parent


class FakeLLM:
    """Scripted LLM: records calls, returns a fixed reply, streams it word by word."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0
        self.histories: list[list[dict[str, str]]] = []

    def complete(self, system: str, user: str, history: list[dict[str, str]] | None = None) -> str:
        self.calls += 1
        self.histories.append(list(history or []))
        return self.reply

    def stream(
        self, system: str, user: str, history: list[dict[str, str]] | None = None
    ) -> Iterator[str]:
        text = self.complete(system, user, history)
        for i, word in enumerate(text.split(" ")):
            yield word if i == text.count(" ") else word + " "


def make_settings(docs_dir: Path, db_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "docs_dir": docs_dir,
        "db_path": db_path,
        "model": "claude-sonnet-5-5",
        "embedding_provider": "hash",
        "top_k": 5,
        "confidence_threshold": 0.30,
        "chunk_size": 700,
        "chunk_overlap": 120,
        "anthropic_api_key": None,
        "database_url": None,
        "api_key": None,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("RAG_") or name in {
            "ANTHROPIC_API_KEY",
            "VOYAGE_API_KEY",
            "OPENAI_API_KEY",
            "DATABASE_URL",
        }:
            monkeypatch.delenv(name, raising=False)
    api.get_settings.cache_clear()
    api.get_service.cache_clear()


@pytest.fixture(scope="session")
def docs_dir() -> Path:
    return ROOT / "docs"


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory, docs_dir: Path) -> Settings:
    db_path = tmp_path_factory.mktemp("index") / "index.sqlite"
    settings = make_settings(docs_dir, db_path)
    build_index(settings)
    return settings


@pytest.fixture
def api_env(tmp_path: Path, docs_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the API at the sample docs and a fresh SQLite file; returns the db path."""
    db_path = tmp_path / "index.sqlite"
    monkeypatch.setenv("RAG_DOCS_DIR", str(docs_dir))
    monkeypatch.setenv("RAG_DB_PATH", str(db_path))
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("RAG_LOG_JSON", "false")
    return db_path


@pytest.fixture
def client(api_env: Path) -> Iterator[TestClient]:
    with TestClient(api.app) as test_client:
        yield test_client
