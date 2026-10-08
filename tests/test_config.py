from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.config import Settings


def test_defaults_are_sqlite_and_hash() -> None:
    settings = Settings()
    assert settings.resolved_embedding_provider == "hash"
    assert not settings.use_postgres
    assert not settings.auth_enabled
    assert settings.top_k == 5


def test_auto_provider_prefers_voyage_then_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert Settings().resolved_embedding_provider == "openai"
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    assert Settings().resolved_embedding_provider == "voyage"
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "hash")
    assert Settings().resolved_embedding_provider == "hash"


def test_blank_secrets_are_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("RAG_API_KEY", "   ")
    monkeypatch.setenv("DATABASE_URL", "")
    settings = Settings()
    assert settings.anthropic_api_key is None and settings.api_key is None
    assert settings.database_url is None


def test_env_prefix_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("RAG_RERANK", "true")
    monkeypatch.setenv("RAG_API_KEY", "secret")
    settings = Settings()
    assert settings.top_k == 7 and settings.rerank and settings.auth_enabled
    assert "secret" not in repr(settings)

    monkeypatch.setenv("RAG_TOP_K", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_cross_field_validation() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=200, chunk_overlap=300)
    with pytest.raises(ValidationError, match="postgresql"):
        Settings(database_url="mysql://x")
    assert Settings(database_url="postgresql://u:p@h/db").use_postgres
