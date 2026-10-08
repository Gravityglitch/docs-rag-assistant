"""Runtime configuration: a pydantic-settings model validated once at startup.

Every field maps to an environment variable (prefix ``RAG_`` unless noted), with an
optional ``.env`` file for local development. ``.env.example`` documents every variable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProviderName = Literal["auto", "hash", "voyage", "openai"]
RerankerName = Literal["llm", "cross-encoder"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- storage -------------------------------------------------------------------
    docs_dir: Path = Path("docs")
    db_path: Path = Path("data/index.sqlite")
    database_url: str | None = Field(
        default=None,
        validation_alias="DATABASE_URL",
        description="postgresql://... selects the pgvector store; unset keeps SQLite.",
    )

    # --- models / providers ----------------------------------------------------------
    model: str = "claude-sonnet-5-5"
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    voyage_api_key: SecretStr | None = Field(default=None, validation_alias="VOYAGE_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    embedding_provider: EmbeddingProviderName = "auto"
    embed_batch_size: int = Field(default=64, ge=1, le=512)

    # --- retrieval -----------------------------------------------------------------
    top_k: int = Field(default=5, ge=1, le=50)
    candidates: int = Field(default=20, ge=1, le=200)
    confidence_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    rerank: bool = False
    reranker: RerankerName = "llm"
    rerank_candidates: int = Field(default=20, ge=1, le=100)
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = Field(default=700, ge=100, le=5000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)

    # --- api -------------------------------------------------------------------------
    api_key: SecretStr | None = Field(
        default=None, description="X-API-Key required on /ask and /ingest when set."
    )
    history_turns: int = Field(default=6, ge=0, le=50)
    max_request_bytes: int = Field(default=64_000, ge=1_000)
    rate_limit: str = "60/minute"
    metrics_enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    # --- reliability / observability -------------------------------------------------
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)
    llm_max_tokens: int = Field(default=1024, ge=64, le=16000)
    embed_timeout_seconds: float = Field(default=30.0, gt=0)
    embed_max_retries: int = Field(default=3, ge=0, le=10)
    log_json: bool = True
    log_level: str = "INFO"

    @field_validator("database_url")
    @classmethod
    def _blank_is_none(cls, value: str | None) -> str | None:
        return value or None

    @field_validator("anthropic_api_key", "voyage_api_key", "openai_api_key", "api_key")
    @classmethod
    def _blank_secret_is_none(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            return None
        return value

    @model_validator(mode="after")
    def _cross_field_checks(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        if self.database_url and not self.database_url.startswith(("postgres://", "postgresql://")):
            raise ValueError("DATABASE_URL must be a postgresql:// URL")
        return self

    @property
    def resolved_embedding_provider(self) -> Literal["hash", "voyage", "openai"]:
        """``auto`` picks Voyage, then OpenAI, by whichever key is present; else hash."""
        if self.embedding_provider != "auto":
            return self.embedding_provider
        if self.voyage_api_key is not None:
            return "voyage"
        if self.openai_api_key is not None:
            return "openai"
        return "hash"

    @property
    def use_postgres(self) -> bool:
        return self.database_url is not None

    @property
    def auth_enabled(self) -> bool:
        return self.api_key is not None

    @classmethod
    def from_env(cls) -> Settings:
        """Build and validate settings from the environment (raises on invalid values)."""
        return cls()
