"""Runtime configuration loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    docs_dir: Path
    db_path: Path
    model: str
    embedding_provider: str
    top_k: int
    confidence_threshold: float
    chunk_size: int
    chunk_overlap: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            docs_dir=Path(os.getenv("RAG_DOCS_DIR", "docs")),
            db_path=Path(os.getenv("RAG_DB_PATH", "data/index.sqlite")),
            model=os.getenv("RAG_MODEL", "claude-sonnet-5-5"),
            embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER", "hash"),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            confidence_threshold=float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.30")),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "700")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "120")),
        )
