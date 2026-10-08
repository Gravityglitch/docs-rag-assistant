from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import Settings
from rag.service import build_index

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def docs_dir() -> Path:
    return ROOT / "docs"


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory, docs_dir: Path) -> Settings:
    db_path = tmp_path_factory.mktemp("index") / "index.sqlite"
    settings = Settings(
        docs_dir=docs_dir,
        db_path=db_path,
        model="claude-sonnet-5-5",
        embedding_provider="hash",
        top_k=5,
        confidence_threshold=0.30,
        chunk_size=700,
        chunk_overlap=120,
    )
    build_index(settings)
    return settings
