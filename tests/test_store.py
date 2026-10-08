from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from rag.config import Settings
from rag.ingest import Chunk
from rag.store import ChunkStore, SQLiteStore, get_store


def _chunks(doc: str, n: int) -> list[Chunk]:
    return [Chunk(f"{doc}#p1#c{i}", doc, 1, f"text {i} of {doc}") for i in range(n)]


def _vectors(n: int, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(n)
    return rng.standard_normal((n, dim)).astype(np.float32)


def store_contract(store: ChunkStore) -> None:
    """Behaviour every store must share; run against SQLite always, pgvector when available."""
    store.clear()
    assert store.ping()
    assert store.provider() is None and store.count() == 0 and store.doc_hashes() == {}
    chunks, matrix = store.load_matrix()
    assert chunks == [] and matrix.shape[0] == 0

    store.set_provider("hash")
    store.upsert_document("a.md", "hash-a", _chunks("a.md", 3), _vectors(3))
    store.upsert_document("b.md", "hash-b", _chunks("b.md", 2), _vectors(2))
    assert store.provider() == "hash"
    assert store.count() == 5
    assert store.doc_hashes() == {"a.md": "hash-a", "b.md": "hash-b"}

    chunks, matrix = store.load_matrix()
    assert [c.chunk_id for c in chunks] == sorted(c.chunk_id for c in chunks)
    assert matrix.shape == (5, 4) and matrix.dtype == np.float32
    assert np.allclose(matrix[0], _vectors(3)[0])

    # Re-upserting a document replaces its chunks instead of duplicating them.
    store.upsert_document("a.md", "hash-a2", _chunks("a.md", 1), _vectors(1))
    assert store.count() == 3 and store.doc_hashes()["a.md"] == "hash-a2"

    store.delete_document("b.md")
    assert store.count() == 1 and "b.md" not in store.doc_hashes()

    with pytest.raises(ValueError):
        store.upsert_document("c.md", "h", _chunks("c.md", 2), _vectors(1))
    with pytest.raises(ValueError):
        store.upsert_document("c.md", "h", _chunks("other.md", 1), _vectors(1))

    store.clear()
    assert store.count() == 0 and store.provider() is None


def test_sqlite_store_contract(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "nested" / "index.sqlite")
    try:
        store_contract(store)
    finally:
        store.close()
    assert not store.ping()


def test_get_store_selects_backend(tmp_path: Path) -> None:
    settings = Settings(db_path=tmp_path / "x.sqlite", database_url=None)
    store = get_store(settings, dim=4)
    assert isinstance(store, SQLiteStore)
    store.close()


@pytest.fixture
def pg_store() -> Iterator[ChunkStore]:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set; skipping live pgvector test")
    from rag.store_pg import PgVectorStore

    store = PgVectorStore(url, dim=4)
    try:
        yield store
    finally:
        store.clear()
        store.close()


@pytest.mark.pg
def test_pgvector_store_contract(pg_store: ChunkStore) -> None:
    store_contract(pg_store)
