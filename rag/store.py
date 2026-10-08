"""Chunk stores: a Protocol plus the SQLite implementation (default for local and tests).

The Postgres + pgvector implementation lives in :mod:`rag.store_pg`; ``get_store`` picks
one from settings. Both keep a ``documents`` table with a content hash per source file so
re-indexing can skip unchanged documents and prune removed ones.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from rag.ingest import Chunk

if TYPE_CHECKING:
    from rag.config import Settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    doc TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc TEXT NOT NULL,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc);
"""


class ChunkStore(Protocol):
    """What the retriever and the ingest pipeline need from a backing store."""

    name: str

    def close(self) -> None: ...
    def ping(self) -> bool: ...
    def provider(self) -> str | None: ...
    def set_provider(self, provider: str) -> None: ...
    def count(self) -> int: ...
    def doc_hashes(self) -> dict[str, str]: ...
    def upsert_document(
        self, doc: str, content_hash: str, chunks: list[Chunk], embeddings: np.ndarray
    ) -> None: ...
    def delete_document(self, doc: str) -> None: ...
    def clear(self) -> None: ...
    def load_matrix(self) -> tuple[list[Chunk], np.ndarray]: ...


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class SQLiteStore:
    name = "sqlite"

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def ping(self) -> bool:
        try:
            self.conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def provider(self) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key='provider'").fetchone()
        return row[0] if row else None

    def set_provider(self, provider: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('provider', ?)", (provider,))

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def doc_hashes(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT doc, content_hash FROM documents").fetchall()
        return dict(rows)

    def upsert_document(
        self, doc: str, content_hash: str, chunks: list[Chunk], embeddings: np.ndarray
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if any(c.doc != doc for c in chunks):
            raise ValueError("all chunks must belong to the document being upserted")
        with self.conn:
            self.conn.execute("DELETE FROM chunks WHERE doc = ?", (doc,))
            self.conn.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                [
                    (c.chunk_id, c.doc, c.page, c.text, vec.astype(np.float32).tobytes())
                    for c, vec in zip(chunks, embeddings, strict=True)
                ],
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?)", (doc, content_hash, _now())
            )

    def delete_document(self, doc: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM chunks WHERE doc = ?", (doc,))
            self.conn.execute("DELETE FROM documents WHERE doc = ?", (doc,))

    def clear(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM chunks")
            self.conn.execute("DELETE FROM documents")
            self.conn.execute("DELETE FROM meta")

    def load_matrix(self) -> tuple[list[Chunk], np.ndarray]:
        rows = self.conn.execute(
            "SELECT chunk_id, doc, page, text, embedding FROM chunks ORDER BY chunk_id"
        ).fetchall()
        chunks = [Chunk(*row[:4]) for row in rows]
        if not rows:
            return chunks, np.zeros((0, 0), dtype=np.float32)
        matrix = np.stack([np.frombuffer(row[4], dtype=np.float32) for row in rows])
        return chunks, matrix


# Backwards-compatible alias for the original class name.
VectorStore = SQLiteStore


def get_store(settings: Settings, dim: int) -> ChunkStore:
    """Open the store selected by settings: pgvector when DATABASE_URL is set, else SQLite."""
    if settings.database_url:
        from rag.store_pg import PgVectorStore

        return PgVectorStore(settings.database_url, dim=dim)
    return SQLiteStore(settings.db_path)
