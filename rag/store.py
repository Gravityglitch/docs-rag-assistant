"""SQLite-backed chunk store with numpy cosine search. No vector DB needed for small corpora."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from rag.ingest import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc TEXT NOT NULL,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
"""


class VectorStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def replace_all(self, chunks: list[Chunk], embeddings: np.ndarray, provider: str) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        with self.conn:
            self.conn.execute("DELETE FROM chunks")
            self.conn.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                [
                    (c.chunk_id, c.doc, c.page, c.text, vec.astype(np.float32).tobytes())
                    for c, vec in zip(chunks, embeddings, strict=True)
                ],
            )
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('provider', ?)", (provider,))

    def provider(self) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key='provider'").fetchone()
        return row[0] if row else None

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def all_chunks(self) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT chunk_id, doc, page, text FROM chunks ORDER BY chunk_id"
        ).fetchall()
        return [Chunk(*row) for row in rows]

    def load_matrix(self) -> tuple[list[Chunk], np.ndarray]:
        rows = self.conn.execute(
            "SELECT chunk_id, doc, page, text, embedding FROM chunks ORDER BY chunk_id"
        ).fetchall()
        chunks = [Chunk(*row[:4]) for row in rows]
        if not rows:
            return chunks, np.zeros((0, 0), dtype=np.float32)
        matrix = np.stack([np.frombuffer(row[4], dtype=np.float32) for row in rows])
        return chunks, matrix

    def cosine_search(self, query: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        chunks, matrix = self.load_matrix()
        if not chunks:
            return []
        scores = matrix @ query.astype(np.float32)
        top = np.argsort(-scores)[:k]
        return [(chunks[i], float(scores[i])) for i in top]
