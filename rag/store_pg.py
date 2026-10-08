"""Postgres + pgvector chunk store. Selected when ``DATABASE_URL`` is set.

Schema mirrors the SQLite store; embeddings are stored as ``vector(dim)`` so the same
table can serve an ANN query (``cosine_search``) once a corpus outgrows the in-memory
matrix that the hybrid retriever uses today.
"""

from __future__ import annotations

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from rag.ingest import Chunk

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    doc TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc TEXT NOT NULL,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding vector({dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc);
"""


class PgVectorStore:
    name = "pgvector"

    def __init__(
        self, database_url: str, dim: int, connection: psycopg.Connection | None = None
    ) -> None:
        self.dim = dim
        self.conn = connection or psycopg.connect(database_url, autocommit=False)
        with self.conn, self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        with self.conn, self.conn.cursor() as cur:
            cur.execute(_SCHEMA.format(dim=dim))

    def close(self) -> None:
        self.conn.close()

    def ping(self) -> bool:
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        except psycopg.Error:
            self.conn.rollback()
            return False

    def provider(self) -> str | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT value FROM meta WHERE key = 'provider'")
            row = cur.fetchone()
        return row[0] if row else None

    def set_provider(self, provider: str) -> None:
        with self.conn, self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO meta (key, value) VALUES ('provider', %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (provider,),
            )

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def doc_hashes(self) -> dict[str, str]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT doc, content_hash FROM documents")
            return dict(cur.fetchall())

    def upsert_document(
        self, doc: str, content_hash: str, chunks: list[Chunk], embeddings: np.ndarray
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if any(c.doc != doc for c in chunks):
            raise ValueError("all chunks must belong to the document being upserted")
        with self.conn, self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc = %s", (doc,))
            cur.executemany(
                "INSERT INTO chunks (chunk_id, doc, page, text, embedding) "
                "VALUES (%s, %s, %s, %s, %s)",
                [
                    (c.chunk_id, c.doc, c.page, c.text, np.asarray(vec, dtype=np.float32))
                    for c, vec in zip(chunks, embeddings, strict=True)
                ],
            )
            cur.execute(
                "INSERT INTO documents (doc, content_hash, indexed_at) VALUES (%s, %s, now()) "
                "ON CONFLICT (doc) DO UPDATE SET content_hash = EXCLUDED.content_hash, "
                "indexed_at = now()",
                (doc, content_hash),
            )

    def delete_document(self, doc: str) -> None:
        with self.conn, self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc = %s", (doc,))
            cur.execute("DELETE FROM documents WHERE doc = %s", (doc,))

    def clear(self) -> None:
        with self.conn, self.conn.cursor() as cur:
            cur.execute("TRUNCATE chunks, documents, meta")

    def load_matrix(self) -> tuple[list[Chunk], np.ndarray]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_id, doc, page, text, embedding FROM chunks ORDER BY chunk_id")
            rows = cur.fetchall()
        chunks = [Chunk(*row[:4]) for row in rows]
        if not rows:
            return chunks, np.zeros((0, 0), dtype=np.float32)
        matrix = np.stack([np.asarray(row[4], dtype=np.float32) for row in rows])
        return chunks, matrix

    def cosine_search(self, query: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """ANN query in the database (not used by the hybrid retriever yet)."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, doc, page, text, 1 - (embedding <=> %s) AS score "
                "FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                (np.asarray(query, dtype=np.float32), np.asarray(query, dtype=np.float32), k),
            )
            rows = cur.fetchall()
        return [(Chunk(*row[:4]), float(row[4])) for row in rows]
