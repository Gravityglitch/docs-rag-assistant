"""PgVectorStore wiring against a scripted fake connection (no database needed).

The live contract test lives in test_store.py and runs only with TEST_DATABASE_URL.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import psycopg
import pytest

from rag import store_pg
from rag.ingest import Chunk
from rag.store_pg import PgVectorStore


class FakeCursor:
    def __init__(self, conn: FakeConnection) -> None:
        self.conn = conn

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        if self.conn.fail:
            raise psycopg.OperationalError("connection lost")
        self.conn.executed.append((" ".join(sql.split()), params))

    def executemany(self, sql: str, rows: list[Any]) -> None:
        self.conn.executed.append((" ".join(sql.split()), rows))

    def fetchone(self) -> Any:
        return self.conn.results.pop(0) if self.conn.results else None

    def fetchall(self) -> list[Any]:
        return self.conn.results.pop(0) if self.conn.results else []


class FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.results: list[Any] = []
        self.fail = False
        self.rolled_back = 0
        self.closed = False

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def rollback(self) -> None:
        self.rolled_back += 1

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> tuple[PgVectorStore, FakeConnection]:
    monkeypatch.setattr(store_pg, "register_vector", lambda conn: None)
    conn = FakeConnection()
    pg = PgVectorStore("postgresql://unused", dim=3, connection=conn)  # type: ignore[arg-type]
    return pg, conn


def test_schema_uses_configured_dimension(store: tuple[PgVectorStore, FakeConnection]) -> None:
    _, conn = store
    assert "vector(3)" in conn.executed[-1][0]


def test_upsert_replaces_chunks_and_records_hash(
    store: tuple[PgVectorStore, FakeConnection],
) -> None:
    pg, conn = store
    conn.executed.clear()
    chunks = [Chunk("a.md#p1#c0", "a.md", 1, "hello"), Chunk("a.md#p1#c1", "a.md", 1, "world")]
    pg.upsert_document("a.md", "hash-a", chunks, np.ones((2, 3), dtype=np.float32))
    sql = [s for s, _ in conn.executed]
    assert sql[0].startswith("DELETE FROM chunks") and conn.executed[0][1] == ("a.md",)
    assert sql[1].startswith("INSERT INTO chunks") and len(conn.executed[1][1]) == 2
    assert "ON CONFLICT (doc)" in sql[2] and conn.executed[2][1] == ("a.md", "hash-a")

    with pytest.raises(ValueError):
        pg.upsert_document("a.md", "h", chunks, np.ones((1, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        pg.upsert_document("b.md", "h", chunks, np.ones((2, 3), dtype=np.float32))


def test_reads_map_rows_to_chunks(store: tuple[PgVectorStore, FakeConnection]) -> None:
    pg, conn = store
    conn.results = [
        ("voyage",),
        (2,),
        [("a.md", "h1"), ("b.md", "h2")],
        [
            ("a.md#p1#c0", "a.md", 1, "hello", [1.0, 0.0, 0.0]),
            ("b.md#p2#c0", "b.md", 2, "x", [0.0, 1.0, 0.0]),
        ],
        [("a.md#p1#c0", "a.md", 1, "hello", 0.9)],
    ]
    assert pg.provider() == "voyage"
    assert pg.count() == 2
    assert pg.doc_hashes() == {"a.md": "h1", "b.md": "h2"}
    chunks, matrix = pg.load_matrix()
    assert [c.chunk_id for c in chunks] == ["a.md#p1#c0", "b.md#p2#c0"]
    assert matrix.shape == (2, 3) and matrix.dtype == np.float32
    hits = pg.cosine_search(np.array([1.0, 0.0, 0.0]), k=1)
    assert hits[0][0].doc == "a.md" and hits[0][1] == 0.9
    assert "<=>" in conn.executed[-1][0]

    conn.results = []
    assert pg.provider() is None and pg.count() == 0
    assert pg.load_matrix()[1].shape == (0, 0)


def test_mutations_and_ping(store: tuple[PgVectorStore, FakeConnection]) -> None:
    pg, conn = store
    conn.executed.clear()
    pg.set_provider("hash")
    pg.delete_document("a.md")
    pg.clear()
    sql = [s for s, _ in conn.executed]
    assert "INSERT INTO meta" in sql[0]
    assert sql[1:3] == ["DELETE FROM chunks WHERE doc = %s", "DELETE FROM documents WHERE doc = %s"]
    assert sql[3].startswith("TRUNCATE")

    assert pg.ping() is True
    conn.fail = True
    assert pg.ping() is False and conn.rolled_back == 1
    pg.close()
    assert conn.closed
