from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag import api


@pytest.fixture
def client(tmp_path: Path, docs_dir: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("RAG_DOCS_DIR", str(docs_dir))
    monkeypatch.setenv("RAG_DB_PATH", str(tmp_path / "index.sqlite"))
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "hash")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    api.get_service.cache_clear()
    return TestClient(api.app)


def test_health_ingest_ask_flow(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"

    empty = client.post("/ask", json={"question": "How many PTO days?"})
    assert empty.status_code == 409

    ingest = client.post("/ingest")
    assert ingest.status_code == 200 and ingest.json()["chunks"] > 0

    answered = client.post("/ask", json={"question": "How many PTO days per year?"}).json()
    assert set(answered) == {"answer", "citations", "confidence", "abstained"}
    assert not answered["abstained"]
    assert answered["citations"][0]["doc"] == "employee_handbook.md"
    assert answered["citations"][0]["page"] == 2

    off_topic = client.post("/ask", json={"question": "What is the capital of Australia?"}).json()
    assert off_topic["abstained"] and off_topic["citations"] == []


def test_ask_validates_input(client: TestClient) -> None:
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
