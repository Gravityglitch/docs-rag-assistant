from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag import api
from rag.answer import ABSTAIN_PHRASE


def test_health_ready_ingest_ask_flow(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200 and health.json()["status"] == "ok"
    assert health.json()["store"] == "sqlite"
    assert health.headers["X-Request-ID"]

    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["index_empty"] is True

    empty = client.post("/ask", json={"question": "How many PTO days?"})
    assert empty.status_code == 409

    ingest = client.post("/ingest")
    assert ingest.status_code == 200 and ingest.json()["chunks"] > 0
    assert len(ingest.json()["indexed"]) == 3
    assert client.post("/ingest").json()["skipped"] == ingest.json()["indexed"]
    assert client.get("/ready").json()["index_empty"] is False

    answered = client.post("/ask", json={"question": "How many PTO days per year?"}).json()
    assert set(answered) == {"answer", "citations", "confidence", "abstained"}
    assert not answered["abstained"]
    assert answered["citations"][0]["doc"] == "employee_handbook.md"
    assert answered["citations"][0]["page"] == 2

    off_topic = client.post("/ask", json={"question": "What is the capital of Australia?"}).json()
    assert off_topic["abstained"] and off_topic["citations"] == []
    assert off_topic["answer"] == ABSTAIN_PHRASE


def test_debug_returns_retrieval_diagnostics(client: TestClient) -> None:
    client.post("/ingest")
    body = client.post("/ask?debug=true", json={"question": "How many PTO days per year?"}).json()
    diag = body["diagnostics"]
    assert diag["reranked"] is False and diag["threshold"] == 0.30
    assert diag["chunks_indexed"] > 0 and len(diag["hits"]) == 5
    assert {"chunk_id", "doc", "page", "fused", "vector", "bm25", "rerank"} <= set(diag["hits"][0])
    assert diag["confidence"] == body["confidence"]
    assert diag["retrieval_ms"] >= 0 and diag["llm_ms"] >= 0


def test_ask_validates_input(client: TestClient) -> None:
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "valid q", "top_k": 99}).status_code == 422
    bad_history = {"question": "valid q", "history": [{"role": "system", "content": "x"}]}
    assert client.post("/ask", json=bad_history).status_code == 422


def test_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"


def test_oversized_body_is_rejected(client: TestClient) -> None:
    big = {"question": "q" * 1000, "history": [{"role": "user", "content": "x" * 4000}] * 20}
    response = client.post(
        "/ask", content=json.dumps(big), headers={"content-type": "application/json"}
    )
    assert response.status_code == 413


def test_rate_limit_is_enforced(api_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_RATE_LIMIT", "2/minute")
    api.get_settings.cache_clear()
    api.limiter.reset()
    with TestClient(api.app) as client:
        client.post("/ingest")
        question = {"question": "How many PTO days per year?"}
        assert client.post("/ask", json=question).status_code == 200
        assert client.post("/ask", json=question).status_code == 200
        limited = client.post("/ask", json=question)
    assert limited.status_code == 429 and "Retry-After" in limited.headers
    api.limiter.reset()


def test_ready_reports_store_failure(client: TestClient) -> None:
    client.post("/ingest")
    api.get_service().store.close()
    response = client.get("/ready")
    assert response.status_code == 503 and response.json()["store_ok"] is False
