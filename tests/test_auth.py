from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag import api

ASK = {"question": "How many PTO days per year?"}


@pytest.fixture
def secured(api_env: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("RAG_API_KEY", "s3cret")
    api.get_settings.cache_clear()
    with TestClient(api.app) as client:
        yield client


def test_protected_routes_require_key(secured: TestClient) -> None:
    assert secured.post("/ingest").status_code == 401
    assert secured.post("/ask", json=ASK).status_code == 401
    assert secured.post("/ask/stream", json=ASK).status_code == 401
    assert secured.post("/ask", json=ASK, headers={"X-API-Key": "wrong"}).status_code == 401
    assert secured.get("/health").status_code == 200
    assert secured.get("/health").json()["auth"] is True


def test_correct_key_is_accepted(secured: TestClient) -> None:
    headers = {"X-API-Key": "s3cret"}
    assert secured.post("/ingest", headers=headers).status_code == 200
    assert secured.post("/ask", json=ASK, headers=headers).status_code == 200


def test_auth_disabled_when_no_key_configured(client: TestClient) -> None:
    assert client.get("/health").json()["auth"] is False
    assert client.post("/ingest").status_code == 200
