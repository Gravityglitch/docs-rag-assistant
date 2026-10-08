from __future__ import annotations

from fastapi.testclient import TestClient

from rag import api
from rag.config import Settings
from rag.service import RagService
from tests.conftest import FakeLLM


def test_service_passes_last_n_turns_to_llm(settings: Settings) -> None:
    llm = FakeLLM("18 days [employee_handbook.md p.2].")
    service = RagService(settings.model_copy(update={"history_turns": 2}), llm=llm)
    history = [
        {"role": "user", "content": f"q{i}"}
        if i % 2 == 0
        else {"role": "assistant", "content": f"a{i}"}
        for i in range(10)
    ]
    service.ask("How many PTO days per year?", history=history)
    assert llm.histories[-1] == history[-4:]


def test_history_disabled_with_zero_turns(settings: Settings) -> None:
    llm = FakeLLM("18 days [employee_handbook.md p.2].")
    service = RagService(settings.model_copy(update={"history_turns": 0}), llm=llm)
    service.ask("How many PTO days per year?", history=[{"role": "user", "content": "hi"}])
    assert llm.histories[-1] == []


def test_api_accepts_history(client: TestClient) -> None:
    client.post("/ingest")
    llm = FakeLLM("Employees get 18 days [employee_handbook.md p.2].")
    api.get_service().llm = llm
    body = {
        "question": "How many PTO days per year can I carry over?",
        "history": [
            {"role": "user", "content": "How many PTO days per year?"},
            {"role": "assistant", "content": "18 days [employee_handbook.md p.2]."},
        ],
    }
    response = client.post("/ask", json=body)
    assert response.status_code == 200
    assert llm.histories[-1] == body["history"]
