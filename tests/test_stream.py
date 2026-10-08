from __future__ import annotations

import json

from fastapi.testclient import TestClient

from rag import api
from rag.answer import ABSTAIN_PHRASE
from rag.config import Settings
from rag.service import RagService
from tests.conftest import FakeLLM


def _parse_sse(text: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for block in text.replace("\r\n", "\n").strip().split("\n\n"):
        event, data = None, []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data.append(line.split(":", 1)[1].strip())
        if event:
            events.append((event, json.loads("\n".join(data))))
    return events


def test_service_stream_yields_meta_deltas_done(settings: Settings) -> None:
    llm = FakeLLM("Employees get 18 days [employee_handbook.md p.2].")
    events = list(RagService(settings, llm=llm).ask_stream("How many PTO days per year?"))
    assert events[0][0] == "meta" and events[-1][0] == "done"
    deltas = [data for event, data in events if event == "delta"]
    assert "".join(deltas) == llm.reply
    assert not events[-1][1].abstained and events[-1][1].citations[0].page == 2


def test_service_stream_abstains_early_on_low_confidence(settings: Settings) -> None:
    llm = FakeLLM("never called")
    events = list(RagService(settings, llm=llm).ask_stream("What is the capital of Australia?"))
    assert [e for e, _ in events] == ["meta", "done"]
    assert events[-1][1].abstained and llm.calls == 0


def test_uncited_stream_is_marked_abstained_in_done(settings: Settings) -> None:
    events = list(RagService(settings, llm=FakeLLM("18 days.")).ask_stream("How many PTO days?"))
    done = events[-1][1]
    assert done.abstained and done.answer == ABSTAIN_PHRASE


def test_sse_endpoint_shape(client: TestClient) -> None:
    client.post("/ingest")
    api.get_service().llm = FakeLLM("Employees get 18 days [employee_handbook.md p.2].")
    with client.stream(
        "POST", "/ask/stream?debug=true", json={"question": "How many PTO days per year?"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse("".join(response.iter_text()))
    names = [name for name, _ in events]
    assert names[0] == "meta" and names[-1] == "done" and names.count("delta") >= 2
    assert set(events[0][1]) == {"confidence", "threshold"}
    done = events[-1][1]
    assert set(done) == {"answer", "citations", "confidence", "abstained", "diagnostics"}
    assert "".join(d["text"] for name, d in events if name == "delta") == done["answer"]


def test_sse_endpoint_requires_index(client: TestClient) -> None:
    assert client.post("/ask/stream", json={"question": "How many PTO days?"}).status_code == 409
