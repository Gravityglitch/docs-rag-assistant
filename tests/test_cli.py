from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.cli import main


def test_cli_ingest_and_ask(
    tmp_path: Path,
    docs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("RAG_DOCS_DIR", str(docs_dir))
    monkeypatch.setenv("RAG_DB_PATH", str(tmp_path / "index.sqlite"))
    monkeypatch.setenv("RAG_LOG_JSON", "false")

    assert main(["ingest"]) == 0
    assert "3 indexed" in capsys.readouterr().out
    assert main(["ingest"]) == 0
    assert "3 unchanged" in capsys.readouterr().out

    assert main(["ask", "How many PTO days per year?"]) == 0
    out = capsys.readouterr().out
    assert "employee_handbook.md p.2" in out and "abstained=False" in out

    assert main(["ask", "--json", "--debug", "What is the capital of Australia?"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["abstained"] and "diagnostics" in payload
