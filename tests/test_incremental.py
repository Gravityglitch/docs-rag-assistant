from __future__ import annotations

import shutil
from pathlib import Path

from rag.service import RagService, build_index
from rag.store import SQLiteStore
from tests.conftest import FakeLLM, make_settings


def test_reindex_skips_unchanged_prunes_removed_and_reindexes_changed(
    tmp_path: Path, docs_dir: Path
) -> None:
    corpus = tmp_path / "docs"
    shutil.copytree(docs_dir, corpus)
    settings = make_settings(corpus, tmp_path / "index.sqlite")

    first = build_index(settings)
    assert sorted(first.indexed) == [
        "employee_handbook.md",
        "product_faq.docx",
        "refund_policy.pdf",
    ]
    assert first.skipped == [] and first.pruned == [] and first.failed == {}
    baseline = first.chunks

    second = build_index(settings)
    assert second.indexed == [] and len(second.skipped) == 3 and second.chunks == baseline

    (corpus / "product_faq.docx").unlink()
    handbook = corpus / "employee_handbook.md"
    handbook.write_text(handbook.read_text() + "\n\n# Extra\n\nThe new office opens in May.\n")
    (corpus / "notes.txt").write_text("# Notes\n\nThe standup is at 9am.\n")

    third = build_index(settings)
    assert sorted(third.indexed) == ["employee_handbook.md", "notes.txt"]
    assert third.skipped == ["refund_policy.pdf"]
    assert third.pruned == ["product_faq.docx"]

    store = SQLiteStore(settings.db_path)
    try:
        docs = {c.doc for c in store.load_matrix()[0]}
        assert docs == {"employee_handbook.md", "refund_policy.pdf", "notes.txt"}
        assert set(store.doc_hashes()) == docs
    finally:
        store.close()

    full = build_index(settings, full=True)
    assert len(full.indexed) == 3 and full.skipped == []


def test_bad_document_is_logged_and_skipped(tmp_path: Path, docs_dir: Path) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    shutil.copy(docs_dir / "employee_handbook.md", corpus / "employee_handbook.md")
    (corpus / "broken.pdf").write_bytes(b"%PDF-1.4 this is not a real pdf")
    settings = make_settings(corpus, tmp_path / "index.sqlite")

    result = build_index(settings)
    assert result.indexed == ["employee_handbook.md"]
    assert list(result.failed) == ["broken.pdf"]
    assert result.chunks > 0

    service = RagService(settings, llm=FakeLLM("x [employee_handbook.md p.2]."))
    assert not service.ask("How many PTO days per year?").abstained


def test_provider_change_forces_rebuild(tmp_path: Path, docs_dir: Path) -> None:
    settings = make_settings(docs_dir, tmp_path / "index.sqlite")
    build_index(settings)
    store = SQLiteStore(settings.db_path)
    store.set_provider("voyage")
    store.close()
    result = build_index(settings)
    assert len(result.indexed) == 3 and result.provider == "hash"
