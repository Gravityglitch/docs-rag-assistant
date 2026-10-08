from __future__ import annotations

from pathlib import Path

from rag.ingest import Page, _split_words, chunk_pages, ingest_directory, load_document


def test_split_words_overlaps() -> None:
    text = " ".join(f"w{i}" for i in range(200))
    chunks = _split_words(text, chunk_size=100, overlap=30)
    assert len(chunks) > 1
    first_tail = chunks[0].split()[-3:]
    assert all(word in chunks[1].split() for word in first_tail)


def test_chunk_ids_carry_doc_and_page() -> None:
    pages = [Page(doc="a.pdf", page=3, text="hello world " * 200)]
    chunks = chunk_pages(pages, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(c.doc == "a.pdf" and c.page == 3 for c in chunks)
    assert chunks[0].chunk_id == "a.pdf#p3#c0"


def test_loaders_produce_pages(docs_dir: Path) -> None:
    pdf = load_document(docs_dir / "refund_policy.pdf")
    docx = load_document(docs_dir / "product_faq.docx")
    md = load_document(docs_dir / "employee_handbook.md")
    assert [p.page for p in pdf] == [1, 2, 3]
    assert "30-day money-back" in pdf[0].text
    assert [p.page for p in docx] == [1, 2, 3, 4, 5]
    assert docx[2].text.startswith("API and Rate Limits")
    assert len(md) == 6 and md[1].text.startswith("# Paid Time Off")


def test_ingest_directory_covers_all_docs(docs_dir: Path) -> None:
    chunks = ingest_directory(docs_dir)
    docs = {c.doc for c in chunks}
    assert docs == {"refund_policy.pdf", "product_faq.docx", "employee_handbook.md"}
