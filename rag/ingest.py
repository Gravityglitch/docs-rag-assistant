"""Document loading (PDF / DOCX / Markdown / text) and overlap chunking.

Every chunk keeps the document name and the 1-based page it came from so answers
can cite `[doc p.N]`. DOCX and Markdown have no real pages; we synthesise pages
by splitting on page breaks (DOCX) or top-level headings (Markdown) so citations
still point at a navigable location.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".txt"}


@dataclass(frozen=True)
class Page:
    doc: str
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc: str
    page: int
    text: str


def load_pdf(path: Path) -> list[Page]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Page(doc=path.name, page=i, text=text))
    return pages


def load_docx(path: Path) -> list[Page]:
    document = DocxDocument(str(path))
    pages: list[Page] = []
    buffer: list[str] = []
    page_no = 1

    def flush() -> None:
        nonlocal page_no
        text = "\n".join(buffer).strip()
        if text:
            pages.append(Page(doc=path.name, page=page_no, text=text))
            page_no += 1
        buffer.clear()

    for paragraph in document.paragraphs:
        has_break = any(
            br.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type") == "page"
            for br in paragraph._p.iter(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}br"
            )
        )
        if paragraph.style is not None and paragraph.style.name.startswith("Heading 1"):
            flush()
        buffer.append(paragraph.text)
        if has_break:
            flush()
    flush()
    return pages


def load_markdown(path: Path) -> list[Page]:
    raw = path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^(?=# )", raw)
    pages = []
    page_no = 1
    for section in sections:
        text = section.strip()
        if not text:
            continue
        pages.append(Page(doc=path.name, page=page_no, text=text))
        page_no += 1
    return pages


def load_document(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".docx":
        return load_docx(path)
    if suffix in {".md", ".txt"}:
        return load_markdown(path)
    raise ValueError(f"Unsupported file type: {path}")


def content_hash(path: Path) -> str:
    """SHA-256 of the file bytes; the key for skipping unchanged documents on re-index."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_documents(docs_dir: Path) -> Iterator[Path]:
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def _split_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split on whitespace into windows of ~chunk_size characters with overlap."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start
        length = 0
        while end < len(words) and length + len(words[end]) + 1 <= chunk_size:
            length += len(words[end]) + 1
            end += 1
        end = max(end, start + 1)
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        # step back by roughly `overlap` characters worth of words
        back = 0
        step = end
        while step > start + 1 and back < overlap:
            step -= 1
            back += len(words[step]) + 1
        start = step
    return chunks


def chunk_pages(pages: list[Page], chunk_size: int = 700, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for i, text in enumerate(_split_words(page.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{page.doc}#p{page.page}#c{i}",
                    doc=page.doc,
                    page=page.page,
                    text=text,
                )
            )
    return chunks


def ingest_directory(docs_dir: Path, chunk_size: int = 700, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in iter_documents(docs_dir):
        chunks.extend(chunk_pages(load_document(path), chunk_size, overlap))
    return chunks
