from __future__ import annotations

import pytest

from rag.config import Settings
from rag.embeddings import HashEmbedder, get_embedder
from rag.retrieval import HybridRetriever, term_coverage
from rag.store import VectorStore


def test_hash_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashEmbedder()
    a = embedder.embed(["refund policy"])
    b = embedder.embed(["refund policy"])
    assert (a == b).all()
    assert abs(float((a[0] ** 2).sum()) - 1.0) < 1e-5


def test_unknown_provider_rejected() -> None:
    with pytest.raises(ValueError):
        get_embedder("nope")


@pytest.fixture
def retriever(settings: Settings) -> HybridRetriever:
    return HybridRetriever(VectorStore(settings.db_path), get_embedder("hash"))


@pytest.mark.parametrize(
    ("question", "doc", "page"),
    [
        ("How many PTO days per year?", "employee_handbook.md", 2),
        ("If a webhook delivery fails, how many retries?", "product_faq.docx", 3),
        ("how long does a refund take to be processed", "refund_policy.pdf", 3),
    ],
)
def test_hybrid_search_finds_expected_page(
    retriever: HybridRetriever, question: str, doc: str, page: int
) -> None:
    hits = retriever.search(question, k=3)
    assert (doc, page) in {(h.chunk.doc, h.chunk.page) for h in hits}


def test_paraphrases_agree(retriever: HybridRetriever) -> None:
    for question in ("What is the Team plan rate limit?", "requests per minute allowed on Team"):
        pages = {(h.chunk.doc, h.chunk.page) for h in retriever.search(question, k=3)}
        assert ("product_faq.docx", 3) in pages


def test_confidence_low_for_off_topic(retriever: HybridRetriever) -> None:
    hits = retriever.search("What is the capital of Australia?", k=5)
    assert retriever.confidence("What is the capital of Australia?", hits) < 0.30
    hits = retriever.search("How many days of PTO do employees get?", k=5)
    assert retriever.confidence("How many days of PTO do employees get?", hits) > 0.4


def test_term_coverage_prefix_matching() -> None:
    assert term_coverage("retry webhook", "webhooks are retried") == 1.0
    assert term_coverage("capital city", "refund policy") == 0.0
