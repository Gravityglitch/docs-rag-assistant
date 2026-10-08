"""Hybrid retrieval: BM25 (lexical) + cosine (semantic), fused with reciprocal rank fusion.

Why both: pure vector search is brittle to paraphrases on small corpora and pure BM25
misses synonyms. RRF makes the two rankings comparable without score calibration.
The returned `confidence` is a separate, interpretable signal used for abstention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from rag.embeddings import EmbeddingProvider, tokenize
from rag.ingest import Chunk
from rag.store import VectorStore

_STOPWORDS = frozenset(
    "a an the is are was were be been of to in on for and or if do does did how what "
    "when where which who why can could should would i you we they it this that my our "
    "your their with from at by as not no there any some about into than much many long "
    "get take will ask after happens before each per".split()
)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    vector_score: float
    bm25_score: float


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: EmbeddingProvider) -> None:
        self.store = store
        self.embedder = embedder
        self.chunks, self.matrix = store.load_matrix()
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks] or [[""]])
        self.idf: dict[str, float] = dict(self.bm25.idf)

    def confidence(self, query: str, hits: list[Hit]) -> float:
        return confidence(query, hits, self.idf)

    def search(self, query: str, k: int = 5, candidates: int = 20) -> list[Hit]:
        if not self.chunks:
            return []
        n = len(self.chunks)
        query_vec = self.embedder.embed([query])[0]
        vector_scores = self.matrix @ query_vec
        bm25_scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

        vector_rank = np.argsort(-vector_scores)[:candidates]
        bm25_rank = np.argsort(-bm25_scores)[:candidates]
        fused = np.zeros(n, dtype=np.float32)
        for rank, idx in enumerate(vector_rank):
            fused[idx] += 1.0 / (60 + rank)
        for rank, idx in enumerate(bm25_rank):
            fused[idx] += 1.0 / (60 + rank)

        top = np.argsort(-fused)[:k]
        return [
            Hit(
                chunk=self.chunks[i],
                score=float(fused[i]),
                vector_score=float(vector_scores[i]),
                bm25_score=float(bm25_scores[i]),
            )
            for i in top
            if fused[i] > 0
        ]


def term_coverage(query: str, text: str, idf: dict[str, float] | None = None) -> float:
    """IDF-weighted fraction of content-bearing query terms present in `text`.

    Terms are prefix-matched (4 chars) so "retry" matches "retried". Terms that are
    common across the corpus (e.g. the company name) carry little weight; terms that
    never appear in the corpus carry the most, which is what pushes off-topic questions
    below the abstention threshold.
    """
    terms = [t for t in tokenize(query) if t not in _STOPWORDS and len(t) > 2]
    if not terms:
        return 0.0
    idf = idf or {}
    # A term that never occurs anywhere in the corpus is the strongest signal that the
    # question is out of scope, so it weighs 1.5x the rarest in-corpus term.
    unknown_weight = 1.5 * max(idf.values(), default=1.0)
    words = set(tokenize(text))
    stems = {w[:4] for w in words}
    total = found = 0.0
    for term in terms:
        weight = idf.get(term, unknown_weight) if idf else 1.0
        weight = max(weight, 0.1)
        total += weight
        if term in words or term[:4] in stems:
            found += weight
    return found / total


def confidence(query: str, hits: list[Hit], idf: dict[str, float] | None = None) -> float:
    """Blend of lexical coverage and semantic similarity of the best hit, in [0, 1].

    Deliberately simple and explainable: a client can look at a low-confidence
    answer and see whether the query terms were actually present in the sources.
    """
    if not hits:
        return 0.0
    best = max(hits, key=lambda h: term_coverage(query, h.chunk.text, idf))
    coverage = term_coverage(query, best.chunk.text, idf)
    semantic = max(0.0, min(1.0, best.vector_score))
    return round(0.6 * coverage + 0.4 * semantic, 3)
