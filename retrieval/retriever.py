# retrieval/retriever.py
# Combines dense vector search with BM25 sparse search (hybrid retrieval).
# Results are merged with Reciprocal Rank Fusion (RRF).

from __future__ import annotations
from typing import Any

from config.settings import settings
from indexing.embedder import EmbedderProtocol
from indexing.vector_store import VectorStore


SearchResult = dict[str, Any]


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a result at 0-based rank."""
    return 1.0 / (k + rank + 1)


class BM25Index:
    """
    In-memory BM25 index built from the same chunks stored in the vector DB.
    Uses rank_bm25 library. Falls back to empty results if not installed.
    """

    def __init__(self):
        self._corpus: list[str] = []
        self._metadatas: list[dict] = []
        self._bm25 = None

    def add(self, texts: list[str], metadatas: list[dict]) -> None:
        self._corpus.extend(texts)
        self._metadatas.extend(metadatas)
        self._build()

    def _build(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
            tokenised = [t.lower().split() for t in self._corpus]
            self._bm25 = BM25Okapi(tokenised)
        except ImportError:
            self._bm25 = None

    def query(self, text: str, top_k: int = settings.top_k) -> list[SearchResult]:
        if not self._bm25 or not self._corpus:
            return []
        scores = self._bm25.get_scores(text.lower().split())
        top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "text": self._corpus[i],
                "metadata": self._metadatas[i],
                "score": float(scores[i]),
            }
            for i in top_idxs
        ]


class HybridRetriever:
    """
    Runs dense ANN search + BM25 keyword search, then fuses with RRF.
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        top_k: int = settings.top_k,
        use_hybrid: bool = settings.use_hybrid,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25_index
        self.top_k = top_k
        self.use_hybrid = use_hybrid

    def retrieve(self, queries: list[str]) -> list[SearchResult]:
        """
        Accept multiple query phrasings (from query rewriter).
        Deduplicate and re-rank across all phrasings.
        """
        dense_results: list[SearchResult] = []
        sparse_results: list[SearchResult] = []

        for query in queries:
            vec = self.embedder.embed_query(query)
            dense_results.extend(self.vector_store.query(vec, top_k=self.top_k))
            if self.use_hybrid:
                sparse_results.extend(self.bm25.query(query, top_k=self.top_k))

        return self._fuse(dense_results, sparse_results)

    def _fuse(
        self,
        dense: list[SearchResult],
        sparse: list[SearchResult],
    ) -> list[SearchResult]:
        scores: dict[str, float] = {}
        by_key: dict[str, SearchResult] = {}

        def _key(r: SearchResult) -> str:
            return r["text"][:200]   # stable dedup key

        for rank, r in enumerate(dense):
            k = _key(r)
            scores[k] = scores.get(k, 0.0) + _rrf_score(rank)
            by_key[k] = r

        for rank, r in enumerate(sparse):
            k = _key(r)
            scores[k] = scores.get(k, 0.0) + _rrf_score(rank)
            by_key.setdefault(k, r)

        ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        results = []
        for k in ranked[: self.top_k]:
            item = dict(by_key[k])
            item["score"] = scores[k]
            results.append(item)
        return results
