# retrieval/retriever.py
# Combines dense vector search with BM25 sparse search (hybrid retrieval).
# Results are merged with Reciprocal Rank Fusion (RRF).

from __future__ import annotations

from typing import Any

from rank_bm25 import BM25Okapi

from config.settings import settings
from indexing.embedder import EmbedderProtocol
from indexing.vector_store import VectorStore


SearchResult = dict[str, Any]


def reciprocal_rank_fusion(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank + 1)


class BM25Index:
    """
    BM25 keyword index.
    """

    def __init__(self):
        self.corpus: list[str] = []
        self.metadatas: list[dict] = []
        self.bm25: BM25Okapi | None = None

    def add(
        self,
        texts: list[str],
        metadatas: list[dict],
    ) -> None:

        self.corpus.extend(texts)
        self.metadatas.extend(metadatas)

        tokenized = [
            text.lower().split()
            for text in self.corpus
        ]

        self.bm25 = BM25Okapi(tokenized)

    def query(
        self,
        query: str,
        top_k: int = settings.top_k,
    ) -> list[SearchResult]:

        if self.bm25 is None:
            return []

        scores = self.bm25.get_scores(
            query.lower().split()
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            {
                "text": self.corpus[i],
                "metadata": self.metadatas[i],
                "score": float(scores[i]),
            }
            for i in ranked_indices
        ]


class HybridRetriever:
    """
    Dense + BM25 + Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        vector_store: VectorStore,
        bm25_index: BM25Index,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25_index

        self.top_k = settings.top_k
        self.use_hybrid = settings.use_hybrid

    # ---------------------------------------------------------
    # NEW
    # ---------------------------------------------------------

    def refresh_bm25(self):
        """
        Build BM25 from everything currently stored in FAISS.
        This keeps BM25 synchronized after every upload.
        """

        if len(self.bm25.corpus) == len(self.vector_store.texts):
            return

        self.bm25 = BM25Index()

        if self.vector_store.texts:
            self.bm25.add(
                self.vector_store.texts,
                self.vector_store.metadatas,
            )

    # ---------------------------------------------------------

    def retrieve(
        self,
        queries: list[str],
    ) -> list[SearchResult]:

        # Always synchronize BM25 with FAISS
        self.refresh_bm25()

        dense_results: list[SearchResult] = []
        sparse_results: list[SearchResult] = []

        for query in queries:

            embedding = self.embedder.embed_query(query)

            dense_results.extend(
                self.vector_store.query(
                    embedding,
                    self.top_k,
                )
            )

            if self.use_hybrid:

                sparse_results.extend(
                    self.bm25.query(
                        query,
                        self.top_k,
                    )
                )

        return self._fuse(
            dense_results,
            sparse_results,
        )

    def _fuse(
        self,
        dense: list[SearchResult],
        sparse: list[SearchResult],
    ) -> list[SearchResult]:

        fused_scores: dict[str, float] = {}
        documents: dict[str, SearchResult] = {}

        def key(result: SearchResult) -> str:
            return result["text"][:200]

        for rank, result in enumerate(dense):

            k = key(result)

            fused_scores[k] = (
                fused_scores.get(k, 0)
                + reciprocal_rank_fusion(rank)
            )

            documents[k] = result

        for rank, result in enumerate(sparse):

            k = key(result)

            fused_scores[k] = (
                fused_scores.get(k, 0)
                + reciprocal_rank_fusion(rank)
            )

            documents.setdefault(k, result)

        ranked = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        final_results: list[SearchResult] = []

        for key, score in ranked[: self.top_k]:

            item = dict(documents[key])

            item["score"] = score

            final_results.append(item)

        return final_results
