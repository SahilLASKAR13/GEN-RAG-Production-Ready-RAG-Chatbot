# retrieval/reranker.py
# Cross-encoder reranking to improve precision of retrieved chunks.
# Supports Cohere Rerank (API) and a local cross-encoder fallback.

from __future__ import annotations
from typing import Any

from config.settings import settings


SearchResult = dict[str, Any]


class CohereReranker:
    """
    Uses Cohere's rerank endpoint — best quality, requires API key.
    """

    def __init__(self, top_n: int = settings.rerank_top_n, model: str = "rerank-english-v3.0"):
        import cohere
        self.client = cohere.Client(settings.cohere_api_key)
        self.top_n = top_n
        self.model = model

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return []

        docs = [r["text"] for r in results]
        resp = self.client.rerank(
            query=query,
            documents=docs,
            model=self.model,
            top_n=self.top_n,
        )
        reranked = []
        for hit in resp.results:
            item = dict(results[hit.index])
            item["score"] = hit.relevance_score
            reranked.append(item)
        return reranked


class CrossEncoderReranker:
    """
    Local cross-encoder via sentence-transformers — no API key needed.
    Slower than Cohere but works offline.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = settings.rerank_top_n,
    ):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.top_n = top_n

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return []

        pairs = [(query, r["text"]) for r in results]
        scores = self.model.predict(pairs)

        scored = sorted(
            zip(scores, results),
            key=lambda x: x[0],
            reverse=True,
        )
        reranked = []
        for score, item in scored[: self.top_n]:
            item = dict(item)
            item["score"] = float(score)
            reranked.append(item)
        return reranked


class NoOpReranker:
    """Pass-through — returns the first top_n results unchanged."""

    def __init__(self, top_n: int = settings.rerank_top_n):
        self.top_n = top_n

    def rerank(self, _query: str, results: list[SearchResult]) -> list[SearchResult]:
        return results[: self.top_n]


def get_reranker():
    if not settings.use_reranker:
        return NoOpReranker()
    if settings.cohere_api_key:
        return CohereReranker()
    return CrossEncoderReranker()
