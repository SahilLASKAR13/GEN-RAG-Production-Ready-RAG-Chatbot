# retrieval/reranker.py
# Cross-encoder reranking to improve precision of retrieved chunks.
# Supports Cohere Rerank (API) and a local cross-encoder fallback.

from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

from config.settings import settings


SearchResult = dict[str, Any]


class CrossEncoderReranker:
    """
    Local CrossEncoder reranker using Hugging Face.
    """

    def __init__(
        self,
        model_name: str = settings.reranker_model,
        top_n: int = settings.rerank_top_n,
    ):
        self.model = CrossEncoder(model_name)
        self.top_n = top_n

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:

        if not results:
            return []

        pairs = [
            (query, result["text"])
            for result in results
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, results),
            key=lambda item: item[0],
            reverse=True,
        )

        reranked: list[SearchResult] = []

        for score, result in ranked[: self.top_n]:

            item = dict(result)

            item["score"] = float(score)

            reranked.append(item)

        return reranked


class NoOpReranker:
    """
    Return the retrieved documents without reranking.
    """

    def __init__(
        self,
        top_n: int = settings.rerank_top_n,
    ):
        self.top_n = top_n

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:

        return results[: self.top_n]


def get_reranker():

    if settings.use_reranker:
        return CrossEncoderReranker()

    return NoOpReranker()
