# indexing/embedder.py
# Thin wrapper around embedding providers.
# Returns normalised float32 vectors.

from __future__ import annotations

from typing import Protocol

from sentence_transformers import SentenceTransformer

from config.settings import settings


class EmbedderProtocol(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class HuggingFaceEmbedder:
    """
    Local Hugging Face embedding model.
    """

    def __init__(self, model_name: str = settings.embedding_model):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def get_embedder() -> EmbedderProtocol:
    """
    Return the configured embedding model.
    """
    return HuggingFaceEmbedder()
