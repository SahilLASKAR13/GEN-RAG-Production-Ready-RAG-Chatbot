# indexing/embedder.py
# Thin wrapper around embedding providers.
# Returns normalised float32 vectors.

from __future__ import annotations
import os
from typing import Protocol

from config.settings import settings


class EmbedderProtocol(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    """Uses OpenAI text-embedding-* models."""

    def __init__(self, model: str = settings.embedding_model):
        from openai import OpenAI
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Batch in groups of 100 to stay within rate limits
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            all_vectors.extend([item.embedding for item in resp.data])
        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HuggingFaceEmbedder:
    """Local sentence-transformers model — no API key needed."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def get_embedder() -> EmbedderProtocol:
    """Factory — swap provider via EMBEDDING_PROVIDER env var."""
    provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    if provider == "openai":
        return OpenAIEmbedder()
    if provider == "huggingface":
        return HuggingFaceEmbedder()
    raise ValueError(f"Unknown embedding provider: {provider}")
