# indexing/vector_store.py
# Abstract interface + concrete implementations for Chroma, Pinecone, and FAISS.
# All stores expose the same add() / query() surface.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import pickle

import faiss
import numpy as np

from config.settings import settings


SearchResult = dict[str, Any]


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        ...

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int = settings.top_k,
    ) -> list[SearchResult]:
        ...

    @abstractmethod
    def save(self) -> None:
        ...

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def delete_all(self) -> None:
        ...


class FAISSStore(VectorStore):
    """Local FAISS vector store with disk persistence."""

    def __init__(
        self,
        dimension: int = settings.embedding_dim,
        index_path: str = settings.faiss_index_path,
    ):
        self.dimension = dimension
        self.index_path = Path(index_path)

        self.index_file = self.index_path / "index.faiss"
        self.metadata_file = self.index_path / "metadata.pkl"

        self.index_path.mkdir(parents=True, exist_ok=True)

        self.index = faiss.IndexFlatIP(dimension)

        self.texts: list[str] = []
        self.metadatas: list[dict] = []

        self.load()

    def add(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
    ) -> None:

        vectors_np = np.asarray(vectors, dtype=np.float32)

        self.index.add(vectors_np)

        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def query(
        self,
        vector: list[float],
        top_k: int = settings.top_k,
    ) -> list[SearchResult]:

        query_vector = np.asarray([vector], dtype=np.float32)

        scores, indices = self.index.search(query_vector, top_k)

        results: list[SearchResult] = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            results.append(
                {
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(score),
                }
            )

        return results

    def save(self) -> None:
        """Persist FAISS index and metadata."""

        faiss.write_index(self.index, str(self.index_file))

        with open(self.metadata_file, "wb") as f:
            pickle.dump(
                {
                    "texts": self.texts,
                    "metadatas": self.metadatas,
                },
                f,
            )

    def load(self) -> None:
        """Load saved FAISS index if it exists."""

        if self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))

        if self.metadata_file.exists():
            with open(self.metadata_file, "rb") as f:
                data = pickle.load(f)

            self.texts = data["texts"]
            self.metadatas = data["metadatas"]

    def delete_all(self) -> None:
        """Clear all vectors and remove persisted files."""

        self.index = faiss.IndexFlatIP(self.dimension)

        self.texts.clear()
        self.metadatas.clear()

        if self.index_file.exists():
            self.index_file.unlink()

        if self.metadata_file.exists():
            self.metadata_file.unlink()


def get_vector_store() -> VectorStore:
    """Return the configured vector store."""

    return FAISSStore()