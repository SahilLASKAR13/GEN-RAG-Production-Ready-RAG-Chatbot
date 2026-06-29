# indexing/vector_store.py
# Abstract interface + concrete implementations for Chroma, Pinecone, and FAISS.
# All stores expose the same add() / query() surface.

from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from typing import Any

from config.settings import settings


SearchResult = dict[str, Any]   # {"text": str, "metadata": dict, "score": float}


class VectorStore(ABC):
    @abstractmethod
    def add(self, texts: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None: ...

    @abstractmethod
    def query(self, vector: list[float], top_k: int = settings.top_k) -> list[SearchResult]: ...

    @abstractmethod
    def delete_all(self) -> None: ...


# ── Chroma (local, zero-config) ────────────────────────────────────────────────

class ChromaStore(VectorStore):
    def __init__(self, collection_name: str = "rag", persist_dir: str = settings.chroma_persist_dir):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, texts: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        ids = [str(uuid.uuid4()) for _ in texts]
        self.collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, vector: list[float], top_k: int = settings.top_k) -> list[SearchResult]:
        res = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            results.append({"text": text, "metadata": meta, "score": 1 - dist})
        return results

    def delete_all(self) -> None:
        self.collection.delete(where={})


# ── Pinecone (managed, production-grade) ────────────────────────────────────────

class PineconeStore(VectorStore):
    def __init__(self):
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = pc.Index(settings.pinecone_index)

    def add(self, texts: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        upserts = []
        for text, vec, meta in zip(texts, vectors, metadatas):
            upserts.append({
                "id": str(uuid.uuid4()),
                "values": vec,
                "metadata": {**meta, "_text": text},
            })
        self.index.upsert(vectors=upserts, batch_size=100)

    def query(self, vector: list[float], top_k: int = settings.top_k) -> list[SearchResult]:
        resp = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        return [
            {
                "text": m.metadata.pop("_text", ""),
                "metadata": m.metadata,
                "score": m.score,
            }
            for m in resp.matches
        ]

    def delete_all(self) -> None:
        self.index.delete(delete_all=True)


# ── FAISS (in-memory, no external deps) ─────────────────────────────────────────

class FAISSStore(VectorStore):
    def __init__(self, dim: int = settings.embedding_dim):
        import faiss, numpy as np
        self.index = faiss.IndexFlatIP(dim)   # inner-product ≈ cosine on unit vecs
        self._texts: list[str] = []
        self._metas: list[dict] = []

    def add(self, texts: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        import numpy as np
        arr = np.array(vectors, dtype="float32")
        self.index.add(arr)
        self._texts.extend(texts)
        self._metas.extend(metadatas)

    def query(self, vector: list[float], top_k: int = settings.top_k) -> list[SearchResult]:
        import numpy as np
        arr = np.array([vector], dtype="float32")
        scores, idxs = self.index.search(arr, top_k)
        return [
            {"text": self._texts[i], "metadata": self._metas[i], "score": float(scores[0][rank])}
            for rank, i in enumerate(idxs[0])
            if i != -1
        ]

    def delete_all(self) -> None:
        self.index.reset()
        self._texts.clear()
        self._metas.clear()


def get_vector_store() -> VectorStore:
    provider = settings.vector_store_provider
    if provider == "chroma":
        return ChromaStore()
    if provider == "pinecone":
        return PineconeStore()
    if provider == "faiss":
        return FAISSStore()
    raise ValueError(f"Unknown vector store: {provider}")
