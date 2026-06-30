from __future__ import annotations

from pathlib import Path

from ingestion.loaders import LoaderFactory
from ingestion.chunker import RecursiveChunker
from indexing.embedder import get_embedder
from indexing.vector_store import get_vector_store
from retrieval.retriever import BM25Index

from utils.logger import get_logger


logger = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Steps
    -----
    1. Load documents
    2. Chunk documents
    3. Generate embeddings
    4. Store vectors in FAISS
    5. Build BM25 index
    6. Persist FAISS index
    """

    def __init__(self):

        self.chunker = RecursiveChunker()

        self.embedder = get_embedder()

        self.vector_store = get_vector_store()

        self.bm25 = BM25Index()

    ####################################################################
    # Internal helper
    ####################################################################

    def _index_documents(
        self,
        documents: list[dict],
    ) -> None:

        logger.info(f"Loaded {len(documents)} documents.")

        chunks = self.chunker.split_many(documents)
        print("\n========== CHUNKS DEBUG ==========")
        print("Chunks:", len(chunks))
        if chunks:
            print("First chunk:")
            print(chunks[0])
        print("==================================\n")

        logger.info(f"Generated {len(chunks)} chunks.")

        texts = [
            chunk["text"]
            for chunk in chunks
        ]
        print("\n========== TEXTS DEBUG ==========")
        print("Texts:", len(texts))
        if texts:
            print(texts[0][:300])
        print("=================================\n")

        metadatas = [
            chunk["metadata"]
            for chunk in chunks
        ]

        logger.info("Generating embeddings...")

        print("Number of chunks:", len(chunks))
        print("Number of texts:", len(texts))

        if texts:
            print("First text:")
            print(texts[0][:300])
        else:
            print("texts is EMPTY")

        vectors = self.embedder.embed(texts)

        print("Texts:", len(texts))
        print("Embeddings:", len(vectors))



    # ---------------- DEBUG ----------------
        import numpy as np

        print("\n========== EMBEDDING DEBUG ==========")
        print("vectors type :", type(vectors))

        try:
            print("vectors length :", len(vectors))
        except Exception:
            print("vectors has no len()")

        if hasattr(vectors, "__len__") and len(vectors) > 0:
            print("first item type :", type(vectors[0]))
            print("first item :", vectors[0])

        vectors_np = np.asarray(vectors)

        print("numpy shape :", vectors_np.shape)
        print("numpy dtype :", vectors_np.dtype)
        print("=====================================\n")
    # ---------------------------------------

        logger.info("Saving vectors to FAISS...")

        self.vector_store.add(
            texts=texts,
            vectors=vectors,
            metadatas=metadatas,
        )

        self.vector_store.save()

        logger.info("Building BM25 index...")

        self.bm25.add(
            texts=texts,
            metadatas=metadatas,
        )

        logger.info("Ingestion completed successfully.")

    ####################################################################
    # Public methods
    ####################################################################

    def ingest_file(
        self,
        path: str,
    ) -> None:

        logger.info(f"Ingesting file: {path}")

        loader = LoaderFactory.for_file(path)

        docs = loader.load(path)

        self._index_documents(docs)

    def ingest_url(
        self,
        url: str,
    ) -> None:

        logger.info(f"Ingesting URL: {url}")

        loader = LoaderFactory.for_url(url)

        docs = loader.load(url)
        print("\n========== DOCS DEBUG ==========")
        print("Documents:", len(docs))
        if docs:
            print("First document:")
            print(docs[0])
        print("================================\n")

        self._index_documents(docs)

    def ingest_directory(
        self,
        directory: str,
    ) -> None:

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(directory)

        supported = {
            ".pdf",
            ".txt",
            ".json",
        }

        files = []

        for file in directory.rglob("*"):

            if file.suffix.lower() in supported:
                files.append(file)

        logger.info(f"Found {len(files)} files.")

        for file in files:

            logger.info(f"Ingesting {file}")

            loader = LoaderFactory.for_file(str(file))

            docs = loader.load(str(file))

            self._index_documents(docs)

        logger.info("Directory ingestion complete.")

    def clear(self):

        logger.warning("Deleting FAISS index...")

        self.vector_store.delete_all()

        self.bm25 = BM25Index()

        logger.info("Index cleared.")

    def ingest(self, path: str):
       """
       Backward-compatible wrapper.
       """
       self.ingest_file(path)