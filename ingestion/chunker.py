# ingestion/chunker.py
# Splits raw documents into overlapping chunks suitable for embedding.

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings


Document = dict[str, Any]


class RecursiveChunker:
    """
    Split documents into overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter.
    """

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def split(self, document: Document) -> list[Document]:
        chunks = self.splitter.split_text(document["text"])

        total_chunks = len(chunks)

        return [
            {
                "text": chunk,
                "metadata": {
                    **document["metadata"],
                    "chunk_index": index,
                    "chunk_total": total_chunks,
                },
            }
            for index, chunk in enumerate(chunks)
        ]

    def split_many(self, documents: list[Document]) -> list[Document]:
        chunked_documents: list[Document] = []

        for document in documents:
            chunked_documents.extend(self.split(document))

        return chunked_documents