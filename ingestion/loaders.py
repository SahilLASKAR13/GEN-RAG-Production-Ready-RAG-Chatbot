# ingestion/loaders.py
# Unified interface for loading documents from different sources.
# Each loader returns a list of dicts: {"text": str, "metadata": dict}

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Type

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader


Document = dict[str, Any]


class PDFLoader:
    """Load all pages from a PDF file."""

    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        reader = PdfReader(str(path))
        documents: list[Document] = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if text:
                documents.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": str(path),
                            "page": page_num,
                            "type": "pdf",
                        },
                    }
                )

        return documents
    

class TextLoader:
    """Load a plain text file."""

    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = path.read_text(encoding="utf-8").strip()

        if not text:
            return []

        return [
            {
                "text": text,
                "metadata": {
                    "source": str(path),
                    "type": "text",
                },
            }
        ]


class JSONLoader:
    """Load documents from a JSON file."""

    def __init__(self, text_key: str = "content"):
        self.text_key = text_key

    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        items = data if isinstance(data, list) else [data]

        documents: list[Document] = []

        for item in items:
            text = str(item.get(self.text_key, "")).strip()

            if not text:
                continue

            metadata = {
                k: v
                for k, v in item.items()
                if k != self.text_key
            }

            metadata.update(
                {
                    "source": str(path),
                    "type": "json",
                }
            )

            documents.append(
                {
                    "text": text,
                    "metadata": metadata,
                }
            )

        return documents


class WebLoader:
    """Load text from a webpage."""

    def load(self, url: str) -> list[Document]:
        response = httpx.get(
            url,
            timeout=20,
            follow_redirects=True,
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        text = soup.get_text(separator="\n", strip=True)

        return [
            {
                "text": text,
                "metadata": {
                    "source": url,
                    "type": "web",
                },
            }
        ]


class LoaderFactory:
    """Factory for creating document loaders."""

    _registry: dict[str, Type] = {
        ".pdf": PDFLoader,
        ".txt": TextLoader,
        ".json": JSONLoader,
    }

    @classmethod
    def for_file(cls, path: str | Path):
        path = Path(path)

        loader_class = cls._registry.get(path.suffix.lower())

        if loader_class is None:
            supported = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unsupported file type '{path.suffix}'. "
                f"Supported types: {supported}"
            )

        return loader_class()

    @staticmethod
    def for_url() -> WebLoader:
        return WebLoader()