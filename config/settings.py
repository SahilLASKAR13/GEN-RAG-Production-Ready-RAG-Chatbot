# config/settings.py
# Single source of truth for all configuration.
# Values are read from environment variables with sensible defaults.

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # ── LLM ────────────────────────────────────────────────────────────
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))

    llm_model: str = field(
        default_factory=lambda: os.getenv(
            "LLM_MODEL",
            "llama-3.3-70b-versatile"
        )
    )

    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2"))
    )

    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "1024"))
    )
    # ── Embeddings ──────────────────────────────────────────────────────
    embedding_model: str = field(
    default_factory=lambda: os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5"
    )
    )

    embedding_dim: int = 384

    # ── Vector store ────────────────────────────────────────────────────
    vector_store_provider: str = field(
        default_factory=lambda: os.getenv("VECTOR_STORE_PROVIDER", "faiss")
)

    faiss_index_path: str = field(
        default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "./storage/faiss_index")
)

    # ── Retrieval ────────────────────────────────────────────────────────
    top_k: int = 5
    rerank_top_n: int = 3

    use_hybrid: bool = True
    use_reranker: bool = True

    reranker_model: str = field(
    default_factory=lambda: os.getenv(
        "RERANKER_MODEL",
        "BAAI/bge-reranker-base"
    )
)
    
    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512"))
)

    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "64"))
)


# Module-level singleton — import this everywhere
settings = Settings()
