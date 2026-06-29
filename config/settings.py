# config/settings.py
# Single source of truth for all configuration.
# Values are read from environment variables with sensible defaults.

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # ==========================================================
    # LLM (Groq)
    # ==========================================================

    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )

    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "groq")
    )

    llm_model: str = field(
        default_factory=lambda: os.getenv(
            "LLM_MODEL",
            "llama-3.3-70b-versatile",
        )
    )

    llm_temperature: float = field(
        default_factory=lambda: float(
            os.getenv("LLM_TEMPERATURE", "0.2")
        )
    )

    llm_max_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("LLM_MAX_TOKENS", "1024")
        )
    )

    # ==========================================================
    # Embeddings
    # ==========================================================

    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL",
            "BAAI/bge-small-en-v1.5",
        )
    )

    embedding_dim: int = field(
        default_factory=lambda: int(
            os.getenv("EMBEDDING_DIM", "384")
        )
    )

    # ==========================================================
    # Vector Store
    # ==========================================================

    vector_store_provider: str = field(
        default_factory=lambda: os.getenv(
            "VECTOR_STORE_PROVIDER",
            "faiss",
        )
    )

    faiss_index_path: str = field(
        default_factory=lambda: os.getenv(
            "FAISS_INDEX_PATH",
            "./storage/faiss_index",
        )
    )

    # ==========================================================
    # Retrieval
    # ==========================================================

    top_k: int = field(
        default_factory=lambda: int(
            os.getenv("TOP_K", "5")
        )
    )

    use_hybrid: bool = field(
        default_factory=lambda: os.getenv(
            "USE_HYBRID",
            "True",
        ).lower() == "true"
    )

    # ==========================================================
    # Reranker
    # ==========================================================

    use_reranker: bool = field(
        default_factory=lambda: os.getenv(
            "USE_RERANKER",
            "True",
        ).lower() == "true"
    )

    rerank_top_n: int = field(
        default_factory=lambda: int(
            os.getenv("RERANK_TOP_N", "3")
        )
    )

    reranker_model: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL",
            "BAAI/bge-reranker-base",
        )
    )

    # ==========================================================
    # Chunking
    # ==========================================================

    chunk_size: int = field(
        default_factory=lambda: int(
            os.getenv("CHUNK_SIZE", "512")
        )
    )

    chunk_overlap: int = field(
        default_factory=lambda: int(
            os.getenv("CHUNK_OVERLAP", "64")
        )
    )

    # ==========================================================
    # Prompt
    # ==========================================================

    max_context_chars: int = field(
        default_factory=lambda: int(
            os.getenv("MAX_CONTEXT_CHARS", "6000")
        )
    )

    # ==========================================================
    # Guardrails
    # ==========================================================
    enable_guardrails: bool = field(
        default_factory=lambda: os.getenv(
            "ENABLE_GUARDRAILS",
            "True",
        ).lower() == "true"
    )

    min_answer_length_for_citation: int = field(
        default_factory=lambda: int(
            os.getenv(
                "MIN_ANSWER_LENGTH_FOR_CITATION",
                "200",
            )
        )
    )


    # ==========================================================
    # Logging
    # ==========================================================
    log_level: str = field(
    default_factory=lambda: os.getenv(
        "LOG_LEVEL",
        "INFO",
    )
)

    log_json: bool = field(
    default_factory=lambda: os.getenv(
        "LOG_JSON",
        "True",
    ).lower() == "true"
)


    





# Singleton
settings = Settings()
