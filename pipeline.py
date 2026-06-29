# pipeline.py
# End-to-end RAG pipeline. Wires all modules together.
# Use RAGPipeline for the full chatbot loop; use index_documents() to ingest.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator

from config.settings import settings
from ingestion.loaders import LoaderFactory
from ingestion.chunker import RecursiveChunker
from indexing.embedder import get_embedder
from indexing.vector_store import get_vector_store
from retrieval.query_rewriter import get_query_rewriter
from retrieval.retriever import HybridRetriever, BM25Index
from retrieval.reranker import get_reranker
from generation.prompt_builder import PromptBuilder
from generation.llm import get_llm, LLMResponse
from generation.guardrails import GuardrailPipeline
from utils.logger import get_logger


log = get_logger(__name__)


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    rewritten_queries: list[str] = field(default_factory=list)


class RAGPipeline:
    """
    Orchestrates: query rewriting → retrieval → reranking → generation → guardrails.
    All dependencies are injected so each module can be swapped independently.
    """

    def __init__(
        self,
        rewriter_strategy: str = "passthrough",   # passthrough | hyde | step_back | multi_query
    ):
        log.info("Initialising RAG pipeline")
        self.embedder      = get_embedder()
        self.vector_store  = get_vector_store()
        self.bm25          = BM25Index()
        self.retriever     = HybridRetriever(self.embedder, self.vector_store, self.bm25)
        self.rewriter      = get_query_rewriter(rewriter_strategy)
        self.reranker      = get_reranker()
        self.prompt_builder = PromptBuilder()
        self.llm           = get_llm()
        self.guardrails    = GuardrailPipeline()
        log.info("Pipeline ready")

    # ── Ingestion ────────────────────────────────────────────────────────

    def index_documents(self, paths_or_urls: list[str]) -> int:
        """Load, chunk, embed, and store documents. Returns chunk count."""
        chunker = RecursiveChunker()
        all_chunks = []

        for source in paths_or_urls:
            log.info({"action": "loading", "source": source})
            loader = (
                LoaderFactory.for_url(source)
                if source.startswith("http")
                else LoaderFactory.for_file(source)
            )
            docs = loader.load(source)
            chunks = chunker.split_many(docs)
            all_chunks.extend(chunks)
            log.info({"action": "chunked", "source": source, "chunks": len(chunks)})

        if not all_chunks:
            return 0

        texts = [c["text"] for c in all_chunks]
        metas = [c["metadata"] for c in all_chunks]
        vectors = self.embedder.embed(texts)

        self.vector_store.add(texts, vectors, metas)
        self.bm25.add(texts, metas)

        log.info({"action": "indexed", "total_chunks": len(all_chunks)})
        return len(all_chunks)

    # ── Query ────────────────────────────────────────────────────────────

    def query(
        self,
        user_query: str,
        chat_history: list[dict] | None = None,
    ) -> RAGResponse:
        """Full RAG turn: rewrite → retrieve → rerank → generate → guard."""

        # 1. Query rewriting
        queries = self.rewriter.rewrite(user_query)
        log.info({"action": "rewritten", "queries": queries})

        # 2. Hybrid retrieval
        retrieved = self.retriever.retrieve(queries)
        log.info({"action": "retrieved", "count": len(retrieved)})

        # 3. Reranking
        reranked = self.reranker.rerank(user_query, retrieved)
        log.info({"action": "reranked", "count": len(reranked)})

        # 4. Prompt assembly
        system, messages = self.prompt_builder.build(user_query, reranked, chat_history)

        # 5. LLM call
        llm_resp: LLMResponse = self.llm.complete(system, messages)
        log.info({"action": "generated", "tokens_in": llm_resp.input_tokens, "tokens_out": llm_resp.output_tokens})

        # 6. Guardrails
        context_texts = [r["text"] for r in reranked]
        guard = self.guardrails.run(llm_resp.text, context_texts)
        final_answer = llm_resp.text if guard.passed else guard.safe_response
        if not guard.passed:
            log.warning({"action": "guardrail_blocked", "reason": guard.reason})

        sources = [
            {"text": r["text"][:200], "metadata": r["metadata"], "score": r["score"]}
            for r in reranked
        ]

        return RAGResponse(
            answer=final_answer,
            sources=sources,
            model=llm_resp.model,
            input_tokens=llm_resp.input_tokens,
            output_tokens=llm_resp.output_tokens,
            rewritten_queries=queries,
        )

    def stream_query(
        self,
        user_query: str,
        chat_history: list[dict] | None = None,
    ) -> Iterator[str]:
        """Streaming variant — yields text tokens as they arrive from the LLM."""
        queries = self.rewriter.rewrite(user_query)
        retrieved = self.retriever.retrieve(queries)
        reranked = self.reranker.rerank(user_query, retrieved)
        system, messages = self.prompt_builder.build(user_query, reranked, chat_history)
        yield from self.llm.stream(system, messages)


# ── Convenience entry-point ──────────────────────────────────────────────────

def index_documents(paths_or_urls: list[str]) -> int:
    pipeline = RAGPipeline()
    return pipeline.index_documents(paths_or_urls)


def chat(query: str, history: list[dict] | None = None) -> RAGResponse:
    pipeline = RAGPipeline()
    return pipeline.query(query, history)
