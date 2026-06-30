from __future__ import annotations

from generation.guardrails import GuardrailPipeline
from generation.llm import get_llm
from generation.prompt_builder import PromptBuilder

from indexing.embedder import get_embedder
from retrieval.query_rewriter import get_query_rewriter
from indexing.vector_store import get_vector_store

from retrieval.retriever import BM25Index, HybridRetriever
from retrieval.reranker import get_reranker

from utils.logger import get_logger


logger = get_logger(__name__)


class RAGPipeline:
    """
    End-to-End Retrieval Augmented Generation Pipeline.

    Steps
    -----
    1. Rewrite query
    2. Retrieve documents
    3. Rerank documents
    4. Build prompt
    5. Generate answer
    6. Run guardrails
    """

    def __init__(self):

        logger.info("Initializing RAG Pipeline...")

        self.embedder = get_embedder()

        self.vector_store = get_vector_store()

        self.bm25 = BM25Index()

        self.retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25_index=self.bm25,
        )

        self.rewriter = get_query_rewriter()

        self.reranker = get_reranker()

        self.prompt_builder = PromptBuilder()

        self.llm = get_llm()

        self.guardrails = GuardrailPipeline()

        logger.info("RAG Pipeline initialized.")

    ###########################################################
    # Main Pipeline
    ###########################################################

    def run(
        self,
        query: str,
        chat_history: list[dict] | None = None,
    ) -> dict:

        logger.info(f"User Query: {query}")

        #######################################################
        # Query Rewrite
        #######################################################

        rewritten_queries = self.rewriter.rewrite(query)

        logger.info(
            f"Generated {len(rewritten_queries)} rewritten queries."
        )

        #######################################################
        # Retrieval
        #######################################################

        retrieved = self.retriever.retrieve(
            rewritten_queries
        )
        print("\n========== RETRIEVED DOCUMENTS ==========")
        print("Retrieved:", len(retrieved))

        for i, doc in enumerate(retrieved):
            print(f"\nDocument {i+1}")
            print(doc["metadata"])
            print(doc["text"][:300])
        print("=========================================\n")

        logger.info(
            f"Retrieved {len(retrieved)} documents."
        )

        #######################################################
        # Reranking
        #######################################################

        retrieved = self.reranker.rerank(
            query,
            retrieved,
        )

        logger.info(
            f"Top documents after reranking: {len(retrieved)}"
        )

        #######################################################
        # Prompt
        #######################################################

        system_prompt, messages = self.prompt_builder.build(
            query=query,
            retrieved=retrieved,
            chat_history=chat_history,
        )

        #######################################################
        # LLM
        #######################################################

        response = self.llm.complete(
            system_prompt,
            messages,
        )

        #######################################################
        # Guardrails
        #######################################################

        guard = self.guardrails.run(
            answer=response.text,
            context_chunks=[
                item["text"]
                for item in retrieved
            ],
        )

        if not guard.passed:

            logger.warning(
                f"Guardrail blocked answer: {guard.reason}"
            )

            return {
                "answer": guard.safe_response,
                "sources": [],
                "model": response.model,
            }

        #######################################################
        # Sources
        #######################################################

        sources = []

        for doc in retrieved:

            metadata = doc["metadata"]

            sources.append(
                {
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "score": round(doc["score"], 4),
                }
            )

        #######################################################
        # Final Output
        #######################################################

        return {
            "answer": response.text,
            "sources": sources,
            "model": response.model,
        }

    ###########################################################
    # Streaming
    ###########################################################

    def stream(
        self,
        query: str,
        chat_history: list[dict] | None = None,
    ):

        rewritten_queries = self.rewriter.rewrite(query)

        retrieved = self.retriever.retrieve(
            rewritten_queries
        )

        retrieved = self.reranker.rerank(
            query,
            retrieved,
        )

        system_prompt, messages = self.prompt_builder.build(
            query=query,
            retrieved=retrieved,
            chat_history=chat_history,
        )

        yield from self.llm.stream(
            system_prompt,
            messages,
        )