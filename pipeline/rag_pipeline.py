from __future__ import annotations

from generation.guardrails import GuardrailPipeline
from generation.llm import get_llm
from generation.prompt_builder import PromptBuilder

from indexing.embedder import get_embedder
from indexing.vector_store import get_vector_store

from retrieval.query_rewriter import get_query_rewriter
from retrieval.retriever import BM25Index, HybridRetriever
from retrieval.reranker import get_reranker

from utils.logger import get_logger


logger = get_logger(__name__)


class RAGPipeline:
    """
    End-to-End Retrieval Augmented Generation Pipeline.

    Steps
    -----
    1. Query Rewrite
    2. Hybrid Retrieval
    3. Reranking
    4. Prompt Building
    5. LLM Generation
    6. Guardrails
    """

    def __init__(self):

        logger.info("Initializing RAG Pipeline...")

        # Embedding model
        print("1. Loading Embedder...")
        self.embedder = get_embedder()
        print("Done")

        # Vector Store
        print("2. Loading Vector Store...")
        self.vector_store = get_vector_store()
        print("Done")

        # BM25
        print("3. Creating BM25 Index...")
        self.bm25 = BM25Index()
        print("Done")

        # Build BM25 from existing FAISS documents
        if len(self.vector_store.texts) > 0:
            logger.info(
                "Building BM25 index from existing FAISS documents..."
            )

            self.bm25.add(
                texts=self.vector_store.texts,
                metadatas=self.vector_store.metadatas,
            )
        print("Done")

        # Hybrid Retriever
        print("5. Creating Hybrid Retriever...")
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25_index=self.bm25,
        )
        print("Done")

        # Query Rewriter
        print("6. Creating Query Rewriter...")
        self.rewriter = get_query_rewriter()
        print("Done")

        # Cross Encoder Reranker
        print("7. Creating Cross Encoder Reranker...")
        self.reranker = get_reranker()
        print("Done")

        # Prompt Builder
        print("8. Creating Prompt Builder...")
        self.prompt_builder = PromptBuilder()
        print("Done")

        # LLM
        print("9. Loading LLM...")
        self.llm = get_llm()
        print("Done")

        # Guardrails
        print("10. Creating Guardrails...")
        self.guardrails = GuardrailPipeline()
        print("Done")

        logger.info("RAG Pipeline initialized successfully.")

    ###########################################################
    # Refresh
    ###########################################################

    def refresh(self):
        """
        Reload FAISS and rebuild BM25 after a new upload.
        """

        logger.info("Refreshing RAG Pipeline...")

        # Reload latest FAISS index
        self.vector_store.load()

        # Rebuild BM25
        self.bm25 = BM25Index()

        if len(self.vector_store.texts) > 0:

            self.bm25.add(
                texts=self.vector_store.texts,
                metadatas=self.vector_store.metadatas,
            )

        # Update retriever
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25_index=self.bm25,
        )

        logger.info(
            f"Refresh complete. Total vectors: {self.vector_store.index.ntotal}"
        )

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
            print(f"\nDocument {i + 1}")
            print(doc["metadata"])
            print(doc["text"][:300])

        print("=========================================\n")

        logger.info(
            f"Retrieved {len(retrieved)} documents."
        )

        #######################################################
        # No documents found
        #######################################################

        if len(retrieved) == 0:

            return {
                "answer": "I couldn't find any relevant information in the uploaded documents.",
                "sources": [],
                "model": "",
            }

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
        # Prompt Builder
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
                doc["text"]
                for doc in retrieved
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

        seen = set()

        for doc in retrieved:

            metadata = doc["metadata"]

            source = {
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "score": round(doc["score"], 4),
            }

            key = (
                source["source"],
                source["page"],
            )

            if key not in seen:
                seen.add(key)
                sources.append(source)

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
        """
        Streaming version of the RAG pipeline.
        """

        logger.info(f"Streaming Query: {query}")

        #######################################################
        # Query Rewrite
        #######################################################

        rewritten_queries = self.rewriter.rewrite(query)

        #######################################################
        # Retrieval
        #######################################################

        retrieved = self.retriever.retrieve(
            rewritten_queries
        )

        if len(retrieved) == 0:

            yield "I couldn't find any relevant information in the uploaded documents."

            return

        #######################################################
        # Reranking
        #######################################################

        retrieved = self.reranker.rerank(
            query,
            retrieved,
        )

        #######################################################
        # Prompt Builder
        #######################################################

        system_prompt, messages = self.prompt_builder.build(
            query=query,
            retrieved=retrieved,
            chat_history=chat_history,
        )

        #######################################################
        # Stream LLM Response
        #######################################################

        answer = ""

        for token in self.llm.stream(
            system_prompt,
            messages,
        ):

            answer += token

            yield token

        #######################################################
        # Guardrails
        #######################################################

        guard = self.guardrails.run(
            answer=answer,
            context_chunks=[
                doc["text"]
                for doc in retrieved
            ],
        )

        if not guard.passed:

            logger.warning(
                f"Guardrail blocked streamed answer: {guard.reason}"
            )

 