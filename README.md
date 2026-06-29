# RAG Chatbot — Modular Implementation

## Project Structure

```
rag_chatbot/
├── config/
│   └── settings.py          # Centralised config (env vars, constants)
├── ingestion/
│   ├── loaders.py           # Document loaders (PDF, web, DB, API)
│   └── chunker.py           # Text splitting strategies
├── indexing/
│   ├── embedder.py          # Embedding model wrapper
│   └── vector_store.py      # Vector DB abstraction (Pinecone / Chroma / FAISS)
├── retrieval/
│   ├── query_rewriter.py    # HyDE, step-back, multi-query
│   ├── retriever.py         # Dense + sparse hybrid retrieval
│   └── reranker.py          # Cross-encoder reranking
├── generation/
│   ├── prompt_builder.py    # Prompt templates + context assembly
│   ├── llm.py               # LLM wrapper (OpenAI / Anthropic / local)
│   └── guardrails.py        # Hallucination check, safety filter
├── utils/
│   ├── logger.py            # Structured logging
│   └── tracing.py           # OpenTelemetry spans
├── deployment/
│   ├── api.py               # FastAPI app
│   └── Dockerfile           # Container definition
├── pipeline.py              # Orchestrates all modules end-to-end
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in API keys
uvicorn deployment.api:app --reload
```

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI key for embeddings + LLM |
| `PINECONE_API_KEY` | Pinecone key |
| `PINECONE_INDEX` | Index name |
| `COHERE_API_KEY` | Cohere key for reranking |
| `LOG_LEVEL` | debug / info / warning |
