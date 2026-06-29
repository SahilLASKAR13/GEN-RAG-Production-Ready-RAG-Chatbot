# deployment/api.py
# FastAPI application exposing the RAG pipeline over HTTP.
# Endpoints: POST /ingest, POST /chat, GET /chat/stream

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pipeline import RAGPipeline, RAGResponse
from utils.logger import get_logger


log = get_logger(__name__)
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Shared pipeline instance (initialised once on startup)
_pipeline: RAGPipeline | None = None


@app.on_event("startup")
async def startup():
    global _pipeline
    _pipeline = RAGPipeline()
    log.info("Pipeline loaded")


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return _pipeline


# ── Request / Response schemas ────────────────────────────────────────────────

class IngestRequest(BaseModel):
    sources: list[str] = Field(..., description="File paths or URLs to ingest")


class IngestResponse(BaseModel):
    chunks_indexed: int


class ChatRequest(BaseModel):
    query: str
    history: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    model: str
    input_tokens: int
    output_tokens: int
    rewritten_queries: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Ingest documents into the vector store."""
    try:
        n = get_pipeline().index_documents(req.sources)
        return IngestResponse(chunks_indexed=n)
    except Exception as exc:
        log.error({"action": "ingest_error", "error": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Run a full RAG query and return the answer with sources."""
    try:
        resp: RAGResponse = get_pipeline().query(req.query, req.history or None)
        return ChatResponse(
            answer=resp.answer,
            sources=resp.sources,
            model=resp.model,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            rewritten_queries=resp.rewritten_queries,
        )
    except Exception as exc:
        log.error({"action": "chat_error", "error": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/chat/stream")
async def chat_stream(query: str, history: str = "[]"):
    """Server-sent events streaming endpoint."""
    import json

    def generate():
        try:
            hist = json.loads(history)
            for token in get_pipeline().stream_query(query, hist or None):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}
