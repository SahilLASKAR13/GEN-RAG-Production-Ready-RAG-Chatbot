from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

# Use the shared pipeline
from pipeline.pipeline_manager import rag_pipeline

from schemes.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    try:
        result = rag_pipeline.run(
            query=request.query,
            chat_history=request.chat_history,
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):

    def event_stream():
        for token in rag_pipeline.stream(
            query=request.query,
            chat_history=request.chat_history,
        ):
            yield token

    return StreamingResponse(
        event_stream(),
        media_type="text/plain",
    )