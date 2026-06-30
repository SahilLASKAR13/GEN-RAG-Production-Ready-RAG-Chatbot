from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from deployment.api import router as api_router
from pipeline.pipeline_manager import (
    ingestion_pipeline,
    rag_pipeline,
)
from utils.logger import get_logger


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server starts.
    Uses the singleton pipelines from pipeline_manager.
    """

    logger.info("Starting RAG Chatbot...")

    try:
        # Load existing FAISS index
        rag_pipeline.vector_store.load()

        logger.info("RAG Pipeline initialized successfully.")

    except Exception:
        logger.exception("Failed to initialize RAG Pipeline.")
        raise

    yield

    logger.info("Shutting down application...")


app = FastAPI(
    title="RAG Chatbot API",
    description="Production Ready Retrieval-Augmented Generation Chatbot",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Home"])
def home():

    return JSONResponse(
        {
            "message": "RAG Chatbot API is running.",
            "docs": "/docs",
            "health": "/health",
        }
    )


app.include_router(api_router)