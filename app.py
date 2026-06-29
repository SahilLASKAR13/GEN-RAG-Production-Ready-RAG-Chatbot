from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from deployment.api import router as api_router
from pipeline.rag_pipeline import RAGPipeline
from utils.logger import get_logger


logger = get_logger(__name__)


# Global RAG Pipeline
rag_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server starts.
    """

    global rag_pipeline

    logger.info("Starting RAG Chatbot...")

    try:
        rag_pipeline = RAGPipeline()

        logger.info("RAG Pipeline initialized successfully.")

    except Exception as e:
        logger.exception("Failed to initialize RAG Pipeline.")
        raise e

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