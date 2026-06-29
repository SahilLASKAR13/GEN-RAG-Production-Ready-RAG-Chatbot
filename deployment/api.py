# deployment/api.py
# FastAPI application exposing the RAG pipeline over HTTP.
# Endpoints: POST /ingest, POST /chat, GET /chat/stream

from fastapi import APIRouter

from deployment.routes.chat import router as chat_router
from deployment.routes.upload import router as upload_router
from deployment.routes.health import router as health_router

router = APIRouter(prefix="/api")

router.include_router(
    upload_router,
    tags=["Upload"],
)

router.include_router(
    chat_router,
    tags=["Chat"],
)

router.include_router(
    health_router,
    tags=["Health"],
)



