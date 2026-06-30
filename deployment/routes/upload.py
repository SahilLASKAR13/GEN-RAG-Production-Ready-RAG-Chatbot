from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from pipeline.pipeline_manager import (
    ingestion_pipeline,
    rag_pipeline,
)
from schemes.upload import UploadResponse


router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post(
    "/upload",
    response_model=UploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
):

    allowed_extensions = {
        ".pdf",
        ".txt",
        ".json",
    }

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT and JSON files are supported.",
        )

    save_path = UPLOAD_DIR / file.filename

    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("\n==============================")
    print("UPLOAD API CALLED")
    print("Saved to:", save_path)
    print("==============================\n")

    # -------------------------------
    # Index the uploaded document
    # -------------------------------
    ingestion_pipeline.ingest_file(str(save_path))

    # -------------------------------
    # Reload FAISS into the shared RAG pipeline
    # -------------------------------
    rag_pipeline.vector_store.load()

    # Share the updated BM25 index
    rag_pipeline.bm25 = ingestion_pipeline.bm25
    rag_pipeline.retriever.bm25 = ingestion_pipeline.bm25

    # Ensure retriever uses the updated FAISS store
    rag_pipeline.retriever.vector_store = rag_pipeline.vector_store

    print("\n==============================")
    print("INGESTION FINISHED")
    print("==============================\n")

    return UploadResponse(
        filename=file.filename,
        chunks=0,  # Replace with actual chunk count if ingest_file() returns it
        message="Document indexed successfully.",
    )