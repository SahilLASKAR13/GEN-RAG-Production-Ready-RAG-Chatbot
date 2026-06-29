from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from pipeline.ingestion_pipeline import IngestionPipeline
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

    pipeline = IngestionPipeline()

    chunks = pipeline.ingest(str(save_path))

    return UploadResponse(
        filename=file.filename,
        chunks=chunks,
        message="Document indexed successfully.",
    )

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ...