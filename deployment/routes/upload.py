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

    print("\n==============================")
    print("UPLOAD API CALLED")
    print("Saved to:", save_path)
    print("==============================\n")

    pipeline = IngestionPipeline()

    pipeline.ingest_file(str(save_path))

    print("\n==============================")
    print("INGESTION FINISHED")
    print("==============================\n")

    return UploadResponse(
        filename=file.filename,
        chunks=0,
        message="Document indexed successfully.",
    )
