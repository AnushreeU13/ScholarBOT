"""FastAPI application entrypoint. Run with: uvicorn app.main:app"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import config
from app.engine import ScholarBotEngine
from app.ingest import ingest_user_pdf
from app.schemas import ChatRequest, ChatResponse, HealthResponse, UploadResponse

app = FastAPI(
    title="ScholarBOT API",
    description="Evidence-only clinical RAG over TB / pneumonia guidelines and drug labels.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: ScholarBotEngine | None = None


def get_engine() -> ScholarBotEngine:
    global _engine
    if _engine is None:
        _engine = ScholarBotEngine()
    return _engine


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    engine = get_engine()
    return HealthResponse(
        status="ok",
        guidelines_chunks=engine.guidelines_store.count(),
        druglabels_chunks=engine.druglabels_store.count(),
        user_chunks=engine.user_store.count(),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    engine = get_engine()
    history = [m.model_dump() for m in req.history]
    answer, confidence, meta = engine.generate_response(
        query=req.query,
        session_id=req.session_id,
        force_user_kb=req.force_user_kb,
        history=history,
    )
    return ChatResponse(
        status=answer["status"],
        abstain_reason=answer["abstain_reason"],
        clinician_bullets=answer["clinician_bullets"],
        patient_bullets=answer["patient_bullets"],
        citations=answer["citations"],
        confidence=confidence,
        evidence_chunks=meta["evidence_chunks"],
        source=meta["source"],
    )


@app.post("/session/{session_id}/reset")
def reset_session(session_id: str) -> dict:
    get_engine().reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    engine = get_engine()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / file.filename
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            stats = ingest_user_pdf(tmp_path, doc_name=file.filename, store=engine.user_store)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Ingestion failed: {e}") from e

    if stats["added_chunks"] == 0:
        raise HTTPException(status_code=422, detail="Zero chunks extracted — is this a scanned/image-only PDF?")

    engine.reload_user_kb()
    return UploadResponse(
        added_chunks=stats["added_chunks"],
        total_chars=stats["total_chars"],
        num_pages=stats["num_pages"],
        doc_name=file.filename,
    )


# Single-container deployments (Hugging Face Spaces) build the frontend into
# STATIC_DIR and serve it from this same FastAPI process, alongside the API
# routes above. Docker Compose / Kubernetes instead run the frontend as its
# own nginx container, so STATIC_DIR is unset there and this mount is skipped.
_static_dir = os.getenv("STATIC_DIR")
if _static_dir and Path(_static_dir).is_dir():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
