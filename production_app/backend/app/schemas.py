"""Pydantic request/response models for the public API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str = "default"
    force_user_kb: bool = False
    history: List[ChatMessage] = Field(default_factory=list)


class EvidenceChunk(BaseModel):
    chunk_id: int
    text: str
    citation: str
    store: str


class ChatResponse(BaseModel):
    status: str
    abstain_reason: str = ""
    clinician_bullets: List[str] = Field(default_factory=list)
    patient_bullets: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    evidence_chunks: List[EvidenceChunk] = Field(default_factory=list)
    source: str = ""


class UploadResponse(BaseModel):
    added_chunks: int
    total_chars: int
    num_pages: int
    doc_name: str


class HealthResponse(BaseModel):
    status: str
    guidelines_chunks: int
    druglabels_chunks: int
    user_chunks: int
