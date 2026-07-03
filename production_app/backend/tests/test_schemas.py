import pytest
from pydantic import ValidationError

from app.schemas import ChatMessage, ChatRequest, ChatResponse, EvidenceChunk, HealthResponse, UploadResponse


def test_chat_request_defaults():
    req = ChatRequest(query="What is TB?")
    assert req.session_id == "default"
    assert req.force_user_kb is False
    assert req.history == []


def test_chat_request_rejects_empty_query():
    with pytest.raises(ValidationError):
        ChatRequest(query="")


def test_chat_message_rejects_invalid_role():
    with pytest.raises(ValidationError):
        ChatMessage(role="system", content="hi")


def test_chat_response_defaults():
    resp = ChatResponse(status="abstain")
    assert resp.citations == []
    assert resp.confidence == 0.0


def test_evidence_chunk_roundtrip():
    chunk = EvidenceChunk(chunk_id=1, text="t", citation="c", store="guidelines_kb")
    assert chunk.model_dump()["chunk_id"] == 1


def test_upload_and_health_response():
    upload = UploadResponse(added_chunks=3, total_chars=100, num_pages=2, doc_name="x.pdf")
    assert upload.added_chunks == 3
    health = HealthResponse(status="ok", guidelines_chunks=1, druglabels_chunks=2, user_chunks=0)
    assert health.status == "ok"
