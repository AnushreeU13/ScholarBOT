import json

import pytest

from app.pipeline import RAGPipeline


class FakeRetriever:
    def __init__(self, chunks=None, sample=None):
        self._chunks = chunks if chunks is not None else []
        self._sample = sample if sample is not None else []

    def retrieve(self, query, target_kbs, **kwargs):
        return self._chunks

    def stratified_sample(self, store_name, n=16, **kwargs):
        return self._sample


def _chunk(chunk_id=1, text="Isoniazid 5mg/kg daily is the standard treatment.", score=0.9, store="guidelines_kb"):
    return {
        "chunk_id": chunk_id, "text": text, "score": score, "store": store,
        "citation": f"KB: Existing KB - Guidelines, Document: WHO Guideline, Page 1",
        "metadata": {},
    }


def test_route_abstain_short_circuits_before_retrieval():
    pipeline = RAGPipeline(FakeRetriever(), sufficiency_check=lambda q, c: True)
    result = pipeline.run("hello", {"abstain": True, "reason": "out_of_scope"})
    assert result.status == "abstain"
    assert result.abstain_reason == "out_of_scope"


def test_guideline_qa_no_chunks_abstains():
    pipeline = RAGPipeline(FakeRetriever(chunks=[]), sufficiency_check=lambda q, c: True)
    result = pipeline.run("what is TB", {"abstain": False, "intent": "general", "target_kbs": ["guidelines_kb"]})
    assert result.status == "abstain"
    assert result.abstain_reason == "no_chunks_retrieved"


def test_guideline_qa_low_confidence_abstains():
    pipeline = RAGPipeline(FakeRetriever(chunks=[_chunk(score=0.1)]), sufficiency_check=lambda q, c: True)
    result = pipeline.run("what is TB", {"abstain": False, "intent": "general", "target_kbs": ["guidelines_kb"]})
    assert result.status == "abstain"
    assert "low_confidence" in result.abstain_reason


def test_guideline_qa_insufficient_evidence_abstains():
    pipeline = RAGPipeline(FakeRetriever(chunks=[_chunk()]), sufficiency_check=lambda q, c: False)
    result = pipeline.run("what is TB", {"abstain": False, "intent": "general", "target_kbs": ["guidelines_kb"]})
    assert result.status == "abstain"
    assert result.abstain_reason == "evidence_insufficient_for_query"


def test_guideline_qa_llm_abstain(monkeypatch):
    monkeypatch.setattr("app.pipeline.llm.complete", lambda *a, **k: "")
    pipeline = RAGPipeline(FakeRetriever(chunks=[_chunk()]), sufficiency_check=lambda q, c: True)
    result = pipeline.run("what is TB", {"abstain": False, "intent": "general", "target_kbs": ["guidelines_kb"]})
    assert result.status == "abstain"
    assert result.abstain_reason == "llm_abstain"


def test_guideline_qa_answer_with_critique(monkeypatch):
    responses = iter([
        json.dumps({
            "status": "answer",
            "clinician_bullets": ["Isoniazid 5mg/kg daily is standard. [1]"],
            "patient_bullets": ["Take isoniazid once a day. [1]"],
        }),
        json.dumps({
            "clinician_bullets": ["Isoniazid 5mg/kg daily is standard. [1]"],
            "patient_bullets": ["Take isoniazid once a day. [1]"],
        }),
    ])
    monkeypatch.setattr("app.pipeline.llm.complete", lambda *a, **k: next(responses))
    pipeline = RAGPipeline(FakeRetriever(chunks=[_chunk()]), sufficiency_check=lambda q, c: True)
    result = pipeline.run("dosage of isoniazid", {"abstain": False, "intent": "general", "target_kbs": ["guidelines_kb"]})
    assert result.status == "answer"
    assert result.clinician_bullets
    assert result.citations
    assert result.evidence_chunks[0]["chunk_id"] == 1


def test_critique_rejects_all_bullets(monkeypatch):
    responses = iter([
        json.dumps({"status": "answer", "clinician_bullets": ["unsupported claim [1]"], "patient_bullets": []}),
        json.dumps({"clinician_bullets": [], "patient_bullets": []}),
    ])
    monkeypatch.setattr("app.pipeline.llm.complete", lambda *a, **k: next(responses))
    pipeline = RAGPipeline(FakeRetriever(chunks=[_chunk()]), sufficiency_check=lambda q, c: True)
    result = pipeline.run("q", {"abstain": False, "intent": "general", "target_kbs": ["guidelines_kb"]})
    assert result.status == "abstain"
    assert result.abstain_reason == "critique_rejected_all"


def test_drug_qa_path(monkeypatch):
    monkeypatch.setattr("app.pipeline.llm.complete", lambda *a, **k: json.dumps({
        "status": "answer",
        "clinician_bullets": ["Max dose is 300mg/day. [1]"],
        "patient_bullets": ["Take as directed. [1]"],
    }))
    chunks = [_chunk(store="druglabels_kb")]
    pipeline = RAGPipeline(FakeRetriever(chunks=chunks), sufficiency_check=lambda q, c: True)
    result = pipeline.run("isoniazid dosage", {"abstain": False, "intent": "drug_info", "target_kbs": ["druglabels_kb"]})
    assert result.status == "answer"


def test_drug_qa_no_chunks_abstains():
    pipeline = RAGPipeline(FakeRetriever(chunks=[]), sufficiency_check=lambda q, c: True)
    result = pipeline.run("isoniazid dosage", {"abstain": False, "domain": "drug", "target_kbs": []})
    assert result.status == "abstain"
    assert result.abstain_reason == "no_drug_chunks_retrieved"


def test_summarize_no_target_kb_abstains():
    pipeline = RAGPipeline(FakeRetriever(), sufficiency_check=lambda q, c: True)
    result = pipeline.run("summarize this", {"abstain": False, "intent": "summarize", "target_kbs": []})
    assert result.status == "abstain"
    assert result.abstain_reason == "no_target_kb_for_summarize"


def test_summarize_empty_store_abstains():
    pipeline = RAGPipeline(FakeRetriever(sample=[]), sufficiency_check=lambda q, c: True)
    result = pipeline.run("summarize this", {"abstain": False, "intent": "summarize", "target_kbs": ["user_kb"]})
    assert result.status == "abstain"
    assert result.abstain_reason == "empty_store_for_summarize"


def test_summarize_answer(monkeypatch):
    monkeypatch.setattr("app.pipeline.llm.complete", lambda *a, **k: json.dumps({
        "status": "answer",
        "clinician_bullets": ["Document covers TB treatment. [1]"],
        "patient_bullets": ["This document is about TB treatment. [1]"],
    }))
    sample = [_chunk(store="user_kb")]
    pipeline = RAGPipeline(FakeRetriever(sample=sample), sufficiency_check=lambda q, c: True)
    result = pipeline.run("summarize this", {"abstain": False, "intent": "summarize", "target_kbs": ["user_kb"]})
    assert result.status == "answer"
    assert result.confidence == 1.0
