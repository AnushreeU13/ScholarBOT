import json

import pytest

from app import config
from app.engine import ScholarBotEngine


class FakeReranker:
    def predict(self, pairs):
        return [0.9 for _ in pairs]


@pytest.fixture
def engine(monkeypatch, fake_embedder, make_fake_store):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("app.retriever._get_reranker", lambda: FakeReranker())

    stores = {}

    def factory(name, embedder=None):
        if name not in stores:
            stores[name] = make_fake_store(name)
        return stores[name]

    return ScholarBotEngine(embedder=fake_embedder, store_factory=factory)


def test_health_counts_start_empty(engine):
    assert engine.guidelines_store.count() == 0
    assert engine.druglabels_store.count() == 0
    assert engine.user_store.count() == 0


def test_out_of_domain_query_abstains(engine):
    answer, confidence, meta = engine.generate_response("What's the weather today?")
    assert answer["status"] == "abstain"
    assert confidence == 0.0


def test_in_domain_query_without_evidence_abstains(engine):
    answer, confidence, meta = engine.generate_response("What is the treatment for tuberculosis?")
    assert answer["status"] == "abstain"
    assert answer["abstain_reason"] == "no_chunks_retrieved"


def test_in_domain_query_with_evidence_answers(engine, monkeypatch):
    engine.guidelines_store.add_texts(
        ["Isoniazid 5mg/kg daily is the WHO-recommended first-line TB treatment."],
        [{"document_name": "WHO Guideline", "page_number": 12}],
    )

    # Engine calls the LLM three times for a full answer turn: QA generation,
    # self-critique, then ContextManager.update_topic's topic extraction.
    responses = iter([
        json.dumps({
            "status": "answer",
            "clinician_bullets": ["Isoniazid 5mg/kg daily is first-line. [1]"],
            "patient_bullets": ["Take isoniazid daily as prescribed. [1]"],
        }),
        json.dumps({
            "clinician_bullets": ["Isoniazid 5mg/kg daily is first-line. [1]"],
            "patient_bullets": ["Take isoniazid daily as prescribed. [1]"],
        }),
        "tuberculosis",
    ])
    monkeypatch.setattr("app.pipeline.llm.complete", lambda *a, **k: next(responses))

    answer, confidence, meta = engine.generate_response("What is the treatment for tuberculosis?")
    assert answer["status"] == "answer"
    assert answer["citations"]
    assert confidence > 0
    assert meta["source"] == config.COLLECTION_GUIDELINES


def test_sessions_are_isolated(engine):
    ctx_a = engine._get_session("session-a")
    ctx_a.topic_summary = "tuberculosis"
    ctx_b = engine._get_session("session-b")
    assert ctx_b.topic_summary == ""


def test_reset_session_clears_state(engine):
    ctx = engine._get_session("session-a")
    ctx.topic_summary = "tuberculosis"
    engine.reset_session("session-a")
    fresh = engine._get_session("session-a")
    assert fresh.topic_summary == ""


def test_reload_user_kb_swaps_store(engine, monkeypatch, fake_embedder, make_fake_store):
    new_store = make_fake_store(config.COLLECTION_USER)
    new_store.add_texts(["uploaded content"], [{"document_name": "upload.pdf"}])
    monkeypatch.setattr("app.vectorstore.get_store", lambda name, embedder=None: new_store)

    engine.reload_user_kb()
    assert engine.user_store.count() == 1
    assert engine.pipeline.retriever.stores[config.COLLECTION_USER] is new_store
