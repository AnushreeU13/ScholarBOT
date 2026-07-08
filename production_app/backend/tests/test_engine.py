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
    assert engine.get_user_store("default").count() == 0


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


def test_user_stores_are_isolated_per_session(engine):
    store_a = engine.get_user_store("session-a")
    store_b = engine.get_user_store("session-b")
    assert store_a is not store_b

    store_a.add_texts(["session A's uploaded document"], [{"document_name": "a.pdf"}])
    assert store_a.count() == 1
    assert store_b.count() == 0  # session B never sees session A's document


def test_get_user_store_is_cached_per_session(engine):
    first = engine.get_user_store("session-a")
    second = engine.get_user_store("session-a")
    assert first is second


def test_user_store_collection_name_is_hashed_not_raw_session_id(fake_embedder, make_fake_store):
    seen_names = []

    def factory(name, embedder=None):
        seen_names.append(name)
        return make_fake_store(name)

    engine = ScholarBotEngine(embedder=fake_embedder, store_factory=factory)
    engine.get_user_store("some session id with spaces/slashes")

    user_kb_calls = [n for n in seen_names if n.startswith(config.COLLECTION_USER)]
    assert len(user_kb_calls) == 1
    # Hashed, not a direct pass-through of client-supplied session_id (which
    # could contain characters Chroma collection names don't allow).
    assert "some session id with spaces/slashes" not in user_kb_calls[0]


def test_reset_session_clears_and_forgets_user_store(engine):
    store = engine.get_user_store("session-a")
    store.add_texts(["doc content"], [{"document_name": "a.pdf"}])
    assert store.count() == 1

    engine.reset_session("session-a")

    fresh = engine.get_user_store("session-a")
    assert fresh.count() == 0


def test_generate_response_only_searches_the_calling_sessions_document(engine, monkeypatch):
    # llm.complete is one shared function used by router.py's LLM fallback AND
    # pipeline.py's generation/critique calls — return "" for the router's
    # classifier call (so it falls back to plain keyword routing) and the
    # canned answer JSON for everything else, otherwise session-b's router
    # call would get a QA-shaped response and misroute instead of cleanly
    # abstaining as out-of-domain.
    def _fake_complete(system, *a, **k):
        if "classifier" in system.lower():
            return ""
        return json.dumps({
            "status": "answer",
            "clinician_bullets": ["Session A's document says X. [1]"],
            "patient_bullets": ["X in plain language. [1]"],
        })

    monkeypatch.setattr("app.pipeline.llm.complete", _fake_complete)

    engine.get_user_store("session-a").add_texts(
        ["Session A's private uploaded document content."],
        [{"document_name": "a.pdf", "page_number": 1, "chunk_index": 0}],
    )
    # session-b never uploaded anything.

    answer_a, _, meta_a = engine.generate_response(
        "Summarize the document I just uploaded.", session_id="session-a"
    )
    answer_b, _, meta_b = engine.generate_response(
        "Summarize the document I just uploaded.", session_id="session-b"
    )

    assert answer_a["status"] == "answer"
    assert meta_a["source"] == config.COLLECTION_USER
    # session-b has no document, so has_user_doc is False for it — the query
    # has no clinical keyword either, so it correctly abstains as out-of-domain
    # rather than ever touching session-a's document.
    assert answer_b["status"] == "abstain"
    assert answer_b["abstain_reason"] == "No domain match."
