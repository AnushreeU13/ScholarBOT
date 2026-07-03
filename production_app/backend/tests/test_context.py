from app.context import ContextManager


def test_strip_meta_reference():
    cm = ContextManager()
    resolved = cm.resolve("According to the document, what is the dosage?", history=[])
    assert "according to" not in resolved.lower()
    assert "dosage" in resolved.lower()


def test_reset_clears_topic():
    cm = ContextManager()
    cm.topic_summary = "tuberculosis"
    cm.reset()
    assert cm.topic_summary == ""


def test_topic_shift_resets_summary():
    cm = ContextManager()
    cm.topic_summary = "tuberculosis"
    resolved = cm.resolve("what about pneumonia now", history=[])
    assert cm.topic_summary == ""
    assert resolved  # non-empty


def test_coreference_resolution_calls_llm(monkeypatch):
    cm = ContextManager()
    cm.topic_summary = "tuberculosis"
    monkeypatch.setattr("app.context.llm.complete", lambda *a, **k: "What is the treatment for tuberculosis?")
    resolved = cm.resolve("how is it treated?", history=[{"role": "user", "content": "tell me about TB"}])
    assert resolved == "What is the treatment for tuberculosis?"


def test_coreference_resolution_falls_back_on_bad_llm_output(monkeypatch):
    cm = ContextManager()
    cm.topic_summary = "tuberculosis"
    monkeypatch.setattr("app.context.llm.complete", lambda *a, **k: "")
    original = "how is it treated?"
    resolved = cm.resolve(original, history=[])
    assert resolved == original


def test_update_topic_only_on_answer(monkeypatch):
    cm = ContextManager()
    monkeypatch.setattr("app.context.llm.complete", lambda *a, **k: "tuberculosis")
    cm.update_topic("what is tuberculosis", answer_status="abstain")
    assert cm.topic_summary == ""

    cm.update_topic("what is tuberculosis", answer_status="answer")
    assert cm.topic_summary == "tuberculosis"


def test_update_topic_ignores_overlong_response(monkeypatch):
    cm = ContextManager()
    monkeypatch.setattr("app.context.llm.complete", lambda *a, **k: "x" * 100)
    cm.update_topic("query", answer_status="answer")
    assert cm.topic_summary == ""
