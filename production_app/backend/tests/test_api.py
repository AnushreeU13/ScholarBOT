import io
import threading

import pytest
from fastapi.testclient import TestClient

from app import main


class FakeCountStore:
    def __init__(self, n):
        self._n = n

    def count(self):
        return self._n


class FakeEngine:
    def __init__(self):
        self.guidelines_store = FakeCountStore(3219)
        self.druglabels_store = FakeCountStore(15185)
        self._user_stores = {}
        self.reset_calls = []

    def get_user_store(self, session_id):
        if session_id not in self._user_stores:
            self._user_stores[session_id] = FakeCountStore(0)
        return self._user_stores[session_id]

    def generate_response(self, query, session_id="default", force_user_kb=False, history=None):
        if "isoniazid" in query.lower():
            answer = {
                "status": "answer", "abstain_reason": "",
                "clinician_bullets": ["Isoniazid 5mg/kg daily. [1]"],
                "patient_bullets": ["Take isoniazid daily. [1]"],
                "citations": ["KB: Existing KB - Guidelines, Document: WHO, Page 1"],
            }
            meta = {
                "status": "answer", "abstain_reason": "", "source": "guidelines_kb",
                "evidence_chunks": [{"chunk_id": 1, "text": "evidence", "citation": "c", "store": "guidelines_kb"}],
            }
            return answer, 0.87, meta

        answer = {"status": "abstain", "abstain_reason": "out_of_scope",
                   "clinician_bullets": [], "patient_bullets": [], "citations": []}
        meta = {"status": "abstain", "abstain_reason": "out_of_scope", "source": "", "evidence_chunks": []}
        return answer, 0.0, meta

    def reset_session(self, session_id):
        self.reset_calls.append(session_id)
        self._user_stores.pop(session_id, None)


@pytest.fixture
def client(monkeypatch):
    fake_engine = FakeEngine()
    monkeypatch.setattr(main, "get_engine", lambda: fake_engine)
    return TestClient(main.app), fake_engine


def test_health_endpoint(client):
    test_client, _ = client
    resp = test_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["guidelines_chunks"] == 3219


def test_chat_endpoint_answer(client):
    test_client, _ = client
    resp = test_client.post("/chat", json={"query": "isoniazid dosage"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "answer"
    assert body["confidence"] == 0.87
    assert body["citations"]


def test_chat_endpoint_abstain(client):
    test_client, _ = client
    resp = test_client.post("/chat", json={"query": "what's the weather"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "abstain"


def test_chat_endpoint_rejects_empty_query(client):
    test_client, _ = client
    resp = test_client.post("/chat", json={"query": ""})
    assert resp.status_code == 422


def test_reset_session_endpoint(client):
    test_client, fake_engine = client
    resp = test_client.post("/session/abc/reset")
    assert resp.status_code == 200
    assert fake_engine.reset_calls == ["abc"]


def test_health_is_scoped_to_session_id(client):
    test_client, fake_engine = client
    fake_engine.get_user_store("session-a")._n = 5
    fake_engine.get_user_store("session-b")._n = 12

    resp_a = test_client.get("/health", params={"session_id": "session-a"})
    resp_b = test_client.get("/health", params={"session_id": "session-b"})
    resp_default = test_client.get("/health")

    assert resp_a.json()["user_chunks"] == 5
    assert resp_b.json()["user_chunks"] == 12
    assert resp_default.json()["user_chunks"] == 0


def test_upload_rejects_non_pdf(client):
    test_client, _ = client
    resp = test_client.post(
        "/upload", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    )
    assert resp.status_code == 400


def test_upload_success(client, monkeypatch):
    test_client, fake_engine = client
    monkeypatch.setattr(
        "app.main.ingest_user_pdf",
        lambda tmp_path, doc_name, store: {"added_chunks": 4, "total_chars": 200, "num_pages": 2},
    )
    resp = test_client.post(
        "/upload", files={"file": ("guideline.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["added_chunks"] == 4
    assert body["doc_name"] == "guideline.pdf"


def test_upload_ingests_into_the_requesting_sessions_store(client, monkeypatch):
    test_client, fake_engine = client
    seen_stores = []

    def _fake_ingest(tmp_path, doc_name, store):
        seen_stores.append(store)
        store._n = 7
        return {"added_chunks": 7, "total_chars": 300, "num_pages": 3}

    monkeypatch.setattr("app.main.ingest_user_pdf", _fake_ingest)

    test_client.post(
        "/upload",
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
        data={"session_id": "session-x"},
    )

    assert seen_stores == [fake_engine.get_user_store("session-x")]
    # A different session's store must be untouched.
    assert fake_engine.get_user_store("session-y").count() == 0


def test_upload_zero_chunks_returns_422(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(
        "app.main.ingest_user_pdf",
        lambda tmp_path, doc_name, store: {"added_chunks": 0, "total_chars": 0, "num_pages": 1},
    )
    resp = test_client.post(
        "/upload", files={"file": ("scanned.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")}
    )
    assert resp.status_code == 422


def test_upload_ingestion_failure_returns_422(client, monkeypatch):
    test_client, _ = client

    def _raise(tmp_path, doc_name, store):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.main.ingest_user_pdf", _raise)
    resp = test_client.post(
        "/upload", files={"file": ("broken.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")}
    )
    assert resp.status_code == 422


def test_get_engine_constructs_exactly_once_under_concurrency(monkeypatch):
    """
    Regression test for a real production bug: concurrent first-requests (a
    platform's startup health probes hitting the container simultaneously)
    each saw `_engine is None` and raced to build their own ScholarBotEngine,
    which corrupted Chroma's PersistentClient shared-system registry.
    get_engine() must serialize construction to exactly one instance.
    """
    import time

    monkeypatch.setattr(main, "_engine", None)
    construction_count = 0
    lock = threading.Lock()

    class SlowEngine:
        def __init__(self):
            nonlocal construction_count
            with lock:
                construction_count += 1
            time.sleep(0.05)

    monkeypatch.setattr(main, "ScholarBotEngine", SlowEngine)

    results = []

    def worker():
        results.append(main.get_engine())

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert construction_count == 1
    assert len({id(r) for r in results}) == 1
