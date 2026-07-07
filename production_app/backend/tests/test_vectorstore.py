import threading
import time

import pytest

chromadb = pytest.importorskip("chromadb")

from app import vectorstore  # noqa: E402
from app.vectorstore import ChromaStore, _sanitize_metadata  # noqa: E402


class FakeEmbedder:
    dim = 4

    def embed_texts(self, texts, batch_size=32):
        return [[float(len(t)), 0.0, 0.0, 1.0] for t in texts]


@pytest.fixture
def ephemeral_client():
    return chromadb.EphemeralClient()


def test_sanitize_metadata_drops_none_and_stringifies_lists():
    meta = {"a": None, "b": 1, "c": "x", "d": [1, 2], "e": True}
    clean = _sanitize_metadata(meta)
    assert "a" not in clean
    assert clean["b"] == 1
    assert clean["c"] == "x"
    assert clean["d"] == "[1, 2]"
    assert clean["e"] is True


def test_add_and_query_roundtrip(ephemeral_client):
    store = ChromaStore("test_collection", client=ephemeral_client, embedder=FakeEmbedder())
    ids = store.add_texts(
        ["short", "a much longer piece of text"],
        [{"document_name": "doc1"}, {"document_name": "doc2"}],
    )
    assert len(ids) == 2
    assert store.count() == 2

    hits = store.similarity_search_by_vector([5.0, 0.0, 0.0, 1.0], k=2)
    assert len(hits) == 2
    assert hits[0]["text"] == "short"  # closest vector match


def test_query_on_empty_collection_returns_empty(ephemeral_client):
    store = ChromaStore("empty_collection", client=ephemeral_client, embedder=FakeEmbedder())
    assert store.similarity_search_by_vector([0.0, 0.0, 0.0, 0.0], k=5) == []


def test_add_texts_with_empty_list_is_noop(ephemeral_client):
    store = ChromaStore("noop_collection", client=ephemeral_client, embedder=FakeEmbedder())
    assert store.add_texts([], []) == []


def test_delete_all(ephemeral_client):
    store = ChromaStore("delete_collection", client=ephemeral_client, embedder=FakeEmbedder())
    store.add_texts(["a", "b"], [{}, {}])
    assert store.count() == 2
    store.delete_all()
    assert store.count() == 0


def test_all_documents(ephemeral_client):
    store = ChromaStore("all_docs_collection", client=ephemeral_client, embedder=FakeEmbedder())
    store.add_texts(["a", "b"], [{"x": 1}, {"x": 2}])
    docs = store.all_documents()
    assert len(docs) == 2
    assert {d["text"] for d in docs} == {"a", "b"}


# ── Concurrency regression tests ────────────────────────────────────────────
# A production Space hit this exact race: several concurrent first-requests
# (platform startup health probes) each saw the singleton as uninitialized and
# raced to construct their own chromadb.PersistentClient for the same path,
# corrupting Chroma's shared-system registry (AttributeError/KeyError deep in
# chromadb internals). get_store/_get_client must serialize construction so
# only one instance is ever built, however many threads call in concurrently.

def test_get_store_constructs_exactly_once_under_concurrency(monkeypatch):
    vectorstore.reset_stores()
    construction_count = 0
    lock = threading.Lock()

    class SlowChromaStore:
        def __init__(self, collection_name, client=None, embedder=None):
            nonlocal construction_count
            with lock:
                construction_count += 1
            time.sleep(0.05)  # widen the race window
            self.collection_name = collection_name

    monkeypatch.setattr(vectorstore, "ChromaStore", SlowChromaStore)

    results = []

    def worker():
        results.append(vectorstore.get_store("guidelines_kb"))

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert construction_count == 1
    assert len({id(r) for r in results}) == 1  # every thread got the same instance

    vectorstore.reset_stores()


def test_get_client_constructs_exactly_once_under_concurrency(monkeypatch, tmp_path):
    vectorstore.reset_client()
    monkeypatch.setattr(vectorstore.config, "CHROMA_DIR", tmp_path)

    construction_count = 0
    lock = threading.Lock()

    class SlowClient:
        def __init__(self, path):
            nonlocal construction_count
            with lock:
                construction_count += 1
            time.sleep(0.05)
            self.path = path

    fake_chromadb = type("FakeChromaModule", (), {"PersistentClient": SlowClient})

    import sys
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)

    results = []

    def worker():
        results.append(vectorstore._get_client())

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert construction_count == 1
    assert len({id(r) for r in results}) == 1

    vectorstore.reset_client()
