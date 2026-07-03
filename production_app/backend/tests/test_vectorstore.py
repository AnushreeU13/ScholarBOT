import pytest

chromadb = pytest.importorskip("chromadb")

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
