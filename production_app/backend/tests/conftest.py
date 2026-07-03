import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fake_vec(text: str, dim: int = 8):
    h = sum(ord(c) for c in text) if text else 0
    return [((h >> i) % 7) / 7.0 + 0.001 for i in range(dim)]


class FakeEmbedder:
    dim = 8
    name = "fake-embedder"

    def embed_query(self, text):
        if not text or not text.strip():
            return [0.0] * self.dim
        return _fake_vec(text, self.dim)

    def embed_texts(self, texts, batch_size=32):
        return [_fake_vec(t, self.dim) if t.strip() else [0.0] * self.dim for t in texts]


class FakeStore:
    """In-memory stand-in for ChromaStore, same public interface."""

    def __init__(self, name="fake", embedder=None):
        self.collection_name = name
        self.embedder = embedder or FakeEmbedder()
        self._docs = []  # list of (id, text, meta, vector)

    def add_texts(self, texts, metadatas, ids=None):
        vectors = self.embedder.embed_texts(texts)
        added_ids = []
        for t, m, v in zip(texts, metadatas, vectors):
            _id = f"{self.collection_name}-{len(self._docs)}"
            self._docs.append((_id, t, m, v))
            added_ids.append(_id)
        return added_ids

    def similarity_search_by_vector(self, query_vector, k=20):
        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1e-9
            nb = math.sqrt(sum(x * x for x in b)) or 1e-9
            return dot / (na * nb)

        scored = [(cos(query_vector, d[3]), d) for d in self._docs]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"text": d[1], "metadata": d[2], "score": s} for s, d in scored[:k]]

    def count(self):
        return len(self._docs)

    def all_documents(self):
        return [{"text": d[1], "metadata": d[2]} for d in self._docs]

    def delete_all(self):
        self._docs = []


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def make_fake_store(fake_embedder):
    def _make(name="fake"):
        return FakeStore(name, embedder=fake_embedder)

    return _make
