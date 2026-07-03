import numpy as np
import pytest

from app import embedder as embedder_mod


class FakeSentenceTransformer:
    """Stands in for sentence_transformers.SentenceTransformer — no download, no torch inference."""

    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device
        self.max_seq_length = 512
        self.last_call = None

    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True):
        self.last_call = texts
        if isinstance(texts, str):
            return np.array([float(len(texts))] * 4)
        return np.array([[float(len(t))] * 4 for t in texts])


@pytest.fixture(autouse=True)
def reset_singleton():
    embedder_mod.reset_embedder()
    yield
    embedder_mod.reset_embedder()


@pytest.fixture
def patched_st(monkeypatch):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(embedder_mod, "_device", lambda: "cpu")


def test_embed_query_prepends_instruction_prefix(patched_st):
    emb = embedder_mod.BGEEmbedder()
    vec = emb.embed_query("tuberculosis dosage")
    assert len(vec) == 4
    expected_len = len(emb._QUERY_PREFIX + "tuberculosis dosage")
    assert vec[0] == float(expected_len)


def test_embed_query_empty_text_returns_zero_vector(patched_st):
    emb = embedder_mod.BGEEmbedder()
    vec = emb.embed_query("   ")
    assert vec == [0.0] * emb.dim


def test_embed_texts_empty_list_returns_empty(patched_st):
    emb = embedder_mod.BGEEmbedder()
    assert emb.embed_texts([]) == []


def test_embed_texts_returns_one_vector_per_text(patched_st):
    emb = embedder_mod.BGEEmbedder()
    vecs = emb.embed_texts(["a", "bb", "ccc"])
    assert len(vecs) == 3


def test_get_embedder_singleton(patched_st):
    a = embedder_mod.get_embedder()
    b = embedder_mod.get_embedder()
    assert a is b


def test_reset_embedder_clears_singleton(patched_st):
    a = embedder_mod.get_embedder()
    embedder_mod.reset_embedder()
    b = embedder_mod.get_embedder()
    assert a is not b


def test_device_detection_without_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert embedder_mod._device() == "cpu"
