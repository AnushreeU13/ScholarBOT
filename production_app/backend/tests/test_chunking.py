import pytest

from app import chunking


class WordTokenizer:
    """1 token per word — deterministic and fast, avoids downloading a real model."""

    def encode(self, text, add_special_tokens=False):
        return text.split()


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch):
    monkeypatch.setattr(chunking, "_get_tokenizer", lambda model_name: WordTokenizer())


def test_empty_text_returns_no_chunks():
    assert chunking.semantic_chunk_text("") == []
    assert chunking.semantic_chunk_text(None) == []


def test_single_short_sentence_is_one_chunk():
    chunks = chunking.semantic_chunk_text("Isoniazid treats tuberculosis.", chunk_size=50, overlap=5)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Isoniazid treats tuberculosis."


def test_long_text_splits_into_multiple_chunks_with_overlap():
    sentences = [f"Sentence number {i} about tuberculosis treatment protocols." for i in range(20)]
    text = " ".join(sentences)
    chunks = chunking.semantic_chunk_text(text, chunk_size=30, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert c["token_count"] > 0


def test_abbreviations_do_not_split_sentences():
    text = "The patient takes 5 mg. of the drug daily. This is standard care."
    sentences = chunking._split_sentences(text)
    # "5 mg." should not have caused a false split before "of the drug"
    assert not any(s.strip() == "5 mg." for s in sentences)


def test_chunk_document_attaches_metadata():
    chunks = chunking.chunk_document("Some clinical text about TB.", "doc.pdf", page_number=3,
                                      chunk_size=50, overlap=5)
    assert len(chunks) == 1
    meta = chunks[0]["metadata"]
    assert meta["document_name"] == "doc.pdf"
    assert meta["page_number"] == 3
    assert meta["chunk_index"] == 0
    assert meta["total_chunks"] == 1
