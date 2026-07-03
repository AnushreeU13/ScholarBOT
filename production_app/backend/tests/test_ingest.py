import pytest

from app.ingest import ingest_user_pdf


def test_ingest_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        ingest_user_pdf("/no/such/file.pdf", doc_name="x.pdf", store=None)


def test_ingest_filters_short_chunks_and_adds_to_store(tmp_path, monkeypatch, make_fake_store):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr(
        "app.ingest.pdf_utils.extract_text_by_page",
        lambda path: [("A reasonably long clinical sentence about tuberculosis treatment.", 1)],
    )
    monkeypatch.setattr(
        "app.ingest.chunking.chunk_document",
        lambda text, doc_name, page_number=None, chunk_size=None, overlap=None: [
            {"text": text, "metadata": {"document_name": doc_name, "page_number": page_number, "chunk_index": 0}},
            {"text": "short", "metadata": {"document_name": doc_name, "page_number": page_number, "chunk_index": 1}},
        ],
    )

    store = make_fake_store("user_kb")
    stats = ingest_user_pdf(pdf_path, doc_name="doc.pdf", store=store)

    assert stats["added_chunks"] == 1  # "short" (< 50 chars) is filtered out
    assert stats["num_pages"] == 1
    assert store.count() == 1


def test_ingest_zero_chunks_when_all_too_short(tmp_path, monkeypatch, make_fake_store):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr("app.ingest.pdf_utils.extract_text_by_page", lambda path: [("short text", 1)])
    monkeypatch.setattr(
        "app.ingest.chunking.chunk_document",
        lambda text, doc_name, page_number=None, chunk_size=None, overlap=None: [
            {"text": "tiny", "metadata": {"page_number": page_number}},
        ],
    )

    store = make_fake_store("user_kb")
    stats = ingest_user_pdf(pdf_path, doc_name="doc.pdf", store=store)
    assert stats["added_chunks"] == 0
    assert store.count() == 0
