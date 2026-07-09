from app.retriever import Retriever, _build_citation, check_sufficiency


class FakeReranker:
    def predict(self, pairs):
        # Score higher for pairs whose text contains "isoniazid"
        return [1.0 if "isoniazid" in text.lower() else 0.2 for _, text in pairs]


def test_dense_search_and_dedup_and_rerank(monkeypatch, make_fake_store, fake_embedder):
    store = make_fake_store("guidelines_kb")
    store.add_texts(
        ["Isoniazid is used for latent TB.", "Isoniazid is used for latent TB.", "Rifampin covers active TB."],
        [{"document_name": "doc1", "page_number": 1}] * 3,
    )
    retriever = Retriever(stores={"guidelines_kb": store}, embedder=fake_embedder)
    monkeypatch.setattr("app.retriever._get_reranker", lambda: FakeReranker())

    results = retriever.retrieve("isoniazid dosage", ["guidelines_kb"])

    assert len(results) == 2  # deduplicated
    assert results[0]["text"].lower().startswith("isoniazid")
    assert results[0]["chunk_id"] == 1
    assert "citation" in results[0]


def test_retrieve_with_missing_store_returns_empty(fake_embedder):
    retriever = Retriever(stores={}, embedder=fake_embedder)
    assert retriever.retrieve("query", ["nonexistent_kb"]) == []


def test_rerank_failure_falls_back_gracefully(monkeypatch, make_fake_store, fake_embedder):
    store = make_fake_store("guidelines_kb")
    store.add_texts(["Some clinical text."], [{"document_name": "doc1"}])
    retriever = Retriever(stores={"guidelines_kb": store}, embedder=fake_embedder)

    def _broken_reranker():
        raise RuntimeError("model load failed")

    monkeypatch.setattr("app.retriever._get_reranker", _broken_reranker)
    results = retriever.retrieve("query", ["guidelines_kb"])
    assert len(results) == 1


def test_stratified_sample_orders_by_page_and_chunk_index(fake_embedder, make_fake_store):
    store = make_fake_store("user_kb")
    store.add_texts(
        ["chunk b", "chunk a"],
        [{"page_number": 2, "chunk_index": 0}, {"page_number": 1, "chunk_index": 0}],
    )
    retriever = Retriever(stores={"user_kb": store}, embedder=fake_embedder)
    sample = retriever.stratified_sample("user_kb", n=5)
    assert sample[0]["text"] == "chunk a"


def test_stratified_sample_missing_store_returns_empty(fake_embedder):
    retriever = Retriever(stores={}, embedder=fake_embedder)
    assert retriever.stratified_sample("missing_kb") == []


def test_build_citation_with_list_pages():
    citation = _build_citation({"document_name": "Doc", "page_numbers": [3, 4]}, "guidelines_kb")
    assert "Page 3" in citation
    assert "Doc" in citation


def test_build_citation_unknown_document():
    citation = _build_citation({}, "guidelines_kb")
    assert "Unknown" in citation


def test_build_citation_falls_back_to_source_title():
    """
    Regression test: the drug-label JSONL export uses "source_title" for the
    document name, not "document_name"/"title"/etc — citations were silently
    showing "Document: Unknown" for every drug-label answer in production.
    """
    citation = _build_citation(
        {"source_title": "AMOXICILLIN AND CLAVULANATE POTASSIUM TABLET [NUCARE PHARMACEUTICALS,INC.]"},
        "druglabels_kb",
    )
    assert "AMOXICILLIN" in citation
    assert "Unknown" not in citation


def test_check_sufficiency_no_api_key_returns_bool_of_chunks(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert check_sufficiency("q", []) is False
    assert check_sufficiency("q", [{"text": "evidence"}]) is True


def test_check_sufficiency_uses_llm(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("app.retriever.llm.complete", lambda *a, **k: "NO")
    assert check_sufficiency("q", [{"text": "evidence"}]) is False

    monkeypatch.setattr("app.retriever.llm.complete", lambda *a, **k: "YES")
    assert check_sufficiency("q", [{"text": "evidence"}]) is True
