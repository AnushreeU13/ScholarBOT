"""
Dense retrieval (Chroma) → cross-encoder rerank → evidence sufficiency gate.

BM25/RRF hybrid search was dropped in favor of dense-only + reranking after
IR evaluation on the ScholarBOT clinical corpus showed BM25 degraded ranking
quality for this domain (see eval/ir_eval.py in the parent repo).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from app import config, llm

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(config.RERANK_MODEL)
    return _reranker


def reset_reranker() -> None:
    """Test hook — clears the singleton so a fake can be injected."""
    global _reranker
    _reranker = None


def check_sufficiency(query: str, chunks: List[Dict]) -> bool:
    """
    Ask the LLM whether the retrieved evidence can answer the query.
    Fails open (True) when no API key is set or on any error, so retrieval
    never over-abstains purely because of an outage — the confidence
    threshold gate is the primary safety backstop.
    """
    import os

    if not os.getenv("OPENAI_API_KEY", "") or not chunks:
        return bool(chunks)

    evidence = "\n".join(f"[{i + 1}] {c['text'][:300]}" for i, c in enumerate(chunks[:6]))
    prompt = (
        f"Does the following evidence contain enough specific information "
        f"to answer the question below?\n\n"
        f"Question: {query}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Reply with exactly one word: YES or NO."
    )
    answer = llm.complete(
        "You are a strict evidence sufficiency checker.",
        prompt,
        config.MAX_TOKENS_SUFFICIENCY,
        config.OPENAI_MODEL,
    )
    if not answer:
        return True
    return "YES" in answer.upper()


def _build_citation(meta: Dict, store_name: str) -> str:
    kb_labels = {
        config.COLLECTION_USER: "User Uploaded",
        config.COLLECTION_DRUGLABELS: "Existing KB - Drug Labels",
        config.COLLECTION_GUIDELINES: "Existing KB - Guidelines",
    }
    kb = kb_labels.get(store_name, "KB")
    # source_title is what the drug-label JSONL export actually uses; the
    # rest cover the guidelines export and user-upload metadata shapes.
    doc = (meta.get("document_name") or meta.get("document") or
           meta.get("title") or meta.get("file_name") or
           meta.get("source_title") or "Unknown")
    pages = meta.get("page_numbers") or meta.get("page_number")
    page_str = ""
    if isinstance(pages, list) and pages:
        page_str = f", Page {pages[0]}"
    elif pages is not None:
        page_str = f", Page {pages}"
    return f"KB: {kb}, Document: {doc}{page_str}"


class Retriever:
    """Retrieves and reranks candidates across one or more named Chroma collections."""

    def __init__(self, stores: Dict[str, object], embedder):
        """
        Args:
            stores  : {collection_name: ChromaStore} — shared, read-only reference
                      collections (guidelines, drug labels). Session-specific
                      collections (the uploaded-document store) are never held
                      here; they're passed in per-call via `session_stores` so
                      concurrent requests for different sessions can't race on
                      shared instance state.
            embedder: BGEEmbedder instance
        """
        self.stores = stores
        self.embedder = embedder

    def _resolve_store(self, collection_name: str, session_stores: Optional[Dict[str, object]]):
        if session_stores and collection_name in session_stores:
            return session_stores[collection_name]
        return self.stores.get(collection_name)

    def _dense_search(
        self, query: str, collection_name: str, k: int, session_stores: Optional[Dict[str, object]]
    ) -> List[Dict]:
        store = self._resolve_store(collection_name, session_stores)
        if store is None:
            return []
        q_vec = self.embedder.embed_query(query)
        hits = store.similarity_search_by_vector(q_vec, k=k)
        for h in hits:
            h["store"] = collection_name
        return hits

    def retrieve(
        self,
        query: str,
        target_kbs: List[str],
        k_dense: Optional[int] = None,
        k_rerank: Optional[int] = None,
        session_stores: Optional[Dict[str, object]] = None,
    ) -> List[Dict]:
        k_dense = k_dense or config.TOP_K_DENSE
        k_rerank = k_rerank or config.RERANK_K

        candidates: List[Dict] = []
        for kb in target_kbs:
            candidates.extend(self._dense_search(query, kb, k_dense, session_stores))

        seen, unique = set(), []
        for c in candidates:
            key = re.sub(r"\W+", "", c["text"][:100]).lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(c)

        unique.sort(key=lambda x: x["score"], reverse=True)
        top = unique[:k_dense]

        if top:
            try:
                reranker = _get_reranker()
                pairs = [[query, c["text"]] for c in top]
                scores = reranker.predict(pairs)
                for c, s in zip(top, scores):
                    c["score"] = float(s)
                top.sort(key=lambda x: x["score"], reverse=True)
                top = top[:k_rerank]
            except Exception:
                top = top[:k_rerank]

        for i, c in enumerate(top):
            c["chunk_id"] = i + 1
            c["citation"] = _build_citation(c.get("metadata", {}), c.get("store", ""))

        return top

    def stratified_sample(
        self, collection_name: str, n: int = 16, session_stores: Optional[Dict[str, object]] = None
    ) -> List[Dict]:
        """For summarization: n chunks sampled evenly across pages, no query embedding used."""
        store = self._resolve_store(collection_name, session_stores)
        if store is None:
            return []

        all_docs = store.all_documents()
        if not all_docs:
            return []

        def _sort_key(d):
            m = d.get("metadata", {})
            return (int(m.get("page_number", 0) or 0), int(m.get("chunk_index", 0) or 0))

        all_docs.sort(key=_sort_key)

        total = len(all_docs)
        step = max(1, total // n)
        sampled = all_docs[::step][:n]

        result = []
        for i, doc in enumerate(sampled):
            meta = doc.get("metadata", {})
            result.append({
                "chunk_id": i + 1,
                "text": doc["text"],
                "metadata": meta,
                "store": collection_name,
                "citation": _build_citation(meta, collection_name),
                "score": 1.0,
            })
        return result
