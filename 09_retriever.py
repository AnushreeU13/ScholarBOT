"""
09_retriever.py
Hybrid retrieval: Dense (BGE) + Sparse (BM25) → RRF merge → Cross-encoder rerank.
Includes evidence sufficiency check (LLM binary YES/NO) before returning to pipeline.
Architecture position: 09 — sits between router (08) and pipeline (10).

Key improvements vs old rag_pipeline_aligned.py:
- BM25 loaded from saved pickle (built by 06_ingest_user or on first access)
- Retrieval is fully separated from generation logic
- Evidence sufficiency gate: after reranking, asks LLM if chunks can answer the query
- Chunk IDs embedded in each result for direct citation (no Jaccard alignment)
"""

from __future__ import annotations

import importlib
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

_cfg     = importlib.import_module("01_config")
_storage = importlib.import_module("03_storage_utils")

# ── Cross-encoder singleton ───────────────────────────────────────────────────

_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"[Retriever] Loading reranker: {_cfg.RERANK_MODEL}")
        _reranker = CrossEncoder(_cfg.RERANK_MODEL)
    return _reranker


# ── RRF merge ─────────────────────────────────────────────────────────────────

def _rrf_merge(dense: List[Dict], sparse: List[Dict], k: int = 60) -> List[Dict]:
    """
    Reciprocal Rank Fusion of dense and sparse result lists.
    score(d) = Σ_rankers [ 1 / (k + rank(d)) ]
    """
    scores: Dict[str, float] = {}
    by_key: Dict[str, Dict]  = {}

    def _key(c: Dict) -> str:
        return re.sub(r"\W+", "", (c.get("text") or "")[:120].lower())

    for rank, c in enumerate(dense, 1):
        key = _key(c)
        if key:
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            by_key.setdefault(key, c)

    for rank, c in enumerate(sparse, 1):
        key = _key(c)
        if key:
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            by_key.setdefault(key, c)

    merged = []
    for key, score in scores.items():
        entry = by_key[key].copy()
        entry["score"] = score
        merged.append(entry)

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged


# ── Sufficiency check ─────────────────────────────────────────────────────────

def _check_sufficiency(query: str, chunks: List[Dict]) -> bool:
    """
    Ask the LLM: does the retrieved evidence contain enough information to
    answer this specific question? Returns True (sufficient) or False (abstain).
    Falls back to True on any error so we don't over-abstain.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or not chunks:
        return bool(chunks)

    evidence = "\n".join(
        f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(chunks[:6])
    )

    prompt = (
        f"Does the following evidence contain enough specific information "
        f"to answer the question below?\n\n"
        f"Question: {query}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Reply with exactly one word: YES or NO."
    )

    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=_cfg.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=_cfg.MAX_TOKENS_SUFFICIENCY,
        )
        answer = (resp.choices[0].message.content or "").strip().upper()
        return "YES" in answer
    except Exception:
        return True   # fail open on error — threshold gate still applies


# ── Citation builder ──────────────────────────────────────────────────────────

def _build_citation(meta: Dict, store_name: str) -> str:
    """Stable citation string: KB label + document name + page number."""
    kb_labels = {
        _cfg.KB_USER:       "User Uploaded",
        _cfg.KB_DRUGLABELS: "Existing KB - Drug Labels",
        _cfg.KB_GUIDELINES: "Existing KB - Guidelines",
    }
    kb   = kb_labels.get(store_name, "KB")
    doc  = (meta.get("document_name") or meta.get("document") or
            meta.get("title") or meta.get("file_name") or "Unknown")
    pages = meta.get("page_numbers") or meta.get("page_number")
    page_str = ""
    if isinstance(pages, list) and pages:
        page_str = f", Page {pages[0]}"
    elif pages is not None:
        page_str = f", Page {pages}"
    return f"KB: {kb}, Document: {doc}{page_str}"


# ── HybridRetriever ───────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Manages dense + BM25 retrieval across multiple FAISS stores.
    BM25 indices are loaded from disk on first access and cached in memory.
    """

    def __init__(self, stores: Dict[str, object], embedder):
        """
        Args:
            stores  : {store_name: FAISS store object}
            embedder: BGEEmbedder instance
        """
        self.stores   = stores
        self.embedder = embedder
        self._bm25: Dict[str, Tuple] = {}  # store_name -> (bm25_obj, docs)

    def reload_store(self, store_name: str, store) -> None:
        """Hot-reload a store (e.g. after user PDF ingestion)."""
        self.stores[store_name] = store
        self._bm25.pop(store_name, None)   # force BM25 rebuild on next access

    def _get_bm25(self, store_name: str):
        if store_name not in self._bm25:
            store = self.stores.get(store_name)
            if store is None:
                return None, []
            bm25, docs = _storage.load_or_build_bm25(store_name, store)
            self._bm25[store_name] = (bm25, docs)
        return self._bm25[store_name]

    # ── Dense search ──────────────────────────────────────────────────────────

    def _dense_search(self, query: str, store_name: str, k: int) -> List[Dict]:
        store = self.stores.get(store_name)
        if store is None:
            return []
        try:
            q_vec   = self.embedder.embed_query(query)
            results = store.similarity_search_with_score_by_vector(q_vec, k=k)
        except Exception as e:
            print(f"[Retriever] Dense search failed for {store_name}: {e}")
            return []

        hits = []
        for doc, score in results:
            # Detect L2 metric and convert to similarity
            is_l2 = (hasattr(store, "index") and
                     hasattr(store.index, "metric_type") and
                     store.index.metric_type == 1)
            sim = (1.0 - score / 2.0) if is_l2 else float(score)
            hits.append({
                "score":    sim,
                "raw_sim":  sim,
                "text":     doc.page_content,
                "metadata": doc.metadata,
                "store":    store_name,
                "type":     "dense",
            })
        return hits

    # ── BM25 search ───────────────────────────────────────────────────────────

    def _bm25_search(self, query: str, store_name: str, k: int) -> List[Dict]:
        bm25, docs = self._get_bm25(store_name)
        if bm25 is None or not docs:
            return []

        tokens = re.findall(r"\w+", query.lower())
        raw_scores = bm25.get_scores(tokens)
        max_score  = max(raw_scores) if max(raw_scores) > 0 else 1.0
        top_idx    = np.argsort(raw_scores)[::-1][:k]

        hits = []
        for i in top_idx:
            if raw_scores[i] <= 0:
                continue
            doc = docs[i]
            norm = float(raw_scores[i]) / max_score
            hits.append({
                "score":    norm,
                "raw_sim":  norm,
                "text":     doc.page_content,
                "metadata": doc.metadata,
                "store":    store_name,
                "type":     "sparse",
            })
        return hits

    # ── Public retrieve ───────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        target_kbs: List[str],
        k_dense:  int = None,
        k_sparse: int = None,
        k_rerank: int = None,
    ) -> List[Dict]:
        """
        Full hybrid retrieval pipeline for a single query.
        Returns up to k_rerank chunks, each with a unique 'chunk_id' field.
        """
        k_dense  = k_dense  or _cfg.TOP_K_DENSE
        k_sparse = k_sparse or _cfg.TOP_K_SPARSE
        k_rerank = k_rerank or _cfg.RERANK_K

        candidates: List[Dict] = []

        for kb in target_kbs:
            dense  = self._dense_search(query, kb, k_dense)
            sparse = self._bm25_search(query, kb, k_sparse)
            merged = _rrf_merge(dense, sparse) if sparse else dense
            candidates.extend(merged)

        # Deduplicate by text prefix
        seen, unique = set(), []
        for c in candidates:
            key = re.sub(r"\W+", "", c["text"][:100]).lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(c)

        unique.sort(key=lambda x: x["score"], reverse=True)
        top = unique[:_cfg.TOP_K_DENSE]   # cap before reranking

        # Cross-encoder reranking
        if top:
            try:
                reranker = _get_reranker()
                pairs  = [[query, c["text"]] for c in top]
                scores = reranker.predict(pairs)
                for c, s in zip(top, scores):
                    c["score"] = float(s)
                top.sort(key=lambda x: x["score"], reverse=True)
                top = top[:k_rerank]
            except Exception as e:
                print(f"[Retriever] Reranking failed: {e}")
                top = top[:k_rerank]

        # Assign stable chunk IDs and build citations
        for i, c in enumerate(top):
            c["chunk_id"]  = i + 1          # [1], [2], … used in LLM prompts
            c["citation"]  = _build_citation(c.get("metadata", {}), c.get("store", ""))

        return top

    def stratified_sample(self, store_name: str, n: int = 16) -> List[Dict]:
        """
        For summarization: return n chunks sampled evenly across all pages
        of the given store. Does not use the query embedding.
        """
        store = self.stores.get(store_name)
        if store is None:
            return []

        try:
            all_docs = list(store.docstore._dict.values())
        except Exception:
            return []

        if not all_docs:
            return []

        # Sort by page number then chunk index
        def _sort_key(doc):
            m = doc.metadata if hasattr(doc, "metadata") else {}
            return (int(m.get("page_number", 0) or 0),
                    int(m.get("chunk_index", 0) or 0))

        all_docs.sort(key=_sort_key)

        # Evenly spaced sample
        total = len(all_docs)
        step  = max(1, total // n)
        sampled = all_docs[::step][:n]

        result = []
        for i, doc in enumerate(sampled):
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            result.append({
                "chunk_id": i + 1,
                "text":     doc.page_content,
                "metadata": meta,
                "store":    store_name,
                "citation": _build_citation(meta, store_name),
                "score":    1.0,
            })
        return result
