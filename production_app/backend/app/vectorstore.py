"""
Chroma-backed vector store.

Chroma was chosen over Pinecone for this rebuild — see README for the
documented tradeoff. This module wraps a persistent `chromadb.PersistentClient`
so each named collection (guidelines_kb, druglabels_kb, user_kb) behaves like
an independent index, mirroring the old FAISS-per-KB layout.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from app import config


class ChromaStore:
    """Thin wrapper around one Chroma collection."""

    def __init__(self, collection_name: str, client=None, embedder=None):
        self.collection_name = collection_name
        self.embedder = embedder
        self._client = client or _get_client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_texts(self, texts: List[str], metadatas: List[Dict], ids: Optional[List[str]] = None) -> List[str]:
        if not texts:
            return []
        if ids is None:
            start = self.count()
            ids = [f"{self.collection_name}-{start + i}" for i in range(len(texts))]

        embeddings = self.embedder.embed_texts(texts)
        # Chroma metadata values must be str/int/float/bool — coerce anything else.
        clean_meta = [_sanitize_metadata(m) for m in metadatas]
        self._collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=clean_meta)
        return ids

    def delete_all(self) -> None:
        existing = self._collection.get(include=[])
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)

    # ── Query ─────────────────────────────────────────────────────────────────

    def similarity_search_by_vector(self, query_vector: List[float], k: int = 20) -> List[Dict]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            # Chroma's cosine space returns distance = 1 - cosine_similarity.
            similarity = 1.0 - float(dist)
            hits.append({"text": doc, "metadata": meta or {}, "score": similarity})
        return hits

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0

    def all_documents(self) -> List[Dict]:
        result = self._collection.get(include=["documents", "metadatas"])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        return [{"text": d, "metadata": m or {}} for d, m in zip(docs, metas)]


def _sanitize_metadata(meta: Dict) -> Dict:
    clean = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    if not clean:
        # Chroma rejects a fully empty metadata dict on add().
        clean["_empty"] = True
    return clean


_client_instance = None
_client_lock = threading.Lock()


def _get_client():
    """
    Thread-safe singleton. Two threads racing to construct a PersistentClient
    for the same path at once corrupts Chroma's shared-system registry
    (surfaces as AttributeError/KeyError deep in chromadb internals) — this
    lock is what prevents that, on top of the app-level eager init in
    app.main's lifespan hook.
    """
    global _client_instance
    if _client_instance is None:
        with _client_lock:
            if _client_instance is None:
                import chromadb

                config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
                _client_instance = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client_instance


def reset_client() -> None:
    """Test hook — clears the singleton so an in-memory client can be injected."""
    global _client_instance
    _client_instance = None


_stores: Dict[str, ChromaStore] = {}
_stores_lock = threading.Lock()


def get_store(collection_name: str, embedder=None) -> ChromaStore:
    if collection_name not in _stores:
        with _stores_lock:
            if collection_name not in _stores:
                _stores[collection_name] = ChromaStore(collection_name, embedder=embedder)
    return _stores[collection_name]


def reset_stores() -> None:
    global _stores
    _stores = {}
