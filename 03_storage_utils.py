"""
03_storage_utils.py
FAISS index management + BM25 index build / save / load.
Architecture position: 03 — used by ingest (06) and retriever (09).

BM25 indices are saved as pickles alongside FAISS indices so they are built
once and reloaded on subsequent starts — no runtime rebuild from docstore.
"""

from __future__ import annotations

import importlib
import pickle
import re
from pathlib import Path
from typing import Optional, Tuple, List

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss as faiss_lib

_cfg = importlib.import_module("01_config")


# ── FAISS ─────────────────────────────────────────────────────────────────────

def get_or_create_faiss(store_name: str, dimension: int, embedder=None) -> FAISS:
    """
    Load an existing FAISS index from disk, or create a new empty one.
    Uses the shared embedder singleton if none is supplied.
    """
    if embedder is None:
        from importlib import import_module
        embedder = import_module("02_embedding_utils").get_embedder()

    path = Path(_cfg.FAISS_INDICES_DIR) / store_name

    if path.exists() and (path / "index.faiss").exists():
        print(f"[Storage] Loading FAISS: {store_name}")
        return FAISS.load_local(str(path), embedder, allow_dangerous_deserialization=True)

    print(f"[Storage] Creating empty FAISS: {store_name}")
    index = faiss_lib.IndexFlatIP(dimension)
    return FAISS(
        embedding_function=embedder,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )


# ── BM25 ──────────────────────────────────────────────────────────────────────

def _bm25_pkl_path(store_name: str) -> Path:
    return Path(_cfg.FAISS_INDICES_DIR) / f"{store_name}_bm25.pkl"


def build_bm25_from_store(faiss_store: FAISS) -> Tuple[object, List]:
    """Build a BM25Okapi index from a loaded FAISS store's docstore."""
    from rank_bm25 import BM25Okapi

    docs = list(faiss_store.docstore._dict.values())
    if not docs:
        return None, []

    texts     = [d.page_content for d in docs]
    tokenized = [re.findall(r"\w+", t.lower()) for t in texts]
    bm25      = BM25Okapi(tokenized)
    return bm25, docs


def save_bm25(store_name: str, bm25_obj, docs: List) -> None:
    pkl = _bm25_pkl_path(store_name)
    with open(pkl, "wb") as f:
        pickle.dump({"bm25": bm25_obj, "docs": docs}, f)
    print(f"[Storage] BM25 saved: {pkl.name}")


def load_or_build_bm25(store_name: str, faiss_store: FAISS, force_rebuild: bool = False):
    """
    Returns (bm25_obj, docs).
    - Loads from pickle if available and force_rebuild is False.
    - Otherwise builds from the FAISS docstore and saves the pickle for next time.
    """
    pkl = _bm25_pkl_path(store_name)

    if not force_rebuild and pkl.exists():
        print(f"[Storage] Loading BM25: {pkl.name}")
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        return data["bm25"], data["docs"]

    print(f"[Storage] Building BM25 for: {store_name}")
    bm25, docs = build_bm25_from_store(faiss_store)
    if bm25 is not None:
        save_bm25(store_name, bm25, docs)
    return bm25, docs


# Backward-compatibility alias used by old ingest scripts
def create_faiss_store(store_name: str, dimension: int, base_dir, embedder=None) -> FAISS:
    return get_or_create_faiss(store_name, dimension, embedder)
