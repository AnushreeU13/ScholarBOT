"""
02_embedding_utils.py
BGE-large-en-v1.5 (1024-dim) embedder with module-level singleton.
Architecture position: 02 — loaded by storage, retriever, and ingest.
"""

from __future__ import annotations

import importlib
from typing import Optional, List

import torch
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

_cfg = importlib.import_module("01_config")


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class BGEEmbedder(Embeddings):
    """
    BAAI/bge-large-en-v1.5 wrapper.
    - 1024-dim output
    - Query-side instruction prefix (required for BGE retrieval quality)
    - Document-side: no prefix
    """

    _QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = None, device: Optional[str] = None, max_length: int = None):
        model_name = model_name or _cfg.EMBED_MODEL
        max_length  = max_length  or _cfg.EMBED_MAX_LENGTH

        self.name   = model_name
        self.dim    = _cfg.EMBED_DIM
        self.device = device or _device()

        print(f"[Embedder] Loading {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_length

    # ── Query embedding (with prefix) ─────────────────────────────────────────
    def embed_query(self, text: str, **kwargs) -> List[float]:
        if not isinstance(text, str) or not text.strip():
            return [0.0] * self.dim
        vec = self.model.encode(
            self._QUERY_PREFIX + text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vec.tolist()

    # ── Document embedding (no prefix) ────────────────────────────────────────
    def embed_texts(self, texts: List[str], batch_size: int = 32, **kwargs) -> List[List[float]]:
        if not texts:
            return []
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    # LangChain Embeddings interface
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)


# ── Module-level singleton ─────────────────────────────────────────────────────
# The BGE model is ~1.3 GB. Load once per process, reuse everywhere.
_instance: BGEEmbedder | None = None


def get_embedder() -> BGEEmbedder:
    global _instance
    if _instance is None:
        _instance = BGEEmbedder()
    return _instance


# Backward-compatibility alias (used by old ingest scripts that still import this)
MedCPTDualEmbedder = BGEEmbedder
