"""BGE-large-en-v1.5 (1024-dim) embedder with a lazily-loaded module singleton."""

from __future__ import annotations

from typing import List, Optional

from app import config


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class BGEEmbedder:
    """
    BAAI/bge-large-en-v1.5 wrapper.
    - 1024-dim output
    - Query-side instruction prefix (required for BGE retrieval quality)
    - Document-side: no prefix
    """

    _QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None,
                 max_length: Optional[int] = None):
        from sentence_transformers import SentenceTransformer

        self.name = model_name or config.EMBED_MODEL
        self.dim = config.EMBED_DIM
        self.device = device or _device()

        self.model = SentenceTransformer(self.name, device=self.device)
        self.model.max_seq_length = max_length or config.EMBED_MAX_LENGTH

    def embed_query(self, text: str) -> List[float]:
        if not isinstance(text, str) or not text.strip():
            return [0.0] * self.dim
        vec = self.model.encode(
            self._QUERY_PREFIX + text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vec.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
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


_instance: Optional[BGEEmbedder] = None


def get_embedder() -> BGEEmbedder:
    global _instance
    if _instance is None:
        _instance = BGEEmbedder()
    return _instance


def reset_embedder() -> None:
    """Test hook — clears the singleton so a fake can be injected."""
    global _instance
    _instance = None
