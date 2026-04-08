from __future__ import annotations

from typing import Optional, List, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    try:
        from langchain.schema.embeddings import Embeddings
    except ImportError:
        from langchain.embeddings.base import Embeddings


def _get_device(device: Optional[str] = None) -> str:
    """Return an available torch device string."""
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class MedCPTDualEmbedder(Embeddings):
    """
    Wrapper that now uses BAAI/bge-large-en-v1.5 (1024-dim) for both query and doc.
    Uses SentenceTransformer to ensure correct pooling and normalization.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        max_length: int = 512,
        **kwargs
    ):
        self.device = _get_device(device)
        self.max_length = int(max_length)
        self.name = model_name
        
        # BGE models perform best with specific query instruction
        self.query_instruction = "Represent this sentence for searching relevant passages: "

        print(f"[v11 embedder] Loading Large Model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()
        
        if self.dim != 1024:
            print(f"[WARN] Unexpected dimension {self.dim} for model {model_name} (Expected 1024).")

    def _embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        processed_texts = texts
        if is_query:
            processed_texts = [self.query_instruction + t for t in texts]

        # SentenceTransformer.encode handles batching, pooling, and normalization
        vecs = self.model.encode(
            processed_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return vecs.astype(np.float32)

    # ----------------------------
    # Query embedding
    # ----------------------------
    def embed_query(self, query: str, **kwargs) -> List[float]:
        """Embed a single query with instruction prefix."""
        if not isinstance(query, str) or not query.strip():
            return [0.0] * self.dim

        vec = self._embed([query], is_query=True)
        return vec[0].tolist()

    # ----------------------------
    # Document embedding (KB chunks)
    # ----------------------------
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """
        Embed a list of documents without prefix.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        return self._embed(texts, is_query=False)

    # ----------------------------
    # Aliases for LangChain compatibility
    # ----------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents and return as list of lists."""
        vecs = self.embed_texts(texts)
        return vecs.tolist()


# Backward-compatible alias
MedCPTEmbedder = MedCPTDualEmbedder

