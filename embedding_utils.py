"""
embedding_utils.py

Compatibility Update (v11/v12):
Switched to 'BAAI/bge-large-en-v1.5' (1024-dim) via SentenceTransformer.
FAISS indices migrated to v11/v12 standards.

Note: Class name 'MedCPTDualEmbedder' is kept for API compatibility.
"""

from __future__ import annotations
import os
from typing import Optional, List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from langchain_core.embeddings import Embeddings

def _get_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"

class MedCPTDualEmbedder(Embeddings):
    """
    Wrapper that uses BAAI/bge-large-en-v1.5 (1024d) via SentenceTransformer.
    Ensures 100% architectural alignment with v11/v12 indices.
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
        
        print(f"[v11 embedder] Loading Large Model: {model_name} on {self.device}")
        
        # Use SentenceTransformer for optimized pooling and normalization
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = self.max_length
        
        # BGE requires a specific instruction for queries
        self.query_instruction = "Represent this sentence for searching relevant passages: "
        
        self.dim = 1024 # bge-large is 1024
        self.name = model_name

    def embed_query(self, query: str, **kwargs) -> List[float]:
        """Embed a single query with BGE prefix."""
        if not isinstance(query, str) or not query.strip():
            return [0.0] * self.dim

        # Prefix is mandatory for BGE query-side retrieval quality
        full_query = self.query_instruction + query
        
        embedding = self.model.encode(
            full_query, 
            normalize_embeddings=True, 
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embedding.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 32, **kwargs) -> List[List[float]]:
        """Embed a list of documents (KB chunks). No prefix needed for docs."""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    # Alias for LangChain compatibility
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)

# Backward-compatible alias
MedCPTEmbedder = MedCPTDualEmbedder
