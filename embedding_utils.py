
"""
embedding_utils.py

Compatibility Update (v3):
Switched to 'sentence-transformers/all-MiniLM-L6-v2' (384-dim) to matches the existing
FAISS indices migrated from the previous version.

Note: Class name 'MedCPTDualEmbedder' is kept for API compatibility, 
but it now uses a single symmetric encoder (MiniLM).
"""

from __future__ import annotations

from typing import Optional, List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

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


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling with attention mask (standard for SentenceTransformer models)."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class MedCPTDualEmbedder(Embeddings):
    """
    Wrapper that now uses all-MiniLM-L6-v2 (384d) for both query and doc.
    Preserves 'embed_query' and 'embed_texts' interface.
    """

    def __init__(
        self,
        query_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        doc_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.device = _get_device(device)
        self.max_length = int(max_length)

        # In this mode, query_model and doc_model are the SAME.
        print(f"[embedder] Loading Model: {query_model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(query_model_name)
        self.model = AutoModel.from_pretrained(query_model_name).to(self.device)
        self.model.eval()

        self.dim = int(self.model.config.hidden_size) # Should be 384
        self.name = query_model_name # Added for metadata tracking

    def _embed_internal(self, texts: List[str], max_length: int) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # MiniLM uses Mean Pooling
            vecs = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            
            # Normalize for Cosine Similarity
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)

        return vecs.cpu().numpy().astype(np.float32)

    # ----------------------------
    # Query embedding
    # ----------------------------
    def embed_query(self, query: str, max_length: Optional[int] = None) -> np.ndarray:
        """Embed a single query."""
        if not isinstance(query, str) or not query.strip():
            return np.zeros((self.dim,), dtype=np.float32)

        ml = int(max_length) if max_length is not None else self.max_length
        matrix = self._embed_internal([query], ml)
        return matrix[0]

    # ----------------------------
    # Document embedding (KB chunks)
    # ----------------------------
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of documents.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        bs = max(1, int(batch_size))
        ml = max(8, int(max_length))

        all_vecs = []
        n = len(texts)

        for i in range(0, n, bs):
            if show_progress and (i == 0 or i % (bs * 10) == 0):
                print(f"[embedder] Embedding batch {i//bs + 1}/{(n + bs - 1)//bs}")

            batch = texts[i : i + bs]
            vecs_batch = self._embed_internal(batch, ml)
            all_vecs.append(vecs_batch)

        return np.vstack(all_vecs)


    # Alias for LangChain compatibility
    embed_documents = embed_texts

# Backward-compatible alias
MedCPTEmbedder = MedCPTDualEmbedder
