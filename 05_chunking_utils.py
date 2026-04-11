"""
05_chunking_utils.py
Token-aware semantic chunking using SciBERT tokenizer.
Architecture position: 05 — used by user ingest (06).

Key fix vs old version: tokenizer is cached at module level,
not reloaded on every call (was loading SciBERT once per page).
"""

import re
from typing import List, Dict

from transformers import AutoTokenizer

# ── Tokenizer singleton ───────────────────────────────────────────────────────

_TOKENIZER_CACHE: dict = {}


def _get_tokenizer(model_name: str):
    if model_name not in _TOKENIZER_CACHE:
        print(f"[Chunker] Loading tokenizer: {model_name}")
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER_CACHE[model_name]


# ── Core chunker ──────────────────────────────────────────────────────────────

def semantic_chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
    model_name: str = "allenai/scibert_scivocab_uncased",
) -> List[Dict]:
    """
    Split text into token-bounded chunks with sentence-aware overlap.
    Returns list of {"text": str, "token_count": int}.
    """
    text = str(text or "").strip()
    if not text:
        return []

    tokenizer = _get_tokenizer(model_name)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    chunks: List[Dict] = []
    current_sents: List[str] = []
    current_tokens: int = 0

    def _tok_count(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    for sent in sentences:
        sent_tokens = _tok_count(sent)

        if current_tokens + sent_tokens > chunk_size and current_sents:
            chunks.append({"text": " ".join(current_sents), "token_count": current_tokens})

            # Build overlap by walking backward through current sentences
            tail, tail_tokens = [], 0
            for s in reversed(current_sents):
                t = _tok_count(s)
                if tail and (tail_tokens + t) > overlap:
                    break
                tail.insert(0, s)
                tail_tokens += t
                if tail_tokens >= overlap:
                    break

            current_sents  = tail or [current_sents[-1]]
            current_tokens = _tok_count(" ".join(current_sents))

        current_sents.append(sent)
        current_tokens += sent_tokens

    if current_sents:
        chunks.append({"text": " ".join(current_sents), "token_count": current_tokens})

    return chunks


def chunk_document(
    text: str,
    document_name: str,
    page_number: int = None,
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[Dict]:
    """
    Chunk a document page and attach metadata to each chunk.
    Returns list of {"text": str, "token_count": int, "metadata": dict}.
    """
    chunks = semantic_chunk_text(text, chunk_size, overlap)
    for idx, chunk in enumerate(chunks):
        chunk["metadata"] = {
            "document_name": document_name,
            "chunk_index":   idx,
            "page_number":   page_number,
            "total_chunks":  len(chunks),
        }
    return chunks
