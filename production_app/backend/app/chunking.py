"""Token-aware semantic chunking using a SciBERT tokenizer."""

from __future__ import annotations

import re
from typing import Dict, List

_TOKENIZER_CACHE: dict = {}


def _get_tokenizer(model_name: str):
    if model_name not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER_CACHE[model_name]


# Abbreviations whose trailing period must NOT trigger a sentence split.
_ABBREV_RE = re.compile(
    r"\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|approx|No|Fig|Tab|vol|al"
    r"|mg|mL|mcg|μg|kg|g|L|dL|mmHg|bpm|wk|mo|yr|min|sec|hrs?"
    r"|e\.g|i\.e|i\.v|p\.o|q\.d|b\.i\.d|t\.i\.d|q\.i\.d|p\.r\.n"
    r"|LTBI|TB|CAP|MDR|XDR|HIV|INH|RIF|PZA|EMB|BCG)\.",
    re.IGNORECASE,
)
_PLACEHOLDER = "\x00PERIOD\x00"


def _split_sentences(text: str) -> List[str]:
    protected = _ABBREV_RE.sub(lambda m: m.group(0)[:-1] + _PLACEHOLDER, text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", protected)
    return [p.replace(_PLACEHOLDER, ".").strip() for p in parts if p.strip()]


def semantic_chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 100,
    model_name: str = "allenai/scibert_scivocab_uncased",
) -> List[Dict]:
    """Split text into token-bounded chunks with sentence-aware overlap."""
    text = str(text or "").strip()
    if not text:
        return []

    tokenizer = _get_tokenizer(model_name)
    sentences = _split_sentences(text)

    chunks: List[Dict] = []
    current_sents: List[str] = []
    current_tokens = 0

    def _tok_count(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    for sent in sentences:
        sent_tokens = _tok_count(sent)

        if current_tokens + sent_tokens > chunk_size and current_sents:
            chunks.append({"text": " ".join(current_sents), "token_count": current_tokens})

            tail, tail_tokens = [], 0
            for s in reversed(current_sents):
                t = _tok_count(s)
                if tail and (tail_tokens + t) > overlap:
                    break
                tail.insert(0, s)
                tail_tokens += t
                if tail_tokens >= overlap:
                    break

            current_sents = tail or [current_sents[-1]]
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
    chunk_size: int = 512,
    overlap: int = 100,
) -> List[Dict]:
    """Chunk a document page and attach metadata to each chunk."""
    chunks = semantic_chunk_text(text, chunk_size, overlap)
    for idx, chunk in enumerate(chunks):
        chunk["metadata"] = {
            "document_name": document_name,
            "chunk_index": idx,
            "page_number": page_number,
            "total_chunks": len(chunks),
        }
    return chunks
