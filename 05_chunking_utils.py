"""
05_chunking_utils.py
Token-aware semantic chunking using SciBERT tokenizer.
Architecture position: 05 — used by user ingest (06).

Key fix vs old version: tokenizer is cached at module level,
not reloaded on every call (was loading SciBERT once per page).

v13 fix: sentence splitter now protects medical abbreviations (mg., e.g., i.v.,
Dr., etc.) so they do not cause false sentence breaks. Overlap raised to 100
tokens (was 50) to preserve cross-boundary clinical context.
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


# ── Sentence splitter with abbreviation protection ────────────────────────────

# Abbreviations whose trailing period must NOT trigger a sentence split.
# Covers clinical units, Latin phrases, titles, and common medical shorthands.
_ABBREV_RE = re.compile(
    r"\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|approx|No|Fig|Tab|vol|al"
    r"|mg|mL|mcg|μg|kg|g|L|dL|mmHg|bpm|wk|mo|yr|min|sec|hrs?"
    r"|e\.g|i\.e|i\.v|p\.o|q\.d|b\.i\.d|t\.i\.d|q\.i\.d|p\.r\.n"
    r"|LTBI|TB|CAP|MDR|XDR|HIV|INH|RIF|PZA|EMB|BCG)\.",
    re.IGNORECASE,
)
_PLACEHOLDER = "\x00PERIOD\x00"


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences, preserving medical abbreviations.
    Only splits on punctuation followed by whitespace + capital letter,
    which is the standard sentence boundary pattern.
    """
    # Temporarily mask abbreviation periods so they survive the split
    protected = _ABBREV_RE.sub(lambda m: m.group(0)[:-1] + _PLACEHOLDER, text)
    # Split at sentence-ending punctuation followed by whitespace + capital
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", protected)
    # Restore masked periods and strip
    return [p.replace(_PLACEHOLDER, ".").strip() for p in parts if p.strip()]


# ── Core chunker ──────────────────────────────────────────────────────────────

def semantic_chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 100,
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
    sentences = _split_sentences(text)

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
    chunk_size: int = 512,
    overlap: int = 100,
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
