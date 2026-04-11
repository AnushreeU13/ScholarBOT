"""
01_config.py
Central configuration — single source of truth for all ScholarBOT settings.
Architecture position: 01 (loaded by everything else)
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path(os.getenv("SCHOLARBOT_ROOT", Path(__file__).resolve().parent)).resolve()
DATA_DIR          = Path(os.getenv("SCHOLARBOT_DATA_DIR",  PROJECT_ROOT / "dataset")).resolve()
KB_PROCESSED_DIR  = (DATA_DIR / "KB_processed").resolve()
FAISS_INDICES_DIR = Path(os.getenv("SCHOLARBOT_FAISS_DIR", PROJECT_ROOT / "faiss_indices")).resolve()

# ── KB store names (must match the actual faiss_indices/ subdirectory names) ──
KB_DRUGLABELS = "druglabels_kb"
KB_GUIDELINES = "guidelines_kb"
KB_USER       = "user_kb"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 400
OVERLAP    = 50

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_DENSE  = 20   # candidates from dense search per KB
TOP_K_SPARSE = 20   # candidates from BM25 per KB
RERANK_K     = 12   # top N kept after cross-encoder reranking
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Confidence gate — applied to the best reranker score after reranking.
# Below threshold → ABSTAIN. Raised from 0.01 (v12 fix).
KB_SIM_THRESHOLD = {
    KB_DRUGLABELS: 0.5,
    KB_GUIDELINES: 0.5,
    KB_USER:       0.3,   # lower for user uploads (smaller, noisier corpus)
}
DEFAULT_SIM_THRESHOLD = 0.5

# ── LLM ───────────────────────────────────────────────────────────────────────
OPENAI_MODEL  = os.getenv("OPENAI_MODEL",             "gpt-4o-mini")
ROUTER_MODEL  = os.getenv("SCHOLARBOT_ROUTER_MODEL",  "gpt-4o-mini")

# Max tokens per LLM call type
MAX_TOKENS_ANSWER      = 500   # clinician answer generation
MAX_TOKENS_PATIENT     = 400   # patient rewrite
MAX_TOKENS_ROUTER      = 250   # router JSON
MAX_TOKENS_SUFFICIENCY = 20    # YES/NO sufficiency check
MAX_TOKENS_CONTEXT     = 120   # coreference rewrite
MAX_TOKENS_SUMMARIZE   = 600   # document summarization

# ── Embedder ──────────────────────────────────────────────────────────────────
EMBED_MODEL      = "BAAI/bge-large-en-v1.5"
EMBED_DIM        = 1024
EMBED_MAX_LENGTH = 512
