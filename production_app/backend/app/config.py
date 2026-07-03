"""
Central configuration for the ScholarBOT production API.
All values are overridable via environment variables so the same image runs
unmodified in Docker Compose, Kubernetes, or Hugging Face Spaces.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", BACKEND_ROOT / "chroma_data")).resolve()

# ── Collections ───────────────────────────────────────────────────────────────
COLLECTION_GUIDELINES = "guidelines_kb"
COLLECTION_DRUGLABELS = "druglabels_kb"
COLLECTION_USER = "user_kb"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "20"))
RERANK_K = int(os.getenv("RERANK_K", "12"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

KB_SIM_THRESHOLD = {
    COLLECTION_GUIDELINES: 0.5,
    COLLECTION_DRUGLABELS: 0.5,
    COLLECTION_USER: 0.3,
}
DEFAULT_SIM_THRESHOLD = 0.5

# ── LLM ───────────────────────────────────────────────────────────────────────
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ROUTER_MODEL = os.getenv("SCHOLARBOT_ROUTER_MODEL", "gpt-4o-mini")

MAX_TOKENS_ANSWER = 500
MAX_TOKENS_ROUTER = 250
MAX_TOKENS_SUFFICIENCY = 20
MAX_TOKENS_CONTEXT = 120
MAX_TOKENS_SUMMARIZE = 600

# ── Embedder ──────────────────────────────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
EMBED_MAX_LENGTH = int(os.getenv("EMBED_MAX_LENGTH", "512"))

# ── CORS ──────────────────────────────────────────────────────────────────────
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
