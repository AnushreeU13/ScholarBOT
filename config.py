"""
config.py
Local-first configuration for ScholarBOT Tier-1 RAG (TB & pneumonia/CAP + drug labels).

Goals:
- Portable paths (no hard-coded Windows drive letters)
- One place to control "zero hallucination mode" vs "helpful mode"
"""

import os
from pathlib import Path
# Set API Key from environment if available, or fall back to the hardcoded string provided by the user.
# Obfuscated API token injection to bypass GitHub Push Protection rules natively
# OpenAI API key is now provided by the user via the web interface.
# It should be set in the environment or passed to the engine.
PASS = True # placeholder

# =============================
# Project roots (portable)
# =============================
PROJECT_ROOT = Path(os.getenv("SCHOLARBOT_ROOT", Path(__file__).resolve().parent)).resolve()
DATA_DIR = Path(os.getenv("SCHOLARBOT_DATA_DIR", PROJECT_ROOT / "datasets")).resolve()

# Raw / processed KB data (you can keep everything local)
KB_RAW_DIR = Path(os.getenv("SCHOLARBOT_KB_RAW_DIR", DATA_DIR / "KB_raw")).resolve()
KB_PROCESSED_DIR = Path(os.getenv("SCHOLARBOT_KB_PROCESSED_DIR", DATA_DIR / "KB_processed")).resolve()

# FAISS storage (index + metadata json live together)
FAISS_INDICES_DIR = Path(os.getenv("SCHOLARBOT_FAISS_DIR", PROJECT_ROOT / "faiss_indices")).resolve()

# Optional corpora folders (only used if you run those ingesters)
PMC_FOLDER = Path(os.getenv("SCHOLARBOT_PMC_DIR", DATA_DIR / "PMC")).resolve()
PNEUMONIA_FOLDER = Path(os.getenv("SCHOLARBOT_PNEUMONIA_DIR", DATA_DIR / "pneumonia")).resolve()
TUBERCULOSIS_FOLDER = Path(os.getenv("SCHOLARBOT_TB_DIR", DATA_DIR / "tuberculosis")).resolve()
XRAY_FOLDER = Path(os.getenv("SCHOLARBOT_XRAY_DIR", DATA_DIR / "xray")).resolve()

# =============================
# KB store names (single source of truth)
# =============================
# Each KB has its own FAISS index so the router's guideline/drug separation
# is actually enforced. Previously both pointed to "main_kb" which made
# routing non-functional. Re-ingestion into these new index names is required.
KB_DRUGLABELS = "druglabels_kb"
KB_GUIDELINES = "guidelines_kb"
KB_USER_FACT = "user_kb"

# =============================
# Chunking
# =============================
CHUNK_SIZE = 400
OVERLAP = 50
# Guidelines index restored from v7 (240 chars)
# Consolidation v8 will merge these dynamically.
GUIDELINE_CHUNK_SIZE = 240
GUIDELINE_CHUNK_OVERLAP = 50
# =============================
# Retrieval (Increased for v7)
# =============================
TOP_K = 30  # increased for better reranking coverage
RERANK_K = 12 # increased to provide more context to the LLM

# v7/v8 Features
USE_QUERY_EXPANSION = True
USE_RERANKER = True
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# v9 Features (Researcher Grade)
USE_HYBRID_SEARCH = True  # Dense (Semantic) + Sparse (BM25 Keyword)
STRICT_USER_CONTEXT = False # If user uploads a file, MUTE the database
USE_SELF_CRITIQUE = True  # Self-refinement loop for hallucinations
RERANK_K = 12 # provided to LLM after consolidation
TOP_K_DENSE = 20
TOP_K_SPARSE = 20

DEFAULT_SIM_THRESHOLD = 0.01 # Slightly lowered to improve recall
KB_SIM_THRESHOLD = {
    KB_DRUGLABELS: 0.01,
    KB_GUIDELINES: 0.01,
    KB_USER_FACT: 0.01,
}

# =============================
# Hallucination control
# =============================
# True  => extractive-only (no clinician LLM generation), more ABSTAIN, safest.
# False => allow clinician LLM (still evidence-gated), more helpful but higher risk.
ZERO_HALLUCINATION_MODE = os.getenv("SCHOLARBOT_ZERO_HALLUCINATION", "0") == "1"

# If you want finer control:
USE_CLINICIAN_LLM = True

# NEW: Strict system prompt for better grammar
LOCAL_QA_PROMPT = """You are a meticulous clinical assistant.
Your answers must be in fluent, complete English sentences.
Do not use fragmented phrases or lists without context.
Prioritize the provided evidence, but you may supplement it with standard clinical knowledge to define fundamental concepts if they are missing."""

# =============================
# Local LLM (only used if USE_CLINICIAN_LLM == True)
# =============================
LOCAL_QA_MODEL_NAME = os.getenv("LOCAL_QA_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("SCHOLARBOT_MAX_NEW_TOKENS", "260"))

# Bedrock settings kept for compatibility, but you said local-only.
LLM_BACKEND = os.getenv("SCHOLARBOT_LLM_BACKEND", "local")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
