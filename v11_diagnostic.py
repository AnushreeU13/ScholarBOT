
import sys, os
from pathlib import Path
import faiss
import numpy as np

# Diagnostic for v11
PROJECT_ROOT = Path(__file__).resolve().parent
FAISS_DIR = PROJECT_ROOT / "faiss_indices"
DATASET_DIR = PROJECT_ROOT / "dataset"

print(f"--- Diagnosing ScholarBOT v11 ---")
print(f"Project Root: {PROJECT_ROOT}")
print(f"FAISS Dir: {FAISS_DIR}")
print(f"Dataset Dir: {DATASET_DIR}")

# 1. Check Index Dimensions
for kb in ["guidelines_kb", "druglabels_kb"]:
    idx_p = FAISS_DIR / kb / "index.faiss"
    if idx_p.exists():
        idx = faiss.read_index(str(idx_p))
        print(f"KB: {kb} | ntotal: {idx.ntotal} | d (dim): {idx.d} | metric: {idx.metric_type}")
    else:
        print(f"KB: {kb} | NOT FOUND at {idx_p}")

# 2. Test Embedding vs Search
try:
    from embedding_utils import MedCPTDualEmbedder
    embedder = MedCPTDualEmbedder()
    print(f"Embedder d: {embedder.dim} | name: {embedder.name}")
    
    q = "What is TB, how is it diagnosed?"
    vec = embedder.embed_query(q)
    print(f"Encoded Query Vector: (shape: {vec.shape}, first 5: {vec[:5]})")
    
    # Try raw search on one index
    idx_p = FAISS_DIR / "guidelines_kb" / "index.faiss"
    if idx_p.exists():
        idx = faiss.read_index(str(idx_p))
        scores, ids = idx.search(vec.reshape(1, -1).astype('float32'), 5)
        print(f"Raw FAISS Similarity Scores: {scores}")
        print(f"Found IDs: {ids}")
except Exception as e:
    print(f"Error during diagnostic: {e}")
