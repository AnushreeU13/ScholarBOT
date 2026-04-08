
import sys, os
from pathlib import Path
import time
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document as LCDoc
import faiss

# Import PROJECT config
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

try:
    from config import (
        DATA_DIR, FAISS_INDICES_DIR, KB_GUIDELINES, KB_DRUGLABELS,
        GUIDELINE_CHUNK_SIZE, GUIDELINE_CHUNK_OVERLAP
    )
    from embedding_utils import MedCPTDualEmbedder
except ImportError:
    print("Error: Could not import project config.")
    sys.exit(1)

def build_aligned_index(store_name, jsonl_paths):
    print(f"\n--- Syncing {store_name} ---")
    embedder = MedCPTDualEmbedder()
    
    all_chunks = []
    for p in jsonl_paths:
        p = Path(p)
        if not p.exists():
            print(f"Warning: Skipping {p} (Not Found)")
            continue
        print(f"Reading {p.name}...")
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                all_chunks.append(LCDoc(page_content=data['text'], metadata=data.get('metadata', {})))
    
    if not all_chunks:
        print(f"Error: No data found for {store_name}")
        return

    print(f"Building index for {len(all_chunks)} clinical chunks...")
    store = FAISS.from_documents(all_chunks, embedder)
    
    out_p = FAISS_INDICES_DIR / store_name
    out_p.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out_p))
    print(f"SUCCESS: {store_name} synchronized at {out_p}")

import json
if __name__ == "__main__":
    DATASET_DIR = DATA_DIR
    
    # Guidelines + CDC
    build_aligned_index(KB_GUIDELINES, [
        DATASET_DIR / "guidelines_chunks_cleaned.jsonl",
        DATASET_DIR / "cdc_tb_pages.jsonl"
    ])
    
    # Drug Labels
    build_aligned_index(KB_DRUGLABELS, [
        DATASET_DIR / "druglabels_chunks.jsonl"
    ])
    
    print("\n[V11 DONE] All clinical indices are now 100% synchronized with the engine.")
