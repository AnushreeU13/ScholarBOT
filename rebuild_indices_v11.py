
import json, sys, time
import numpy as np
from pathlib import Path

# V11 Standards: 1024 dimensions, Reading from 'dataset/'
FAISS_DIR = Path(__file__).resolve().parent / "faiss_indices"
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBED_DIM = 1024

# Files
GUIDELINES_CLEANED = DATASET_DIR / "guidelines_chunks_cleaned.jsonl"
CDC_TB_JSONL = DATASET_DIR / "cdc_tb_pages.jsonl"
DRUGLABELS_JSONL = DATASET_DIR / "druglabels_chunks.jsonl"

print(f"--- ScholarBOT v11 Index Builder ---")
print(f"Model: {MODEL_NAME} ({EMBED_DIM} dims)")

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
import faiss as faiss_lib
from embedding_utils import MedCPTDualEmbedder

# Initialize
st_model = SentenceTransformer(MODEL_NAME)
embedder = MedCPTDualEmbedder()

def load_jsonl(path):
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: records.append(json.loads(line))
                except: pass
    return records

def build_v11_index(store_name, records):
    print(f"\n=== Building: {store_name} ({len(records)} records) ===")
    if not records:
        print(f"  Empty records for {store_name}, skipping.")
        return

    docs, texts = [], []
    for r in records:
        t = r.get("text") or r.get("chunk_text") or ""
        if t.strip():
            meta = {k:v for k,v in r.items() if k not in ("text","chunk_text")}
            docs.append(Document(page_content=t.strip(), metadata=meta))
            texts.append(t.strip())
    
    print(f"  Non-empty: {len(texts)}")
    
    # Embed in batches
    t0 = time.time()
    all_vecs = []
    batch_size = 32 # Large model needs smaller batches on CPU/GPU
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vecs = st_model.encode(batch, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        all_vecs.append(vecs)
        print(f"  {min(i+batch_size,len(texts))}/{len(texts)} ({time.time()-t0:.1f}s)", end="\r")
    print()
    vectors = np.vstack(all_vecs).astype("float32")
    
    # Build FAISS
    index = faiss_lib.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    
    docstore = InMemoryDocstore({str(i): docs[i] for i in range(len(docs))})
    index_to_docstore_id = {i: str(i) for i in range(len(docs))}
    
    store = FAISS(
        embedding_function=embedder,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    
    out_dir = FAISS_DIR / store_name
    out_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out_dir))
    print(f"  Saved: faiss_indices/{store_name}/ (ntotal={index.ntotal})")

def main():
    # 1. Guidelines KB (Merge cleaned chunks + CDC pages)
    print("Loading Guidelines & CDC Data...")
    g_records = load_jsonl(GUIDELINES_CLEANED)
    cdc_records = load_jsonl(CDC_TB_JSONL)
    combined_guidelines = g_records + cdc_records
    build_v11_index("guidelines_kb", combined_guidelines)
    
    # 2. Drug Labels KB
    print("Loading Drug Labels Data...")
    d_records = load_jsonl(DRUGLABELS_JSONL)
    build_v11_index("druglabels_kb", d_records)
    
    print("\n[V11 Done] All indices upgraded to 1024 dimensions.")

if __name__ == "__main__":
    main()
