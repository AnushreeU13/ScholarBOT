import json, sys, time
import numpy as np
from pathlib import Path

parser_args = sys.argv[1:]
kb_dir = None
for i, a in enumerate(parser_args):
    if a == "--kb_processed_dir":
        kb_dir = Path(parser_args[i+1]).expanduser().resolve()

if not kb_dir:
    print("Usage: python3 rebuild_indices_v2.py --kb_processed_dir ~/Downloads/KB_processed")
    sys.exit(1)

GUIDELINES_JSONL = kb_dir / "guidelines_text" / "guidelines_chunks_cleaned.jsonl"
DRUGLABELS_JSONL = kb_dir / "druglabels_text" / "druglabels_chunks.jsonl"
FAISS_DIR = Path(__file__).resolve().parent / "faiss_indices"

print("Loading BAAI/bge-base-en-v1.5 ...")
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
import faiss as faiss_lib

st_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
EMBED_DIM = st_model.get_sentence_embedding_dimension()
print(f"Embedding dim: {EMBED_DIM}")

from embedding_utils import MedCPTDualEmbedder
embedder = MedCPTDualEmbedder()

def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: records.append(json.loads(line))
                except: pass
    return records

def build_langchain_index(store_name, records):
    print(f"\n=== Building: {store_name} ({len(records)} records) ===")
    docs, texts = [], []
    for r in records:
        t = r.get("text") or r.get("chunk_text") or ""
        if t.strip():
            meta = {k:v for k,v in r.items() if k not in ("text","chunk_text")}
            docs.append(Document(page_content=t.strip(), metadata=meta))
            texts.append(t.strip())
    print(f"  Non-empty: {len(texts)}")
    
    # Embed with BGE
    t0 = time.time()
    all_vecs = []
    for i in range(0, len(texts), 256):
        batch = texts[i:i+256]
        vecs = st_model.encode(batch, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        all_vecs.append(vecs)
        print(f"  {min(i+256,len(texts))}/{len(texts)} ({time.time()-t0:.1f}s)", end="\r")
    print()
    vectors = np.vstack(all_vecs).astype("float32")
    
    # Build LangChain FAISS store
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

build_langchain_index("guidelines_kb", load_jsonl(GUIDELINES_JSONL))
build_langchain_index("druglabels_kb", load_jsonl(DRUGLABELS_JSONL))
print("\nDone! Both LangChain-compatible indices built.")
