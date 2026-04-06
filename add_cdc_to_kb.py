"""
add_cdc_to_kb.py

Adds CDC TB guideline HTML files to the existing guidelines_kb FAISS index.
Uses the same BGE embedding model and LangChain format as rebuild_indices_v2.py

Usage:
    cd ~/ScholarBOT
    python3 add_cdc_to_kb.py --html_dir /path/to/html/files

Example:
    python3 add_cdc_to_kb.py --html_dir ~/Downloads/cdc_html
"""

import argparse, json, re, sys, time, pickle
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser()
parser.add_argument("--html_dir", required=True, help="Directory containing CDC HTML files")
parser.add_argument("--chunk_size", type=int, default=400, help="Chunk size in characters")
parser.add_argument("--overlap", type=int, default=80, help="Overlap between chunks")
args = parser.parse_args()

HTML_DIR = Path(args.html_dir).expanduser().resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
FAISS_DIR = SCRIPT_DIR / "faiss_indices" / "guidelines_kb"

if not HTML_DIR.exists():
    print(f"ERROR: HTML directory not found: {HTML_DIR}")
    sys.exit(1)

if not FAISS_DIR.exists():
    print(f"ERROR: guidelines_kb index not found at {FAISS_DIR}")
    print("Please run rebuild_indices_v2.py first.")
    sys.exit(1)

html_files = list(HTML_DIR.glob("*.html"))
if not html_files:
    print(f"ERROR: No HTML files found in {HTML_DIR}")
    sys.exit(1)

print(f"Found {len(html_files)} HTML files:")
for f in html_files:
    print(f"  {f.name}")
print()

# ── Extract text from HTML ───────────────────────────────────────────────────
def extract_text_from_html(path: Path) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    # Remove nav, header, footer, scripts
    for tag in soup.find_all(["nav", "header", "footer", "script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

# ── Chunk text ───────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int, overlap: int, source_name: str):
    chunks = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current += " " + sent
        else:
            if current.strip():
                chunks.append({
                    "text": current.strip(),
                    "metadata": {
                        "source": "CDC",
                        "organization": "Centers for Disease Control and Prevention",
                        "document_name": source_name,
                        "section": "CDC TB Guidelines",
                        "section_title": source_name,
                        "doc_type": "guideline",
                        "page_numbers": [],
                    }
                })
            # Start new chunk with overlap
            words = current.split()
            overlap_text = " ".join(words[-overlap//5:]) if len(words) > overlap//5 else ""
            current = overlap_text + " " + sent
    if current.strip():
        chunks.append({
            "text": current.strip(),
            "metadata": {
                "source": "CDC",
                "organization": "Centers for Disease Control and Prevention",
                "document_name": source_name,
                "section": "CDC TB Guidelines",
                "section_title": source_name,
                "doc_type": "guideline",
                "page_numbers": [],
            }
        })
    return chunks

# ── Process all HTML files ───────────────────────────────────────────────────
all_new_chunks = []
for html_file in html_files:
    print(f"Processing: {html_file.name}")
    text = extract_text_from_html(html_file)
    source_name = html_file.stem.replace("_", " ").title()
    chunks = chunk_text(text, args.chunk_size, args.overlap, source_name)
    print(f"  Extracted {len(text)} chars → {len(chunks)} chunks")
    all_new_chunks.extend(chunks)

print(f"\nTotal new chunks to add: {len(all_new_chunks)}")

# ── Load embedding model ─────────────────────────────────────────────────────
print("\nLoading BAAI/bge-base-en-v1.5...")
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
EMBED_DIM = st_model.get_sentence_embedding_dimension()
print(f"Embedding dim: {EMBED_DIM}")

# ── Load existing index ───────────────────────────────────────────────────────
import faiss as faiss_lib
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

print(f"\nLoading existing guidelines_kb from {FAISS_DIR}...")
sys.path.insert(0, str(SCRIPT_DIR))
from embedding_utils import MedCPTDualEmbedder
embedder = MedCPTDualEmbedder()

store = FAISS.load_local(str(FAISS_DIR), embedder, allow_dangerous_deserialization=True)
print(f"  Existing ntotal: {store.index.ntotal}")

# ── Embed new chunks ──────────────────────────────────────────────────────────
print(f"\nEmbedding {len(all_new_chunks)} new chunks...")
texts = [c["text"] for c in all_new_chunks]
metas = [c["metadata"] for c in all_new_chunks]

all_vecs = []
t0 = time.time()
for i in range(0, len(texts), 64):
    batch = texts[i:i+64]
    vecs = st_model.encode(batch, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    all_vecs.append(vecs)
    print(f"  {min(i+64, len(texts))}/{len(texts)} ({time.time()-t0:.1f}s)", end="\r")
print()

vectors = np.vstack(all_vecs).astype("float32")

# ── Add to existing store ─────────────────────────────────────────────────────
print("\nAdding new chunks to guidelines_kb...")
new_docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metas)]

# Get current max id
current_count = store.index.ntotal
store.index.add(vectors)

# Add to docstore
for i, doc in enumerate(new_docs):
    doc_id = str(current_count + i)
    store.docstore.add({doc_id: doc})
    store.index_to_docstore_id[current_count + i] = doc_id

print(f"  New ntotal: {store.index.ntotal}")

# ── Save updated index ────────────────────────────────────────────────────────
print(f"\nSaving updated guidelines_kb to {FAISS_DIR}...")
store.save_local(str(FAISS_DIR))
print(f"Done! guidelines_kb updated: {store.index.ntotal} total chunks")
print(f"  Added {len(all_new_chunks)} new CDC chunks")
print(f"  Previous: {current_count} chunks")
