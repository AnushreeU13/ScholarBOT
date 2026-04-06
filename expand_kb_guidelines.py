"""
expand_kb_guidelines.py

Downloads and adds TB + Pneumonia guidelines to guidelines_kb:

TB Guidelines:
- NICE NG33 TB Guidelines (UK, 2024)
- ATS/CDC/IDSA Drug-Susceptible TB 2016
- CDC TB Clinical Guidance

Pneumonia Guidelines:
- BTS Pneumonia Adults (UK)
- ERS/ESICM/ESCMID/ALAT CAP Guidelines 2023
- IDSA/ATS CAP Severity Criteria
- Surviving Sepsis Campaign Pneumonia
- WHO Pneumonia factsheet

Usage:
    cd ~/ScholarBOT
    pip install pdfminer.six beautifulsoup4
    python3 expand_kb_guidelines.py
"""

import sys, re, time
import numpy as np
from pathlib import Path
import urllib.request

SCRIPT_DIR = Path(__file__).resolve().parent
FAISS_DIR  = SCRIPT_DIR / "faiss_indices" / "guidelines_kb"

# ── Sources ───────────────────────────────────────────────────────────────────
SOURCES = [
    # ── TB ────────────────────────────────────────────────────────────────────
    {
        "url":  "https://www.nice.org.uk/guidance/ng33/chapter/recommendations",
        "name": "NICE TB Guidelines NG33 - Recommendations (UK 2024)",
        "org":  "NICE UK",
        "type": "html",
    },
    {
        "url":  "https://www.nice.org.uk/guidance/ng33/chapter/context",
        "name": "NICE TB Guidelines NG33 - Context (UK 2024)",
        "org":  "NICE UK",
        "type": "html",
    },
    {
        "url":  "https://www.cdc.gov/tb/hcp/clinical-guidance/index.html",
        "name": "CDC TB Clinical Guidelines Overview",
        "org":  "CDC",
        "type": "html",
    },
    {
        "url":  "https://www.cdc.gov/tb/publications/guidelines/pdf/clin-infect-dis.-2016-nahid-cid_ciw376.pdf",
        "name": "ATS CDC IDSA Drug-Susceptible TB Treatment Guidelines 2016",
        "org":  "ATS/CDC/IDSA",
        "type": "pdf",
    },

    # ── Pneumonia ─────────────────────────────────────────────────────────────
    {
        "url":  "https://www.brit-thoracic.org.uk/quality-improvement/guidelines/pneumonia-adults/",
        "name": "BTS Pneumonia in Adults Guidelines (UK)",
        "org":  "British Thoracic Society",
        "type": "html",
    },
    {
        "url":  "https://www.who.int/news-room/fact-sheets/detail/pneumonia",
        "name": "WHO Pneumonia Fact Sheet",
        "org":  "WHO",
        "type": "html",
    },
    {
        "url":  "https://www.cdc.gov/pneumonia/hcp/clinical-overview/index.html",
        "name": "CDC Pneumonia Clinical Overview",
        "org":  "CDC",
        "type": "html",
    },
    {
        "url":  "https://www.cdc.gov/pneumonia/hcp/bacterial-pneumonia/index.html",
        "name": "CDC Bacterial Pneumonia Clinical Guidance",
        "org":  "CDC",
        "type": "html",
    },
    {
        "url":  "https://www.cdc.gov/pneumonia/hcp/diagnosis-treatment/index.html",
        "name": "CDC Pneumonia Diagnosis and Treatment",
        "org":  "CDC",
        "type": "html",
    },
    {
        "url":  "https://www.thoracic.org/statements/resources/respiratory-infections/community-acquired-pneumonia-in-adults.pdf",
        "name": "ATS Community-Acquired Pneumonia Adults Statement",
        "org":  "ATS",
        "type": "pdf",
    },
    {
        "url":  "https://emedicine.medscape.com/article/234240-guidelines",
        "name": "Medscape CAP Clinical Practice Guidelines",
        "org":  "Medscape/IDSA",
        "type": "html",
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_html(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  ERROR: {e}")
        return ""

def fetch_pdf_text(url):
    import tempfile, os
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            pdf_bytes = resp.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp = f.name
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(tmp)
            return text or ""
        except Exception:
            try:
                import pypdf
                r = pypdf.PdfReader(tmp)
                return "\n".join(p.extract_text() or "" for p in r.pages)
            except Exception as e2:
                print(f"  PDF error: {e2}")
                return ""
        finally:
            os.unlink(tmp)
    except Exception as e:
        print(f"  ERROR: {e}")
        return ""

def html_to_text(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["nav","header","footer","script","style","noscript","aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def chunk_text(text, name, org, chunk_size=400, overlap=80):
    chunks = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current += " " + sent
        else:
            if current.strip() and len(current.strip()) > 80:
                chunks.append({
                    "text": current.strip(),
                    "metadata": {
                        "source":        org,
                        "organization":  org,
                        "document_name": name,
                        "section":       name,
                        "section_title": name,
                        "doc_type":      "guideline",
                        "page_numbers":  [],
                    }
                })
            words = current.split()
            overlap_text = " ".join(words[-overlap//5:]) if len(words) > overlap//5 else ""
            current = overlap_text + " " + sent
    if current.strip() and len(current.strip()) > 80:
        chunks.append({
            "text": current.strip(),
            "metadata": {
                "source":        org,
                "organization":  org,
                "document_name": name,
                "section":       name,
                "section_title": name,
                "doc_type":      "guideline",
                "page_numbers":  [],
            }
        })
    return chunks

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not FAISS_DIR.exists():
        print(f"ERROR: guidelines_kb not found: {FAISS_DIR}")
        sys.exit(1)

    print("=" * 60)
    print("Expanding guidelines_kb: TB + Pneumonia guidelines")
    print("=" * 60)

    all_chunks = []
    for src in SOURCES:
        print(f"\n[{src['org']}] {src['name']}")
        if src["type"] == "html":
            raw = fetch_html(src["url"])
            text = html_to_text(raw) if raw else ""
        else:
            text = fetch_pdf_text(src["url"])

        if len(text) < 200:
            print(f"  SKIPPED (too short or failed)")
            continue

        print(f"  Extracted {len(text):,} chars")
        chunks = chunk_text(text, src["name"], src["org"])
        print(f"  → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("\nNo new chunks. Exiting.")
        sys.exit(0)

    print(f"\nTotal new chunks: {len(all_chunks)}")

    # Load model
    print("\nLoading BAAI/bge-base-en-v1.5...")
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    # Load existing index
    sys.path.insert(0, str(SCRIPT_DIR))
    from embedding_utils import MedCPTDualEmbedder
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    embedder = MedCPTDualEmbedder()
    store = FAISS.load_local(str(FAISS_DIR), embedder, allow_dangerous_deserialization=True)
    prev = store.index.ntotal
    print(f"Existing ntotal: {prev}")

    # Embed
    texts = [c["text"] for c in all_chunks]
    metas = [c["metadata"] for c in all_chunks]
    print(f"\nEmbedding {len(texts)} chunks...")
    all_vecs = []
    t0 = time.time()
    for i in range(0, len(texts), 64):
        vecs = st_model.encode(texts[i:i+64], batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        all_vecs.append(vecs)
        print(f"  {min(i+64,len(texts))}/{len(texts)} ({time.time()-t0:.1f}s)", end="\r")
    print()

    vectors = np.vstack(all_vecs).astype("float32")
    store.index.add(vectors)
    for i, doc in enumerate([Document(page_content=t, metadata=m) for t,m in zip(texts,metas)]):
        doc_id = str(prev + i)
        store.docstore.add({doc_id: doc})
        store.index_to_docstore_id[prev + i] = doc_id

    store.save_local(str(FAISS_DIR))
    print(f"\n✅ Done! guidelines_kb: {prev} → {store.index.ntotal} chunks")
    print(f"   Added {len(all_chunks)} new chunks\n")

    from collections import Counter
    for org, n in Counter(c["metadata"]["org"] if "org" in c["metadata"] else c["metadata"]["organization"] for c in all_chunks).most_common():
        print(f"   {n:3d}x  {org}")

if __name__ == "__main__":
    main()
