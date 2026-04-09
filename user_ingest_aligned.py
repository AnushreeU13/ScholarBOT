"""
user_ingest_local.py

On-demand ingestion of a user PDF into FAISS (KB_USER_FACT).
Local-first and portable.

Usage:
  python user_ingest_local.py --pdf path/to/file.pdf --doc_name mydoc

Outputs:
  - checkpoint jsonl under datasets/KB_processed/user_fact/
  - updates FAISS index under faiss_indices/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from config import (
    FAISS_INDICES_DIR, KB_USER_FACT,
    CHUNK_SIZE, OVERLAP,
    KB_PROCESSED_DIR,
)
from embedding_utils import MedCPTDualEmbedder
from storage_utils import create_faiss_store
from pdf_utils import extract_text_by_page
from chunking_utils import chunk_document

# Module-level singleton — avoids reloading the 1.3 GB BGE model on every upload
_embedder_instance: MedCPTDualEmbedder | None = None

def _get_embedder() -> MedCPTDualEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = MedCPTDualEmbedder()
    return _embedder_instance


def ingest_user_pdf(
    pdf_path: str | Path,
    doc_name: str,
    store_name: str = KB_USER_FACT,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    embed_batch_size: int = 16,
) -> Dict:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    embedder = _get_embedder()
    dim = embedder.dim
    store = create_faiss_store(store_name, dim, base_dir=FAISS_INDICES_DIR, embedder=embedder)

    # Ingest
    pages = extract_text_by_page(str(pdf_path))
    chunks = []
    chunk_texts: List[str] = []
    chunk_metas: List[Dict] = []
    
    debug_sample = "No pages extracted"
    if pages:
        # pages[0] is (text, page_num)
        p1_text = pages[0][0]
        debug_sample = f"Page 1 ({len(p1_text)} chars): {p1_text[:100]}..."
    
    for page_text, page_num in pages:
        page_chunks = chunk_document(
            text=page_text,
            document_name=doc_name,
            page_number=page_num,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunks.extend(page_chunks)

    for ch in chunks:
            t = (ch.get("text") or "").strip()
            if len(t) < 50:  # Relaxed from 120
                continue

            # Read page_number from the chunk's own metadata (set by chunk_document),
            # not from the outer loop variable which holds the last page's value.
            ch_meta = (ch.get("metadata") or {})
            chunk_page_num = ch_meta.get("page_number")

            meta = ch_meta.copy()
            meta.update({
                "source_type": "user_pdf",
                "organization": "user",
                "document_name": doc_name,
                "page_number": chunk_page_num,
                # CRITICAL: keep the chunk text in metadata for downstream evidence assembly
                "text": t,
                "ingested_at": datetime.utcnow().isoformat(),
                "embed_model": embedder.name,
            })

            chunk_texts.append(t)
            chunk_metas.append(meta)

    # Save checkpoint
    out_dir = (KB_PROCESSED_DIR / "user_fact").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{doc_name}_chunks.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for t, m in zip(chunk_texts, chunk_metas):
            f.write(json.dumps({"text": t, "metadata": m}, ensure_ascii=False) + "\n")

    # Embed + add
    for i in range(0, len(chunk_texts), embed_batch_size):
        batch_texts = chunk_texts[i:i + embed_batch_size]
        batch_metas = chunk_metas[i:i + embed_batch_size]
        # embs = embedder.embed_texts(batch_texts, batch_size=len(batch_texts)).astype(np.float32)
        store.add_texts(batch_texts, batch_metas)

    store.save_local(str(Path(FAISS_INDICES_DIR) / store_name))
    stats = {}
    # FAISS object doesn't have get_stats standard, so we manual it
    stats["total_vectors"] = store.index.ntotal
    stats["checkpoint_jsonl"] = str(out_jsonl)
    stats["added_chunks"] = len(chunk_texts)
    stats["total_chars"] = sum(len(t) for t in chunk_texts)
    stats["num_pages"] = len(pages)
    stats["debug_sample"] = debug_sample
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--doc_name", required=True)
    args = ap.parse_args()

    stats = ingest_user_pdf(args.pdf, args.doc_name)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
