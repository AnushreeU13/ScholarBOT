"""
06_ingest_user.py
On-demand ingestion of a user-uploaded PDF into the user_kb FAISS index.
Architecture position: 06 — called by the Streamlit UI on file upload.

Improvements vs old user_ingest_aligned.py:
- Reads each chunk's page_number from its own metadata (not outer loop variable)
- Uses embedder singleton (no re-load of 1.3 GB model per upload)
- Builds and saves BM25 pickle after ingestion so retriever loads it instantly
"""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

_cfg     = importlib.import_module("01_config")
_emb     = importlib.import_module("02_embedding_utils")
_storage = importlib.import_module("03_storage_utils")
_pdf     = importlib.import_module("04_pdf_utils")
_chunk   = importlib.import_module("05_chunking_utils")


def ingest_user_pdf(
    pdf_path,
    doc_name: str,
    store_name: str = None,
    chunk_size: int = None,
    overlap: int = None,
    embed_batch_size: int = 16,
) -> Dict:
    """
    Extract, chunk, embed, and index a user PDF.
    Returns stats dict: {added_chunks, total_chars, num_pages, debug_sample, ...}
    """
    store_name = store_name or _cfg.KB_USER
    chunk_size = chunk_size or _cfg.CHUNK_SIZE
    overlap    = overlap    or _cfg.OVERLAP
    pdf_path   = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    embedder = _emb.get_embedder()
    store    = _storage.get_or_create_faiss(store_name, embedder.dim, embedder)

    # ── Extract ───────────────────────────────────────────────────────────────
    pages = _pdf.extract_text_by_page(str(pdf_path))

    debug_sample = "No pages extracted"
    if pages:
        debug_sample = f"Page 1 ({len(pages[0][0])} chars): {pages[0][0][:120]}..."

    # ── Chunk ─────────────────────────────────────────────────────────────────
    all_raw_chunks = []
    for page_text, page_num in pages:
        page_chunks = _chunk.chunk_document(
            text=page_text,
            document_name=doc_name,
            page_number=page_num,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_raw_chunks.extend(page_chunks)

    # ── Filter + build final metadata ─────────────────────────────────────────
    chunk_texts: List[str] = []
    chunk_metas: List[Dict] = []

    for ch in all_raw_chunks:
        text = (ch.get("text") or "").strip()
        if len(text) < 50:
            continue

        # Read page_number from the chunk's own metadata — NOT from the outer loop variable
        ch_meta       = ch.get("metadata") or {}
        chunk_page    = ch_meta.get("page_number")

        meta = ch_meta.copy()
        meta.update({
            "source_type":   "user_pdf",
            "organization":  "user",
            "document_name": doc_name,
            "page_number":   chunk_page,
            "text":          text,          # kept for downstream evidence assembly
            "ingested_at":   datetime.utcnow().isoformat(),
            "embed_model":   embedder.name,
        })

        chunk_texts.append(text)
        chunk_metas.append(meta)

    # ── Save checkpoint JSONL ─────────────────────────────────────────────────
    out_dir  = (_cfg.KB_PROCESSED_DIR / "user_fact").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{doc_name}_chunks.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for t, m in zip(chunk_texts, chunk_metas):
            f.write(json.dumps({"text": t, "metadata": m}, ensure_ascii=False) + "\n")

    # ── Embed + add to FAISS ──────────────────────────────────────────────────
    for i in range(0, len(chunk_texts), embed_batch_size):
        store.add_texts(chunk_texts[i:i + embed_batch_size],
                        chunk_metas[i:i + embed_batch_size])

    save_path = Path(_cfg.FAISS_INDICES_DIR) / store_name
    store.save_local(str(save_path))

    # ── Build + save BM25 index so retriever loads it instantly ──────────────
    _storage.load_or_build_bm25(store_name, store, force_rebuild=True)

    return {
        "added_chunks":    len(chunk_texts),
        "total_chars":     sum(len(t) for t in chunk_texts),
        "num_pages":       len(pages),
        "checkpoint_jsonl": str(out_jsonl),
        "debug_sample":    debug_sample,
    }


def main():
    ap = argparse.ArgumentParser(description="Ingest a user PDF into ScholarBOT user_kb")
    ap.add_argument("--pdf",      required=True)
    ap.add_argument("--doc_name", required=True)
    args = ap.parse_args()
    stats = ingest_user_pdf(args.pdf, args.doc_name)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
