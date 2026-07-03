"""On-demand ingestion of a user-uploaded PDF into the user_kb Chroma collection."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from app import chunking, config, pdf_utils


def ingest_user_pdf(pdf_path, doc_name: str, store, chunk_size: int = None, overlap: int = None) -> Dict:
    """
    Extract, chunk, and index a user PDF into the given ChromaStore.
    Returns stats: {added_chunks, total_chars, num_pages}.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.OVERLAP
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    pages = pdf_utils.extract_text_by_page(str(pdf_path))

    all_raw_chunks = []
    for page_text, page_num in pages:
        all_raw_chunks.extend(
            chunking.chunk_document(page_text, doc_name, page_number=page_num,
                                     chunk_size=chunk_size, overlap=overlap)
        )

    chunk_texts: List[str] = []
    chunk_metas: List[Dict] = []
    for ch in all_raw_chunks:
        text = (ch.get("text") or "").strip()
        if len(text) < 50:
            continue

        ch_meta = ch.get("metadata") or {}
        meta = ch_meta.copy()
        meta.update({
            "source_type": "user_pdf",
            "organization": "user",
            "document_name": doc_name,
            "page_number": ch_meta.get("page_number"),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        })
        chunk_texts.append(text)
        chunk_metas.append(meta)

    if chunk_texts:
        store.add_texts(chunk_texts, chunk_metas)

    return {
        "added_chunks": len(chunk_texts),
        "total_chars": sum(len(t) for t in chunk_texts),
        "num_pages": len(pages),
    }
