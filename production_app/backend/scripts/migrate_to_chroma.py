"""
One-time migration: load the existing ScholarBOT KB chunk JSONL files
(produced by the original FAISS-based pipeline in the parent repo) into
Chroma collections for this production app.

Usage (from production_app/backend):
    python scripts/migrate_to_chroma.py \
        --guidelines ../../dataset/guidelines_chunks_cleaned.jsonl \
        --druglabels ../../dataset/druglabels_chunks.jsonl \
        --batch-size 64 \
        --limit 0          # 0 = no limit; use a small number for a quick smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.embedder import get_embedder  # noqa: E402
from app.vectorstore import get_store  # noqa: E402


def _read_jsonl(path: Path, limit: int) -> list:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _split_text_metadata(row: dict) -> tuple:
    text = (row.get("text") or "").strip()
    if "metadata" in row and isinstance(row["metadata"], dict):
        meta = dict(row["metadata"])
    else:
        meta = {k: v for k, v in row.items() if k != "text"}
    return text, meta


def migrate(jsonl_path: Path, collection_name: str, batch_size: int, limit: int) -> int:
    if not jsonl_path.exists():
        print(f"[migrate] SKIP — file not found: {jsonl_path}")
        return 0

    embedder = get_embedder()
    store = get_store(collection_name, embedder=embedder)

    rows = _read_jsonl(jsonl_path, limit)
    print(f"[migrate] {collection_name}: {len(rows)} rows from {jsonl_path.name}")

    added = 0
    batch_texts, batch_metas = [], []
    for row in rows:
        text, meta = _split_text_metadata(row)
        if len(text) < 20:
            continue
        batch_texts.append(text)
        batch_metas.append(meta)

        if len(batch_texts) >= batch_size:
            store.add_texts(batch_texts, batch_metas)
            added += len(batch_texts)
            print(f"[migrate] {collection_name}: {added}/{len(rows)} embedded", end="\r")
            batch_texts, batch_metas = [], []

    if batch_texts:
        store.add_texts(batch_texts, batch_metas)
        added += len(batch_texts)

    print(f"\n[migrate] {collection_name}: done — {added} chunks indexed (collection count={store.count()})")
    return added


def main():
    ap = argparse.ArgumentParser(description="Migrate ScholarBOT KB JSONL chunks into Chroma")
    ap.add_argument("--guidelines", type=Path, default=None)
    ap.add_argument("--druglabels", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit; cap rows per file for a smoke test")
    args = ap.parse_args()

    total = 0
    if args.guidelines:
        total += migrate(args.guidelines, config.COLLECTION_GUIDELINES, args.batch_size, args.limit)
    if args.druglabels:
        total += migrate(args.druglabels, config.COLLECTION_DRUGLABELS, args.batch_size, args.limit)

    print(f"[migrate] TOTAL indexed: {total}")


if __name__ == "__main__":
    main()
