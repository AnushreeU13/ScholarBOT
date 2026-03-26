"""
ingest_kb.py

Builds ScholarBOT's two FAISS knowledge-base indices from raw source files:
  - guidelines_kb  : TB and Pneumonia/CAP clinical guidelines and research papers
                     Sources: PDF, HTML (CDC pages)
  - druglabels_kb  : Drug label / prescribing information documents
                     Sources: PDF, FDA SPL XML

Usage
-----
    # Ingest guidelines (PDF + HTML) and drug labels (PDF + XML)
    python ingest_kb.py \\
        --guidelines   "C:/path/to/guideline/pdfs" \\
        --guidelines-html "C:/path/to/cdc/html" \\
        --druglabels   "C:/path/to/druglabel/pdfs" \\
        --druglabels-xml "C:/path/to/spl/xml"

    # Rebuild from scratch (deletes existing indices first)
    python ingest_kb.py --guidelines "..." --druglabels "..." --rebuild

    # Dry-run: show what would be ingested without writing anything
    python ingest_kb.py --guidelines "..." --dry-run

    # Skip specific files (in addition to the built-in TB exclusion list)
    python ingest_kb.py --guidelines "..." --exclude "bad_file.pdf" "another.pdf"

Output
------
- faiss_indices/guidelines_kb/   FAISS index for guidelines
- faiss_indices/druglabels_kb/   FAISS index for drug labels
- datasets/KB_processed/guidelines_chunks.jsonl   Chunk cache (inspect/debug)
- datasets/KB_processed/druglabels_chunks.jsonl   Chunk cache (inspect/debug)

Notes
-----
- Requires re-running after any change to pdf_utils.py, chunking_utils.py,
  or the embedding model (BAAI/bge-base-en-v1.5).
- The existing faiss_indices/main_kb/ is NOT modified or deleted by this script.
  You can remove it manually once the new indices are verified.
- m729.pdf in the Tuberculosis/ folder is AES-encrypted and cannot be read;
  it is excluded by default and flagged for manual review.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    FAISS_INDICES_DIR,
    KB_DRUGLABELS,
    KB_GUIDELINES,
    KB_PROCESSED_DIR,
    CHUNK_SIZE,
    OVERLAP,
)
from pdf_utils import extract_text_by_page
from chunking_utils import chunk_document
from embedding_utils import MedCPTDualEmbedder
from storage_utils import create_faiss_store

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NOW = time.strftime("%Y-%m-%d %H:%M:%S")

# Files confirmed out-of-scope during data audit (2026-03-26).
# These are skipped even when they appear inside a guidelines directory.
# - m729.pdf          : AES-encrypted; content unknown; cannot be ingested safely.
# - Scaling-up-*      : Hepatitis C elimination model (Lancet 2018); misplaced.
# - Origins-of-the-*  : Drug-resistant malaria genetics (Lancet ID 2018); misplaced.
_DEFAULT_EXCLUDE: frozenset = frozenset(
    [
        "m729.pdf",
        "Scaling-up-prevention-and-treatment-towards-the-el.pdf",
        "Origins-of-the-current-outbreak-of-multidrug-resis.pdf",
        # Previously audited out-of-scope files (HIV, Zika, malaria, errata, misc)
        "15-7832_bouchet.pdf",
        "19-0987.pdf",
        "19-1079.pdf",
        "22_23_Revised_HIV_TB.pdf",
        "a78695.pdf",
        "Chandrasekaran2020_Article_ImpactOfCOVID-19OnTheTBPandemi.pdf",
        "EML_6thEd_ExecutiveSummary_ENG_14web.pdf",
        "Fauci.pdf",
        "funding-overview.pdf",
        "global-fund-strategy-summary-en.pdf",
        "Global-Plan-to-End-TB-2016-2020.pdf",
        "hiv-tb-collaboration.pdf",
        "MSF_TB-Report_2015.pdf",
        "RoadMapforZikaVirusResearchOPEN.pdf",
        "session3.pdf",
        "the-end-tb-strategy.pdf",
        "Unitaid TB Diagnostics Technology Landscape 2015.pdf",
        "WHO_2019_Global_TB_Report.pdf",  # keep only if not a duplicate of WHO 2023
        "ZikavirusWHO.pdf",
        "zika-virus-classification-tables.pdf",
    ]
)

# Section header keywords used to tag chunks with a section_group.
# Mirrors rag_pipeline_aligned._section_group_from_meta so that
# section-bias boosting works correctly at query time.
_SECTION_PATTERNS: List[Tuple[str, List[str]]] = [
    ("dosage",          ["dose", "dosage", "admin", "administration", "posology"]),
    ("contraindications", ["contraindication"]),
    ("warnings",        ["warn", "precaution", "boxed", "black box"]),
    ("adverse",         ["adverse", "side effect", "reaction", "toxicity", "undesirable"]),
    ("interactions",    ["interact", "cyp", "drug-drug"]),
    ("indications",     ["indication", "indications", "use in", "approved for"]),
    ("g_treatment",     ["recommendation", "treatment", "therapy", "management",
                         "regimen", "duration", "monitoring", "rationale"]),
    ("g_diagnosis",     ["diagnos", "testing", "evaluation", "workup",
                         "radiograph", "x-ray", "imaging", "culture", "sputum", "pcr"]),
    ("g_prevention",    ["prevention", "prevent", "prophylaxis", "vaccin"]),
]

# FDA SPL LOINC codes → section_group (for XML drug label ingestion)
_SPL_LOINC_TO_GROUP: Dict[str, str] = {
    "34068-7": "dosage",           # Dosage & Administration
    "34067-9": "indications",      # Indications & Usage
    "34070-3": "contraindications",# Contraindications
    "43685-7": "warnings",         # Warnings and Precautions
    "34071-1": "warnings",         # Warnings (older code)
    "34084-4": "adverse",          # Adverse Reactions
    "34073-7": "interactions",     # Drug Interactions
    "34090-1": "other",            # Clinical Pharmacology
    "34089-3": "other",            # Description
    "34092-7": "other",            # Clinical Studies
    "34076-0": "other",            # Information for Patients
    "42229-5": "other",            # Special Populations
    "34082-8": "other",            # Geriatric Use
    "34083-6": "other",            # Pediatric Use
}

# HL7 SPL namespace used in FDA XML files
_SPL_NS = "urn:hl7-org:v3"


def _detect_section_group(text_fragment: str) -> str:
    """Guess section group from the first 200 chars of a chunk."""
    fragment = text_fragment[:200].lower()
    for group, keywords in _SECTION_PATTERNS:
        if any(k in fragment for k in keywords):
            return group
    return "other"


def _annotate_chunks(
    chunks: List[Dict[str, Any]],
    doc_type: str,
    source_type: str,
    section_group_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Add required metadata fields to every chunk in-place and return the list."""
    for i, c in enumerate(chunks):
        meta = c.get("metadata", {}) or {}
        meta.update(
            {
                "doc_type": doc_type,
                "source_type": source_type,
                "section_group": section_group_override or _detect_section_group(c["text"]),
                "embed_model": "BAAI/bge-base-en-v1.5",
                "ingested_at": _NOW,
            }
        )
        c["metadata"] = meta
    # Re-index globally
    for i, c in enumerate(chunks):
        c["metadata"]["chunk_index"] = i
        c["metadata"]["total_chunks"] = len(chunks)
    return chunks


# ---------------------------------------------------------------------------
# PDF → chunks
# ---------------------------------------------------------------------------

def _pdf_to_chunks(
    pdf_path: Path,
    doc_type: str,
    source_type: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> List[Dict[str, Any]]:
    """Extract, clean, chunk, and annotate one PDF."""
    pages: List[Tuple[str, int]] = extract_text_by_page(str(pdf_path))
    if not pages:
        print(f"    [WARN] No text extracted from {pdf_path.name} — skipping.")
        return []

    all_chunks: List[Dict[str, Any]] = []
    for page_text, page_number in pages:
        if not page_text.strip():
            continue
        chunks = chunk_document(
            text=page_text,
            document_name=pdf_path.name,
            page_number=page_number,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(chunks)

    return _annotate_chunks(all_chunks, doc_type, source_type)


# ---------------------------------------------------------------------------
# HTML → chunks  (CDC clinical pages)
# ---------------------------------------------------------------------------

def _html_to_text(html_path: Path) -> List[Tuple[str, int]]:
    """
    Parse an HTML file and return a list of (text_block, pseudo_page_number) tuples.

    Each top-level heading (h1–h3) starts a new logical "page" so that
    page_number metadata is meaningful for citation purposes.

    Requires BeautifulSoup4. Falls back to raw text stripping if unavailable.
    """
    raw = html_path.read_text(encoding="utf-8", errors="replace")

    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(raw, "html.parser")

        # Remove non-content elements
        for tag in soup.find_all(
            ["script", "style", "nav", "header", "footer", "aside",
             "form", "button", "noscript", "iframe"]
        ):
            tag.decompose()
        for tag in soup.find_all(attrs={"class": re.compile(
            r"breadcrumb|utility-nav|sidebar|menu|ad-|cookie|skip-nav|sr-only",
            re.I
        )}):
            tag.decompose()

        # Try to isolate the main content block
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id=re.compile(r"content|main", re.I))
            or soup.find(attrs={"class": re.compile(r"\bcontent\b|\bmain\b", re.I)})
            or soup.body
            or soup
        )

        # Split at h1/h2/h3 boundaries to create logical sections
        pages: List[Tuple[str, int]] = []
        current_lines: List[str] = []
        pseudo_page = 1

        def flush():
            nonlocal pseudo_page
            text = "\n".join(current_lines).strip()
            if len(text) > 80:
                pages.append((text, pseudo_page))
                pseudo_page += 1
            current_lines.clear()

        for element in main.descendants:
            if not hasattr(element, "name"):
                continue
            if element.name in ("h1", "h2", "h3"):
                flush()
                heading = element.get_text(" ", strip=True)
                if heading:
                    current_lines.append(f"\n{heading.upper()}\n")
            elif element.name in ("p", "li", "td", "th", "dt", "dd"):
                text = element.get_text(" ", strip=True)
                if text:
                    current_lines.append(text)

        flush()
        return pages

    except ImportError:
        # Fallback: strip all HTML tags with regex
        print(f"    [WARN] BeautifulSoup4 not installed; using regex fallback for {html_path.name}")
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"\s{3,}", "\n\n", text)
        text = text.strip()
        if not text:
            return []
        # Split into ~2000-char pseudo-pages
        pages = []
        for i, start in enumerate(range(0, len(text), 2000), start=1):
            chunk = text[start:start + 2000].strip()
            if chunk:
                pages.append((chunk, i))
        return pages


def _html_to_chunks(
    html_path: Path,
    doc_type: str,
    source_type: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> List[Dict[str, Any]]:
    """Parse one HTML file into annotated chunks."""
    pages = _html_to_text(html_path)
    if not pages:
        print(f"    [WARN] No usable text in {html_path.name} — skipping.")
        return []

    all_chunks: List[Dict[str, Any]] = []
    for page_text, page_number in pages:
        if not page_text.strip():
            continue
        chunks = chunk_document(
            text=page_text,
            document_name=html_path.name,
            page_number=page_number,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(chunks)

    return _annotate_chunks(all_chunks, doc_type, source_type)


# ---------------------------------------------------------------------------
# SPL XML → chunks  (FDA drug labels)
# ---------------------------------------------------------------------------

def _spl_xml_to_sections(xml_path: Path) -> List[Tuple[str, str, str]]:
    """
    Parse one FDA SPL XML file and return a list of
        (section_title, section_text, section_group)
    for each clinically relevant section.

    Skips sections with no substantive text (< 30 chars).
    """
    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError as exc:
        print(f"    [WARN] XML parse error in {xml_path.name}: {exc} — skipping.")
        return []

    root = tree.getroot()

    def _ns(tag: str) -> str:
        return f"{{{_SPL_NS}}}{tag}"

    def _get_text(element) -> str:
        """Recursively extract all text from an element, joining with spaces."""
        parts = []
        for node in element.iter():
            if node.text and node.text.strip():
                parts.append(node.text.strip())
            if node.tail and node.tail.strip():
                parts.append(node.tail.strip())
        return " ".join(parts)

    sections: List[Tuple[str, str, str]] = []

    for sec in root.iter(_ns("section")):
        code_el = sec.find(_ns("code"))
        if code_el is None:
            continue
        loinc = code_el.get("code", "")
        if loinc not in _SPL_LOINC_TO_GROUP:
            continue

        group = _SPL_LOINC_TO_GROUP[loinc]

        title_el = sec.find(_ns("title"))
        title = title_el.text.strip() if title_el is not None and title_el.text else loinc

        text_el = sec.find(_ns("text"))
        if text_el is None:
            # Some SPL files nest content in paragraphs directly
            text = _get_text(sec)
        else:
            text = _get_text(text_el)

        # Clean up excessive whitespace
        text = re.sub(r" {3,}", "  ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        if len(text) < 30:
            continue

        # Prepend title so section context is in every chunk
        full_text = f"{title}\n\n{text}"
        sections.append((title, full_text, group))

    return sections


def _spl_xml_to_chunks(
    xml_path: Path,
    doc_type: str,
    source_type: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> List[Dict[str, Any]]:
    """Parse one SPL XML file into annotated chunks."""
    sections = _spl_xml_to_sections(xml_path)
    if not sections:
        print(f"    [WARN] No usable sections in {xml_path.name} — skipping.")
        return []

    all_chunks: List[Dict[str, Any]] = []
    for pseudo_page, (title, text, group) in enumerate(sections, start=1):
        chunks = chunk_document(
            text=text,
            document_name=xml_path.name,
            page_number=pseudo_page,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        # Override section_group with the LOINC-derived value (more precise)
        annotated = _annotate_chunks(chunks, doc_type, source_type,
                                     section_group_override=group)
        all_chunks.extend(annotated)

    # Re-index globally after merging all sections
    for i, c in enumerate(all_chunks):
        c["metadata"]["chunk_index"] = i
        c["metadata"]["total_chunks"] = len(all_chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Core ingestion engine
# ---------------------------------------------------------------------------

def _collect_files(
    directory: Path,
    extensions: List[str],
    exclude: frozenset,
) -> List[Path]:
    """Return sorted files with matching extensions, after applying exclusion list."""
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*.{ext}"))
    files = sorted(set(files))
    excluded = [f for f in files if f.name in exclude]
    if excluded:
        print(f"  [EXCLUDE] Skipping {len(excluded)} excluded file(s):")
        for f in excluded:
            print(f"    - {f.name}")
    return [f for f in files if f.name not in exclude]


def _process_file(
    file_path: Path,
    doc_type: str,
    source_type: str,
    chunk_size: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    """Dispatch to the correct parser based on file extension."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return _pdf_to_chunks(file_path, doc_type, source_type, chunk_size, overlap)
    elif ext in (".html", ".htm"):
        return _html_to_chunks(file_path, doc_type, source_type, chunk_size, overlap)
    elif ext == ".xml":
        return _spl_xml_to_chunks(file_path, doc_type, source_type, chunk_size, overlap)
    else:
        print(f"    [WARN] Unsupported file type: {file_path.suffix} — skipping.")
        return []


def ingest_source(
    source_dir: Path,
    extensions: List[str],
    kb_name: str,
    doc_type: str,
    source_type: str,
    faiss_dir: Path,
    processed_dir: Path,
    embedder: MedCPTDualEmbedder,
    exclude: frozenset,
    rebuild: bool = False,
    dry_run: bool = False,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> Tuple[int, int, int]:
    """
    Ingest all matching files from source_dir into a FAISS index.

    Returns (total_chunks, total_pages, skipped) for summary reporting.
    """
    files = _collect_files(source_dir, extensions, exclude)
    if not files:
        ext_str = "/".join(extensions)
        print(f"\n  [WARN] No .{ext_str} files found in {source_dir}")
        return 0, 0, 0

    ext_str = "/".join(extensions)
    print(f"\n  Source : {source_dir}  [{ext_str}]  ({len(files)} files)")

    index_path = faiss_dir / kb_name
    jsonl_path = processed_dir / f"{kb_name}_chunks.jsonl"

    if rebuild and not dry_run and index_path.exists():
        print(f"  [REBUILD] Deleting existing index at {index_path}")
        shutil.rmtree(index_path)

    if not dry_run:
        store = create_faiss_store(
            store_name=kb_name,
            dimension=embedder.dim,
            base_dir=str(faiss_dir),
            embedder=embedder,
        )
        jsonl_handle = open(jsonl_path, "a", encoding="utf-8")
    else:
        store = None
        jsonl_handle = None

    total_chunks = 0
    total_pages  = 0
    skipped      = 0

    for i, file_path in enumerate(files, start=1):
        print(f"\n  [{i}/{len(files)}] {file_path.name}")
        t0 = time.time()

        chunks = _process_file(file_path, doc_type, source_type, chunk_size, overlap)

        if not chunks:
            skipped += 1
            continue

        pages_in_doc = len({c["metadata"].get("page_number") for c in chunks})
        print(f"    Pages/sections: {pages_in_doc}  |  Chunks: {len(chunks)}")

        if dry_run:
            total_chunks += len(chunks)
            total_pages  += pages_in_doc
            sample = chunks[0]
            print(f"    Sample chunk ({len(sample['text'])} chars):")
            print(f"    {sample['text'][:200].replace(chr(10), ' ')!r}")
            print(f"    Section group: {sample['metadata'].get('section_group')}")
            continue

        texts = [c["text"]     for c in chunks]
        metas = [c["metadata"] for c in chunks]

        print(f"    Embedding {len(texts)} chunks...", end=" ", flush=True)
        embedder.embed_texts(texts, batch_size=32, show_progress=False)
        print(f"done ({time.time()-t0:.1f}s)")

        store.add_texts(texts=texts, metadatas=metas)

        for c in chunks:
            jsonl_handle.write(json.dumps(c, ensure_ascii=False) + "\n")

        total_chunks += len(chunks)
        total_pages  += pages_in_doc

    if not dry_run and store is not None:
        print(f"\n  Saving FAISS index to {index_path} ...")
        store.save_local(str(index_path))
        jsonl_handle.close()
        print(f"  Saved. Index size: {store.index.ntotal} vectors.")

    return total_chunks, total_pages, skipped


def ingest_kb(
    kb_name: str,
    doc_type: str,
    source_type: str,
    sources: List[Tuple[Path, List[str]]],   # [(dir, [ext, ...]), ...]
    faiss_dir: Path,
    processed_dir: Path,
    embedder: MedCPTDualEmbedder,
    exclude: frozenset,
    rebuild: bool = False,
    dry_run: bool = False,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> None:
    """Ingest all sources for one KB, printing a unified summary."""
    print(f"\n{'='*60}")
    print(f"  KB: {kb_name}")
    print(f"  Mode: {'DRY RUN' if dry_run else ('REBUILD' if rebuild else 'APPEND')}")
    print(f"{'='*60}")

    total_chunks = 0
    total_pages  = 0
    total_skipped = 0
    first = True

    for source_dir, extensions in sources:
        if not source_dir.is_dir():
            print(f"\n  [WARN] Directory not found: {source_dir} — skipping.")
            continue
        # Only delete the index on the first source if rebuild is requested
        c, p, s = ingest_source(
            source_dir=source_dir,
            extensions=extensions,
            kb_name=kb_name,
            doc_type=doc_type,
            source_type=source_type,
            faiss_dir=faiss_dir,
            processed_dir=processed_dir,
            embedder=embedder,
            exclude=exclude,
            rebuild=(rebuild and first),
            dry_run=dry_run,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        total_chunks  += c
        total_pages   += p
        total_skipped += s
        first = False

    jsonl_path = processed_dir / f"{kb_name}_chunks.jsonl"
    print(f"\n  Summary for {kb_name}:")
    print(f"    Sources ingested : {len(sources)}")
    print(f"    Total pages/secs : {total_pages}")
    print(f"    Total chunks     : {total_chunks}")
    print(f"    Files skipped    : {total_skipped}")
    if not dry_run:
        print(f"    JSONL cache      : {jsonl_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PDF/HTML/XML files into ScholarBOT FAISS knowledge bases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--guidelines",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory containing TB/Pneumonia guideline PDFs.",
    )
    parser.add_argument(
        "--guidelines-html",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory containing CDC clinical HTML pages (guidelines KB).",
    )
    parser.add_argument(
        "--druglabels",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory containing drug label PDFs.",
    )
    parser.add_argument(
        "--druglabels-xml",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory containing FDA SPL XML drug label files (druglabels KB).",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        metavar="FILENAME",
        help="Additional filenames to exclude (in addition to the built-in exclusion list).",
    )
    parser.add_argument(
        "--no-default-exclude",
        action="store_true",
        default=False,
        help="Disable the built-in exclusion list (use only --exclude).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help="Delete and recreate FAISS indices from scratch (default: append).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Process files and report stats without writing any index.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Token chunk size (default: {CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=OVERLAP,
        help=f"Token overlap between chunks (default: {OVERLAP}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    has_guidelines = args.guidelines is not None or args.guidelines_html is not None
    has_druglabels = args.druglabels is not None or args.druglabels_xml is not None

    if not has_guidelines and not has_druglabels:
        print("ERROR: Provide at least one of --guidelines, --guidelines-html,")
        print("       --druglabels, or --druglabels-xml.")
        print("Run with --help for usage.")
        sys.exit(1)

    # Build exclusion set
    exclude = frozenset() if args.no_default_exclude else _DEFAULT_EXCLUDE
    if args.exclude:
        exclude = exclude | frozenset(args.exclude)

    faiss_dir     = Path(FAISS_INDICES_DIR)
    processed_dir = Path(KB_PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nScholarBOT KB Ingestion")
    print(f"  FAISS dir      : {faiss_dir.resolve()}")
    print(f"  Processed dir  : {processed_dir.resolve()}")
    print(f"  Embedding model: BAAI/bge-base-en-v1.5  (768-dim)")
    print(f"  Chunk size     : {args.chunk_size} tokens")
    print(f"  Overlap        : {args.overlap} tokens")
    print(f"  Excluded files : {len(exclude)}")
    if args.exclude:
        print(f"  Extra excludes : {args.exclude}")

    print("\nLoading embedding model (this may take ~30s on first run)...")
    embedder = MedCPTDualEmbedder()
    print(f"  Model dim: {embedder.dim}")

    t_start = time.time()

    # ---- Guidelines KB ----
    if has_guidelines:
        guideline_sources: List[Tuple[Path, List[str]]] = []
        if args.guidelines is not None:
            guideline_sources.append((args.guidelines, ["pdf"]))
        if args.guidelines_html is not None:
            guideline_sources.append((args.guidelines_html, ["html", "htm"]))

        ingest_kb(
            kb_name=KB_GUIDELINES,
            doc_type="guideline",
            source_type="guidelines_kb",
            sources=guideline_sources,
            faiss_dir=faiss_dir,
            processed_dir=processed_dir,
            embedder=embedder,
            exclude=exclude,
            rebuild=args.rebuild,
            dry_run=args.dry_run,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )

    # ---- Drug Labels KB ----
    if has_druglabels:
        druglabel_sources: List[Tuple[Path, List[str]]] = []
        if args.druglabels is not None:
            druglabel_sources.append((args.druglabels, ["pdf"]))
        if args.druglabels_xml is not None:
            druglabel_sources.append((args.druglabels_xml, ["xml"]))

        ingest_kb(
            kb_name=KB_DRUGLABELS,
            doc_type="druglabel_spl",
            source_type="druglabels_kb",
            sources=druglabel_sources,
            faiss_dir=faiss_dir,
            processed_dir=processed_dir,
            embedder=embedder,
            exclude=exclude,
            rebuild=args.rebuild,
            dry_run=args.dry_run,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )

    elapsed = time.time() - t_start
    print(f"\nDone. Total time: {elapsed:.1f}s")
    if args.dry_run:
        print("(DRY RUN — no files were written)")


if __name__ == "__main__":
    main()
