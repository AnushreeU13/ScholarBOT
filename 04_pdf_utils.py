"""
04_pdf_utils.py
PDF text extraction with reference-section stripping.
Architecture position: 04 — used by user ingest (06).
"""

import re
from pathlib import Path
from typing import List, Tuple

import PyPDF2


# ── Reference stripping ───────────────────────────────────────────────────────

_BRACKET_REF  = re.compile(r"^\[\d+\]\s+[A-Z]")
_PERIOD_REF   = re.compile(r"^\d{1,3}\.\s+[A-Z][A-Za-z\-]+.*\(\d{4}\)")
_URL_LINE     = re.compile(r"^\s*(https?://|www\.)\S+\s*$", re.IGNORECASE)
_AUTHOR_LIST  = re.compile(
    r"^([A-Z][A-Za-z\-]+\s+[A-Z]{1,3},?\s*){2,}.*(et al\.?|[A-Z]{1,3}\.)?\s*$"
)
_JOURNAL_FRAG = re.compile(
    r"\d{4};\d+\(\d+\):\d+[-–]\d+|Vol\.\s*\d+|pp\.\s*\d+"
)
_ACCESS_LINE  = re.compile(
    r"^\s*(available\s+at|retrieved\s+from|accessed|last\s+accessed)\s*[:\-]",
    re.IGNORECASE,
)
_SECTION_ENDERS = {"references", "bibliography", "works cited", "literature cited",
                   "acknowledgements", "acknowledgments"}


def _strip_reference_lines(text: str) -> str:
    lines, cleaned, in_terminal = text.splitlines(), [], False
    for line in lines:
        stripped = line.strip()
        lower    = stripped.lower()

        if not in_terminal:
            for header in _SECTION_ENDERS:
                if lower == header or lower.startswith(header + " ") or lower.startswith(header + ":"):
                    in_terminal = True
                    break
        if in_terminal:
            continue
        if _BRACKET_REF.match(stripped):  continue
        if _PERIOD_REF.match(stripped):   continue
        if "doi:" in lower or "doi.org" in lower: continue
        if _URL_LINE.match(stripped):     continue
        if _AUTHOR_LIST.match(stripped) and not stripped.endswith("."): continue
        if len(stripped) < 120 and _JOURNAL_FRAG.search(stripped):     continue
        if _ACCESS_LINE.match(stripped):  continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_text_by_page(pdf_path: str) -> List[Tuple[str, int]]:
    """
    Returns [(page_text, page_number), ...] for all non-empty pages.
    page_number is 1-based.
    """
    pages = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = str(page.extract_text() or "")
                text = _strip_reference_lines(text)
                if text.strip():
                    pages.append((text, page_num + 1))
    except Exception as e:
        raise RuntimeError(f"PDF read error [{pdf_path}]: {e}") from e
    return pages


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, int]:
    """Returns (full_text, num_pages). Used by legacy scripts."""
    pages = extract_text_by_page(pdf_path)
    full  = "\n".join(t for t, _ in pages)
    return full, len(pages)
