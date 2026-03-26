"""
PDF processing utilities using PyPDF2.
"""

import PyPDF2
from typing import List, Tuple, Optional
from pathlib import Path
import re

def _strip_reference_like_lines(text: str) -> str:
    """
    Remove reference sections and reference-style lines from page text.

    Heuristics applied in order:
      1. Section-header triggers: 'References', 'Bibliography', 'Works Cited',
         'Literature Cited', 'Acknowledgements/Acknowledgments' — drop the
         header line and everything that follows it on the same page.
      2. Bracket-numbered references: '[1] Smith J ...' or '[12] ...'
      3. Period-numbered references: '12. Smith J, Johnson K. Title... (2010)'
      4. DOI lines: any line containing 'doi:' / 'https://doi.org' / 'doi.org'
      5. URL-only lines: lines whose only non-whitespace content is a URL
         (http://, https://, www.)
      6. Author-list lines: lines that look like 'Smith JA, Jones RB, et al.'
         with no sentence-ending punctuation and multiple comma-separated tokens.
      7. Journal citation fragments: lines matching volume/issue/page patterns
         (e.g., 'J Bras Pneumol. 2017;43(5):472-486')
      8. 'Available at:' / 'Retrieved from:' / 'Accessed:' lines.
    """
    # Patterns compiled once
    _BRACKET_REF   = re.compile(r"^\[\d+\]\s+[A-Z]")
    _PERIOD_REF    = re.compile(r"^\d{1,3}\.\s+[A-Z][A-Za-z\-]+.*\(\d{4}\)")
    _URL_LINE      = re.compile(r"^\s*(https?://|www\.)\S+\s*$", re.IGNORECASE)
    _AUTHOR_LIST   = re.compile(
        r"^([A-Z][A-Za-z\-]+\s+[A-Z]{1,3},?\s*){2,}.*(et al\.?|[A-Z]{1,3}\.)?\s*$"
    )
    _JOURNAL_FRAG  = re.compile(
        r"\d{4};\d+\(\d+\):\d+[-–]\d+|"   # 2017;43(5):472-486
        r"Vol\.\s*\d+|"
        r"pp\.\s*\d+"
    )
    _ACCESS_LINE   = re.compile(
        r"^\s*(available\s+at|retrieved\s+from|accessed|last\s+accessed)\s*[:\-]",
        re.IGNORECASE,
    )

    _SECTION_ENDERS = {
        "references", "bibliography", "works cited",
        "literature cited", "acknowledgements", "acknowledgments",
    }

    lines = text.splitlines()
    cleaned_lines: List[str] = []
    in_terminal_section = False

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        # 1. Section-header triggers — drop header + everything after
        if not in_terminal_section:
            for header in _SECTION_ENDERS:
                if lower == header or lower.startswith(header + " ") or lower.startswith(header + ":"):
                    in_terminal_section = True
                    break
        if in_terminal_section:
            continue

        # 2. Bracket-numbered reference  [1] / [12]
        if _BRACKET_REF.match(stripped):
            continue

        # 3. Period-numbered reference  12. Smith J ...
        if _PERIOD_REF.match(stripped):
            continue

        # 4. DOI lines
        if "doi:" in lower or "doi.org" in lower:
            continue

        # 5. URL-only lines
        if _URL_LINE.match(stripped):
            continue

        # 6. Author-list lines (comma-separated surnames + initials, no period ending sentence)
        if _AUTHOR_LIST.match(stripped) and not stripped.endswith("."):
            continue

        # 7. Journal citation fragments embedded in short lines (<120 chars)
        if len(stripped) < 120 and _JOURNAL_FRAG.search(stripped):
            continue

        # 8. "Available at:" / "Retrieved from:" / "Accessed:" lines
        if _ACCESS_LINE.match(stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, int]:
    """
    Extract text from PDF file.
    """
    text = ""
    num_pages = 0

    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            for page in pdf_reader.pages:
                page_text = str(page.extract_text() or "")
                page_text = _strip_reference_like_lines(page_text)
                if page_text.strip():
                    text += page_text + "\n"

    except Exception as e:
        raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")

    return text, num_pages



def extract_text_by_page(pdf_path: str) -> List[Tuple[str, int]]:
    """
    Extract text from PDF file page by page.
    """
    pages = []
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = str(page.extract_text() or "")
                page_text = _strip_reference_like_lines(page_text)
                if page_text.strip():
                    pages.append((page_text, page_num + 1))
    
    except Exception as e:
        raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")
    
    return pages


