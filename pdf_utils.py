"""
PDF processing utilities using PyPDF2.
"""

import PyPDF2
from typing import List, Tuple, Optional
from pathlib import Path
import re

def _strip_reference_like_lines(text: str) -> str:
    """
    Remove obvious reference sections and reference-style lines from page text.

    Heuristics:
      * If a line starts with 'References' (case-insensitive), drop that line and
        everything after it (typical end-of-paper reference section).
      * Drop lines that look like numbered references, e.g.
        '12. Smith J, Johnson K. Title... (2010)...'
      * Drop lines that are mostly DOI information.
    """
    lines = text.splitlines()
    cleaned_lines: List[str] = []
    in_references_section = False

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        # Start of references section
        if not in_references_section and lower.startswith("references"):
            in_references_section = True
            continue

        # If we are already in the references section, skip everything
        if in_references_section:
            continue

        # Lines that look like numbered references, e.g. "12. Smith J ..."
        if re.match(r"^\d+\.\s*[A-Z][A-Za-z\-]+.*\(\d{4}\)", stripped):
            continue

        # Lines that look like DOI lines
        if "doi:" in lower or lower.startswith("doi "):
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


