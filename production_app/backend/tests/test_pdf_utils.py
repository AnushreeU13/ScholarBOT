from unittest.mock import MagicMock, patch

import pytest

from app.pdf_utils import _strip_reference_lines, extract_text_by_page


def test_strip_reference_lines_removes_references_section():
    text = (
        "Isoniazid is first-line therapy.\n"
        "References\n"
        "1. Smith J, et al. (2020) Some journal article.\n"
        "https://example.com/paper\n"
    )
    cleaned = _strip_reference_lines(text)
    assert "Isoniazid is first-line therapy." in cleaned
    assert "Smith" not in cleaned
    assert "example.com" not in cleaned


def test_strip_reference_lines_removes_doi_and_bracket_refs():
    text = "Main finding here.\n[12] Some bracketed reference text.\ndoi:10.1000/xyz\n"
    cleaned = _strip_reference_lines(text)
    assert "Main finding here." in cleaned
    assert "[12]" not in cleaned
    assert "doi:" not in cleaned


def test_extract_text_by_page_raises_on_missing_file():
    with pytest.raises(RuntimeError):
        extract_text_by_page("/nonexistent/path/does-not-exist.pdf")


def test_extract_text_by_page_parses_pages(tmp_path):
    fake_page = MagicMock()
    fake_page.extract_text.return_value = "Clinical content about tuberculosis."
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    with patch("PyPDF2.PdfReader", return_value=fake_reader):
        pages = extract_text_by_page(str(pdf_path))

    assert pages == [("Clinical content about tuberculosis.", 1)]


def test_extract_text_by_page_skips_empty_pages(tmp_path):
    empty_page = MagicMock()
    empty_page.extract_text.return_value = ""
    fake_reader = MagicMock()
    fake_reader.pages = [empty_page]

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    with patch("PyPDF2.PdfReader", return_value=fake_reader):
        pages = extract_text_by_page(str(pdf_path))

    assert pages == []
