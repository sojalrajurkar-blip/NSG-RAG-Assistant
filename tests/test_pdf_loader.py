"""Tests for PDF loader module."""

import os
from pathlib import Path
import pytest
import pymupdf

from src.pdf_loader import (
    load_pdf,
    safe_preview,
    DocumentPage,
    PDFNotFoundError,
    InvalidPDFFormatError,
    CorruptedPDFError,
    EmptyPDFError,
)


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """Creates a temporary synthetic 3-page PDF fixture for deterministic testing."""
    pdf_path = tmp_path / "sample_guide.pdf"
    doc = pymupdf.open()

    # Page 1: Standard text
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Welcome to NSG Academy AI Engineering.\nThis is Page 1 content.")

    # Page 2: Multiple paragraphs
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Module 2: RAG Pipeline Overview.\nHere we discuss vector databases.")

    # Page 3: Blank page (no text)
    doc.new_page()

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_load_pdf_success(sample_pdf_path: Path):
    """Verifies that load_pdf extracts all pages with accurate provenance metadata."""
    pages = load_pdf(str(sample_pdf_path))

    assert len(pages) == 3
    assert pages[0].doc_name == "sample_guide.pdf"
    assert pages[0].page_number == 1
    assert "Welcome to NSG Academy" in pages[0].text
    assert pages[0].char_count > 0
    assert not pages[0].is_empty

    # Verify Page 2
    assert pages[1].page_number == 2
    assert "Module 2: RAG Pipeline" in pages[1].text
    assert not pages[1].is_empty

    # Verify Page 3 (Empty page)
    assert pages[2].page_number == 3
    assert pages[2].is_empty
    assert pages[2].char_count == 0


def test_load_pdf_file_not_found(tmp_path: Path):
    """Verifies that non-existent files trigger PDFNotFoundError."""
    non_existent = tmp_path / "ghost_file.pdf"
    with pytest.raises(PDFNotFoundError) as exc_info:
        load_pdf(str(non_existent))
    assert "not found" in str(exc_info.value).lower()


def test_load_pdf_invalid_format(tmp_path: Path):
    """Verifies that non-pdf extensions trigger InvalidPDFFormatError."""
    fake_txt = tmp_path / "notes.txt"
    fake_txt.write_text("Just some text")
    with pytest.raises(InvalidPDFFormatError) as exc_info:
        load_pdf(str(fake_txt))
    assert "Expected a .pdf file" in str(exc_info.value)


def test_load_pdf_corrupted_file(tmp_path: Path):
    """Verifies that invalid/corrupt PDF binaries trigger CorruptedPDFError."""
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"NOT_A_VALID_PDF_HEADER_OR_CONTENT")
    with pytest.raises(CorruptedPDFError):
        load_pdf(str(corrupt_file))


def test_safe_preview(sample_pdf_path: Path):
    """Verifies that safe_preview formats readable output without dumping entire text."""
    pages = load_pdf(str(sample_pdf_path))
    preview = safe_preview(pages, max_chars_per_page=50)

    assert "Document: sample_guide.pdf (Total Pages: 3)" in preview
    assert "Page 1" in preview
    assert "Page 2" in preview
    assert "Page 3 (0 chars) [EMPTY]: (No text)" in preview


def test_user_actual_pdf_if_present():
    """Tests ingestion of user-provided assignment PDF if present in data/documents."""
    user_pdf = Path("data/documents/NSG_Product_Engineering_AI_Assignment_Series_2026_STUDENT.pdf")
    if not user_pdf.exists():
        pytest.skip("User assignment PDF not found, skipping real file test")

    pages = load_pdf(str(user_pdf))
    assert len(pages) > 0
    assert pages[0].doc_name == "NSG_Product_Engineering_AI_Assignment_Series_2026_STUDENT.pdf"
    assert pages[0].page_number == 1
    assert pages[0].char_count > 0
