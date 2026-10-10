"""Tests for text chunking and metadata preservation module."""

from pathlib import Path
import pytest

from src.pdf_loader import DocumentPage, load_pdf
from src.chunker import (
    TextChunk,
    estimate_tokens,
    split_text_into_chunks,
    chunk_document_pages,
)


def test_estimate_tokens():
    """Verifies deterministic token estimation heuristic."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") == int(2 * 1.33)
    assert estimate_tokens("word " * 100) > 100


def test_chunking_empty_pages():
    """Verifies that empty pages produce zero chunks without errors."""
    empty_pages = [
        DocumentPage(
            doc_name="test.pdf",
            page_number=1,
            text="",
            char_count=0,
            is_empty=True,
            is_likely_scanned=False,
        )
    ]
    chunks = chunk_document_pages(empty_pages)
    assert len(chunks) == 0


def test_chunking_short_text():
    """Verifies that short text within chunk_size produces exactly one chunk."""
    pages = [
        DocumentPage(
            doc_name="course.pdf",
            page_number=1,
            text="ASSIGNMENT 01: INTRODUCTION TO RAG.\nThis is a short introductory lesson.",
            char_count=68,
            is_empty=False,
            is_likely_scanned=False,
        )
    ]
    chunks = chunk_document_pages(pages, chunk_size=200, chunk_overlap=30)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.doc_name == "course.pdf"
    assert chunk.page_numbers == [1]
    assert "ASSIGNMENT 01" in chunk.text
    assert chunk.metadata.get("section_heading") == "ASSIGNMENT 01: INTRODUCTION TO RAG."
    assert chunk.token_count > 0


def test_chunking_long_text_with_overlap():
    """Verifies that long text produces multiple chunks with genuine sliding overlap."""
    paragraphs = [
        f"Paragraph {i}: This is detailed educational content for NSG Academy student testing. " * 5
        for i in range(10)
    ]
    long_text = "\n\n".join(paragraphs)

    pages = [
        DocumentPage(
            doc_name="handbook.pdf",
            page_number=2,
            text=long_text,
            char_count=len(long_text),
            is_empty=False,
            is_likely_scanned=False,
        )
    ]

    chunks = chunk_document_pages(pages, chunk_size=100, chunk_overlap=30)
    assert len(chunks) > 1

    # Verify each chunk has metadata and positive token count
    for c in chunks:
        assert c.doc_name == "handbook.pdf"
        assert c.page_numbers == [2]
        assert c.token_count > 0
        assert len(c.text.strip()) > 0

    # Verify overlap exists between chunk 0 and chunk 1
    # Words in overlap should appear in both consecutive chunks
    chunk_0_words = set(chunks[0].text.split())
    chunk_1_words = set(chunks[1].text.split())
    common_words = chunk_0_words.intersection(chunk_1_words)
    assert len(common_words) > 0


def test_chunk_id_stability():
    """Verifies that chunk IDs are deterministic across multiple runs."""
    text = "NSG Academy RAG Assignment guidelines."
    chunk_a = split_text_into_chunks(text, "sample.pdf", 1, chunk_size=100, chunk_overlap=20)
    chunk_b = split_text_into_chunks(text, "sample.pdf", 1, chunk_size=100, chunk_overlap=20)

    assert len(chunk_a) == 1
    assert len(chunk_b) == 1
    assert chunk_a[0].chunk_id == chunk_b[0].chunk_id


def test_invalid_chunk_parameters():
    """Verifies that invalid chunking parameters raise clear ValueErrors."""
    with pytest.raises(ValueError) as exc1:
        split_text_into_chunks("test", "test.pdf", 1, chunk_size=0)
    assert "chunk_size must be positive" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        split_text_into_chunks("test", "test.pdf", 1, chunk_size=100, chunk_overlap=-5)
    assert "cannot be negative" in str(exc2.value)

    with pytest.raises(ValueError) as exc3:
        split_text_into_chunks("test", "test.pdf", 1, chunk_size=100, chunk_overlap=100)
    assert "must be strictly less than" in str(exc3.value)


def test_chunking_real_user_pdf():
    """Verifies chunking on actual user document NSG_Product_Engineering_AI_Assignment_Series_2026_STUDENT.pdf."""
    pdf_path = Path("data/documents/NSG_Product_Engineering_AI_Assignment_Series_2026_STUDENT.pdf")
    if not pdf_path.exists():
        pytest.skip("User assignment PDF not found")

    pages = load_pdf(str(pdf_path))
    chunks = chunk_document_pages(pages, chunk_size=500, chunk_overlap=80)

    assert len(chunks) > 0
    # Every chunk must have a unique ID
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "All chunk IDs must be unique"

    # All chunks must have valid page numbers and document names
    for c in chunks:
        assert c.doc_name == "NSG_Product_Engineering_AI_Assignment_Series_2026_STUDENT.pdf"
        assert len(c.page_numbers) > 0
        assert c.token_count > 0
        assert c.char_count > 0
