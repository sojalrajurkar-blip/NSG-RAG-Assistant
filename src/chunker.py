"""Text Chunking and Metadata Preservation Module for NSG RAG Assistant.

Splits extracted document text into configurable, overlapping chunks
while preserving strict provenance (document name, source page numbers, chunk ID).
"""

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Dict, List, Optional

from src.pdf_loader import DocumentPage


@dataclass
class TextChunk:
    """Represents a discrete semantic chunk of text with strict provenance."""
    chunk_id: str
    doc_name: str
    page_numbers: List[int]
    text: str
    token_count: int
    char_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk object to dictionary format."""
        return {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "page_numbers": self.page_numbers,
            "text": self.text,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "metadata": self.metadata,
        }


def estimate_tokens(text: str) -> int:
    """Estimates token count using standard word/subword heuristic (~1.3 tokens per word).

    This provides a fast, deterministic token estimate without requiring heavy
    external tokenizer models.
    """
    words = text.split()
    if not words:
        return 0
    # Rule of thumb: 1 word ≈ 1.33 tokens
    return max(1, int(len(words) * 1.33))


def _generate_chunk_id(doc_name: str, page_numbers: List[int], chunk_index: int, text: str) -> str:
    """Generates a stable, deterministic chunk ID using document name, page, and content hash."""
    clean_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", doc_name).strip("_")
    first_page = page_numbers[0] if page_numbers else 1
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{clean_name}_p{first_page}_c{chunk_index}_{content_hash}"


def _detect_heading(text: str) -> Optional[str]:
    """Extracts prominent uppercase or titled header line if present at start."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        first_line = lines[0]
        if len(first_line) < 80 and (first_line.isupper() or "ASSIGNMENT" in first_line.upper()):
            return first_line
    return None


def split_text_into_chunks(
    text: str,
    doc_name: str,
    page_number: int,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    start_chunk_index: int = 0,
) -> List[TextChunk]:
    """Splits a single block of text into overlapping chunks using paragraph & sentence boundaries.

    Args:
        text: Text to split.
        doc_name: Document name for provenance.
        page_number: Page number where text originates.
        chunk_size: Target maximum token count per chunk.
        chunk_overlap: Number of tokens to overlap between successive chunks.
        start_chunk_index: Starting sequential counter for chunk IDs.

    Returns:
        List of TextChunk objects.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})")

    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    # Approximate word limits (1 word ≈ 1.33 tokens)
    chunk_size_words = max(10, int(chunk_size / 1.33))
    chunk_overlap_words = max(0, int(chunk_overlap / 1.33))
    step = max(1, chunk_size_words - chunk_overlap_words)

    words = cleaned_text.split()
    if len(words) <= chunk_size_words:
        chunk_id = _generate_chunk_id(doc_name, [page_number], start_chunk_index, cleaned_text)
        heading = _detect_heading(cleaned_text)
        return [
            TextChunk(
                chunk_id=chunk_id,
                doc_name=doc_name,
                page_numbers=[page_number],
                text=cleaned_text,
                token_count=estimate_tokens(cleaned_text),
                char_count=len(cleaned_text),
                metadata={"section_heading": heading, "has_overlap": False},
            )
        ]

    chunks: List[TextChunk] = []
    chunk_idx = start_chunk_index

    for i in range(0, len(words), step):
        chunk_slice = words[i : i + chunk_size_words]
        if not chunk_slice:
            break
        chunk_str = " ".join(chunk_slice)
        chunk_tokens = estimate_tokens(chunk_str)
        chunk_id = _generate_chunk_id(doc_name, [page_number], chunk_idx, chunk_str)
        heading = _detect_heading(chunk_str)

        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                doc_name=doc_name,
                page_numbers=[page_number],
                text=chunk_str,
                token_count=chunk_tokens,
                char_count=len(chunk_str),
                metadata={
                    "section_heading": heading,
                    "has_overlap": chunk_idx > start_chunk_index,
                },
            )
        )
        chunk_idx += 1

        if i + chunk_size_words >= len(words):
            break

    return chunks


def chunk_document_pages(
    pages: List[DocumentPage],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> List[TextChunk]:
    """Splits a list of DocumentPage objects into chunks while maintaining provenance.

    Args:
        pages: List of DocumentPage instances from pdf_loader.
        chunk_size: Target token count per chunk (default: 500).
        chunk_overlap: Token overlap between chunks (default: 80).

    Returns:
        List of TextChunk instances ready for vector embedding.
    """
    all_chunks: List[TextChunk] = []
    chunk_counter = 0

    for page in pages:
        if page.is_empty or not page.text.strip():
            continue

        page_chunks = split_text_into_chunks(
            text=page.text,
            doc_name=page.doc_name,
            page_number=page.page_number,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_chunk_index=chunk_counter,
        )

        all_chunks.extend(page_chunks)
        chunk_counter += len(page_chunks)

    return all_chunks
