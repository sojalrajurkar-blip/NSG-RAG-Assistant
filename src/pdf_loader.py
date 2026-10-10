"""PDF Ingestion and Text Extraction Module for NSG RAG Assistant.

Uses PyMuPDF to extract text page-by-page while retaining complete
provenance metadata (document name, page number, character counts).
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pymupdf


class PDFLoaderError(Exception):
    """Base exception for all PDF loading errors."""
    pass


class PDFNotFoundError(PDFLoaderError):
    """Raised when the specified PDF file cannot be found."""
    pass


class InvalidPDFFormatError(PDFLoaderError):
    """Raised when the file does not have a valid .pdf extension."""
    pass


class CorruptedPDFError(PDFLoaderError):
    """Raised when the PDF file is corrupted or cannot be parsed."""
    pass


class EmptyPDFError(PDFLoaderError):
    """Raised when the PDF contains zero pages or no extractable content."""
    pass


@dataclass
class DocumentPage:
    """Represents text extracted from a single PDF page with provenance metadata."""
    doc_name: str
    page_number: int  # 1-indexed (Page 1 is the first page)
    text: str
    char_count: int
    is_empty: bool
    is_likely_scanned: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert page object to dictionary format for storage/display."""
        return {
            "doc_name": self.doc_name,
            "page_number": self.page_number,
            "text": self.text,
            "char_count": self.char_count,
            "is_empty": self.is_empty,
            "is_likely_scanned": self.is_likely_scanned,
            "metadata": self.metadata,
        }


def load_pdf(
    file_path: str,
    min_chars_threshold: int = 30,
) -> List[DocumentPage]:
    """Loads a PDF file and extracts text page by page.

    Args:
        file_path: Absolute or relative path to the PDF file.
        min_chars_threshold: Pages with fewer characters than this threshold
            are flagged as empty or potentially scanned/image-only.

    Returns:
        List of DocumentPage objects, one for each page in the document.

    Raises:
        PDFNotFoundError: If file does not exist.
        InvalidPDFFormatError: If file is not a PDF.
        CorruptedPDFError: If PyMuPDF cannot parse the file.
        EmptyPDFError: If document has 0 pages.
    """
    path = Path(file_path)

    # 1. Validate file existence
    if not path.is_file():
        raise PDFNotFoundError(f"PDF file not found at path: '{file_path}'")

    # 2. Validate file extension
    if path.suffix.lower() != ".pdf":
        raise InvalidPDFFormatError(
            f"Expected a .pdf file, but got '{path.suffix}' for '{file_path}'"
        )

    # 3. Open and parse document with PyMuPDF
    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        raise CorruptedPDFError(
            f"Failed to open or parse PDF '{file_path}': {str(exc)}"
        ) from exc

    try:
        total_pages = len(doc)
        if total_pages == 0:
            raise EmptyPDFError(f"PDF document '{file_path}' contains zero pages.")

        pages: List[DocumentPage] = []
        doc_name = path.name

        for page_idx in range(total_pages):
            page_num = page_idx + 1  # 1-indexed for human readability
            page = doc[page_idx]

            # Extract textual content
            raw_text = page.get_text("text") or ""
            cleaned_text = raw_text.strip()
            char_count = len(cleaned_text)

            is_empty = char_count == 0

            # Detect if page contains images but very little/no text (scanned PDF indicator)
            image_list = page.get_images()
            is_likely_scanned = bool(image_list and char_count < min_chars_threshold)

            page_metadata = {
                "total_pages": total_pages,
                "file_path": str(path.resolve()),
                "has_images": len(image_list) > 0,
                "image_count": len(image_list),
            }

            pages.append(
                DocumentPage(
                    doc_name=doc_name,
                    page_number=page_num,
                    text=cleaned_text,
                    char_count=char_count,
                    is_empty=is_empty,
                    is_likely_scanned=is_likely_scanned,
                    metadata=page_metadata,
                )
            )

        return pages

    finally:
        # Guarantee resource closure
        doc.close()


def safe_preview(pages: List[DocumentPage], max_chars_per_page: int = 200) -> str:
    """Generates a safe preview summary of extracted pages without huge dumps."""
    if not pages:
        return "No pages extracted."

    summary_lines = [
        f"Document: {pages[0].doc_name} (Total Pages: {len(pages)})",
        "=" * 60,
    ]

    for p in pages:
        preview_snippet = (
            p.text[:max_chars_per_page].replace("\n", " ") + "..."
            if len(p.text) > max_chars_per_page
            else p.text.replace("\n", " ")
        )
        status = []
        if p.is_empty:
            status.append("EMPTY")
        if p.is_likely_scanned:
            status.append("LIKELY SCANNED / OCR NEEDED")

        status_tag = f" [{', '.join(status)}]" if status else ""
        summary_lines.append(
            f"Page {p.page_number} ({p.char_count} chars){status_tag}: {preview_snippet or '(No text)'}"
        )

    return "\n".join(summary_lines)
