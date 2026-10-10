"""Semantic Retrieval Module for NSG RAG Assistant.

Performs vector similarity search in ChromaDB using query embeddings.
Enriches retrieved chunks with provenance metadata, similarity scores,
and provides a transparent debug view before generation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.embeddings import EmbeddingService
from src.vector_store import ChromaVectorStore


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk with similarity score and provenance."""
    chunk_id: str
    doc_name: str
    page_numbers: List[int]
    primary_page: int
    text: str
    similarity: float  # Cosine similarity (1.0 = identical, 0.0 = orthogonal)
    distance: float    # Cosine distance (0.0 = identical)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert retrieved chunk to dictionary format."""
        return {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "page_numbers": self.page_numbers,
            "primary_page": self.primary_page,
            "text": self.text,
            "similarity": self.similarity,
            "distance": self.distance,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    """Encapsulates the complete result of a semantic search operation."""
    query: str
    chunks: List[RetrievedChunk]
    top_k: int
    min_similarity_threshold: float
    is_empty_query: bool = False
    is_empty_index: bool = False
    has_confident_results: bool = False

    def debug_summary(self) -> str:
        """Generates a detailed, beginner-friendly diagnostic summary."""
        lines = [
            f"Query: '{self.query}'",
            f"Index status: {'EMPTY' if self.is_empty_index else 'POPULATED'}",
            f"Retrieved: {len(self.chunks)} chunk(s) (requested top_k={self.top_k})",
            f"Confidence threshold: {self.min_similarity_threshold}",
            f"Confident retrieval: {self.has_confident_results}",
            "-" * 60,
        ]

        if not self.chunks:
            if self.is_empty_query:
                lines.append("Reason: Question was empty.")
            elif self.is_empty_index:
                lines.append("Reason: Vector database is empty. No documents indexed yet.")
            else:
                lines.append("Reason: No chunks found matching query.")
            return "\n".join(lines)

        for i, c in enumerate(self.chunks, 1):
            snippet = c.text[:140].replace("\n", " ") + ("..." if len(c.text) > 140 else "")
            lines.append(
                f"[{i}] ID: {c.chunk_id} | Doc: {c.doc_name} | Page: {c.primary_page} | "
                f"Similarity: {c.similarity:.4f} (Dist: {c.distance:.4f})"
            )
            lines.append(f"    Excerpt: {snippet}")

        lines.append("-" * 60)
        lines.append(
            "Note: Vector similarity indicates geometric topic proximity, "
            "not a mathematical guarantee that the chunk answers the question."
        )
        return "\n".join(lines)


class SemanticRetriever:
    """Orchestrates embedding questions and retrieving nearest chunks from ChromaDB."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_service: EmbeddingService,
        default_top_k: int = 3,
        min_similarity_threshold: float = 0.25,
    ):
        """Initializes semantic retriever.

        Args:
            vector_store: Initialized ChromaVectorStore instance.
            embedding_service: Initialized EmbeddingService instance.
            default_top_k: Number of nearest chunks to retrieve.
            min_similarity_threshold: Cutoff below which retrieval is marked unconfident.
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.default_top_k = default_top_k
        self.min_similarity_threshold = min_similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Retrieves top-k most relevant chunks for a given user query.

        Args:
            query: The user's question or search query.
            top_k: Optional override for number of chunks to fetch.

        Returns:
            RetrievalResult containing ranked chunks and diagnostics.
        """
        cleaned_query = (query or "").strip()
        k = top_k if top_k is not None and top_k > 0 else self.default_top_k

        # 1. Handle empty query
        if not cleaned_query:
            return RetrievalResult(
                query=cleaned_query,
                chunks=[],
                top_k=k,
                min_similarity_threshold=self.min_similarity_threshold,
                is_empty_query=True,
                is_empty_index=False,
                has_confident_results=False,
            )

        # 2. Handle empty index
        if self.vector_store.count() == 0:
            return RetrievalResult(
                query=cleaned_query,
                chunks=[],
                top_k=k,
                min_similarity_threshold=self.min_similarity_threshold,
                is_empty_query=False,
                is_empty_index=True,
                has_confident_results=False,
            )

        # 3. Embed the user question
        query_vector = self.embedding_service.embed_query(cleaned_query)

        # 4. Search ChromaDB
        raw_results = self.vector_store.query(
            query_embedding=query_vector,
            top_k=k,
        )

        retrieved_chunks: List[RetrievedChunk] = []
        for r in raw_results:
            meta = r.get("metadata", {})
            doc_name = meta.get("doc_name", "unknown")
            primary_page = int(meta.get("primary_page", 1))

            pages_raw = meta.get("pages", str(primary_page))
            try:
                page_numbers = [int(p) for p in str(pages_raw).split(",") if p.strip().isdigit()]
            except Exception:
                page_numbers = [primary_page]

            chunk_obj = RetrievedChunk(
                chunk_id=r.get("chunk_id", ""),
                doc_name=doc_name,
                page_numbers=page_numbers or [primary_page],
                primary_page=primary_page,
                text=r.get("text", ""),
                similarity=float(r.get("similarity", 0.0)),
                distance=float(r.get("distance", 1.0)),
                metadata=meta,
            )
            retrieved_chunks.append(chunk_obj)

        # 5. Assess confidence based on top similarity score
        has_confidence = False
        if retrieved_chunks:
            highest_similarity = max(c.similarity for c in retrieved_chunks)
            has_confidence = highest_similarity >= self.min_similarity_threshold

        return RetrievalResult(
            query=cleaned_query,
            chunks=retrieved_chunks,
            top_k=k,
            min_similarity_threshold=self.min_similarity_threshold,
            is_empty_query=False,
            is_empty_index=False,
            has_confident_results=has_confidence,
        )
