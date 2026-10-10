"""Complete End-to-End RAG Pipeline Orchestrator for NSG RAG Assistant.

Connects document ingestion, text chunking, local embeddings, persistent
ChromaDB storage, semantic retrieval, and Google Gemini generation into
a unified, modular workflow.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional

from src.config import AppConfig, load_config
from src.pdf_loader import load_pdf, DocumentPage
from src.chunker import chunk_document_pages, TextChunk
from src.embeddings import EmbeddingService
from src.vector_store import ChromaVectorStore
from src.retriever import SemanticRetriever, RetrievalResult, RetrievedChunk
from src.llm_service import LLMService, GenerationResult


@dataclass
class PipelineAnswer:
    """Structured response object containing generation output, citations, and full provenance diagnostics."""
    query: str
    mode: str  # "RAG" or "LLM-Only"
    answer: str
    citations: List[str] = field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    total_latency_seconds: float = 0.0
    retrieval_latency_seconds: float = 0.0
    generation_latency_seconds: float = 0.0
    is_grounded: bool = False
    has_confident_context: bool = False
    error: Optional[str] = None
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline answer to dictionary format."""
        return {
            "query": self.query,
            "mode": self.mode,
            "answer": self.answer,
            "citations": self.citations,
            "retrieved_chunk_count": len(self.retrieved_chunks),
            "total_latency_seconds": round(self.total_latency_seconds, 3),
            "retrieval_latency_seconds": round(self.retrieval_latency_seconds, 3),
            "generation_latency_seconds": round(self.generation_latency_seconds, 3),
            "is_grounded": self.is_grounded,
            "has_confident_context": self.has_confident_context,
            "error": self.error,
            "model_name": self.model_name,
        }


class RAGPipeline:
    """Coordinates end-to-end RAG indexing and question answering."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        retriever: Optional[SemanticRetriever] = None,
        llm_service: Optional[LLMService] = None,
    ):
        """Initializes the RAG Pipeline components.

        Supports dependency injection for deterministic testing and flexibility.
        """
        self.config = config or load_config()

        # Shared embedding service
        self.embedding_service = embedding_service or EmbeddingService(
            model_name=self.config.embedding_model_name
        )

        # Persistent vector database
        self.vector_store = vector_store or ChromaVectorStore(
            persist_directory=self.config.chroma_persist_directory
        )

        # Semantic retriever
        self.retriever = retriever or SemanticRetriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            default_top_k=self.config.default_top_k,
            min_similarity_threshold=self.config.min_similarity_threshold,
        )

        # Gemini LLM generation service
        self.llm_service = llm_service or LLMService(config=self.config)

    def index_document(
        self,
        file_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Indexes a PDF file into the local vector database.

        Indexing is decoupled from question answering so documents are not
        re-embedded repeatedly on every query.

        Args:
            file_path: Path to the target PDF document.
            chunk_size: Optional override for tokens per chunk.
            chunk_overlap: Optional override for token overlap.

        Returns:
            Dictionary containing indexing metrics (pages, chunks, elapsed time).
        """
        start_time = time.perf_counter()
        c_size = chunk_size or self.config.default_chunk_size
        c_overlap = chunk_overlap or self.config.default_chunk_overlap

        # 1. Page extraction with PyMuPDF
        pages: List[DocumentPage] = load_pdf(file_path)

        # 2. Text chunking with sliding window
        chunks: List[TextChunk] = chunk_document_pages(
            pages=pages,
            chunk_size=c_size,
            chunk_overlap=c_overlap,
        )

        if not chunks:
            return {
                "doc_name": pages[0].doc_name if pages else "unknown",
                "pages_loaded": len(pages),
                "chunks_indexed": 0,
                "total_index_count": self.vector_store.count(),
                "indexing_latency_seconds": round(time.perf_counter() - start_time, 3),
            }

        # 3. Dense embedding generation
        texts_to_embed = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_texts(texts_to_embed)

        # 4. Upsert into ChromaDB
        indexed_count = self.vector_store.add_chunks(chunks, embeddings)
        elapsed = time.perf_counter() - start_time

        return {
            "doc_name": chunks[0].doc_name,
            "pages_loaded": len(pages),
            "chunks_indexed": indexed_count,
            "total_index_count": self.vector_store.count(),
            "indexing_latency_seconds": round(elapsed, 3),
        }

    def answer_question(
        self,
        query: str,
        mode: str = "RAG",
        top_k: Optional[int] = None,
    ) -> PipelineAnswer:
        """Executes question answering in either RAG or LLM-Only baseline mode.

        Args:
            query: The user's question.
            mode: "RAG" (context-grounded) or "LLM-Only" (direct baseline).
            top_k: Optional count of chunks to retrieve in RAG mode.

        Returns:
            PipelineAnswer with timing, citations, and retrieved excerpts.
        """
        total_start = time.perf_counter()
        normalized_mode = "LLM-Only" if mode.strip().lower() in ("llm-only", "llm_only", "baseline") else "RAG"

        # -------------------------------------------------------------
        # 1. LLM-Only Baseline Path
        # -------------------------------------------------------------
        if normalized_mode == "LLM-Only":
            gen_res = self.llm_service.generate_baseline(query)
            total_elapsed = time.perf_counter() - total_start

            return PipelineAnswer(
                query=query,
                mode="LLM-Only",
                answer=gen_res.answer,
                citations=[],
                retrieved_chunks=[],
                total_latency_seconds=total_elapsed,
                retrieval_latency_seconds=0.0,
                generation_latency_seconds=gen_res.latency_seconds,
                is_grounded=False,
                has_confident_context=False,
                error=gen_res.error,
                model_name=gen_res.model_name,
            )

        # -------------------------------------------------------------
        # 2. RAG Path (Retrieval -> Context Construction -> Grounded Gen)
        # -------------------------------------------------------------
        retrieval_start = time.perf_counter()
        retrieval_result: RetrievalResult = self.retriever.retrieve(query, top_k=top_k)
        retrieval_elapsed = time.perf_counter() - retrieval_start

        # Handle empty index edge case
        if retrieval_result.is_empty_index:
            total_elapsed = time.perf_counter() - total_start
            return PipelineAnswer(
                query=query,
                mode="RAG",
                answer="No documents are currently indexed in the knowledge base. Please index a document first before asking questions in RAG mode.",
                citations=[],
                retrieved_chunks=[],
                total_latency_seconds=total_elapsed,
                retrieval_latency_seconds=retrieval_elapsed,
                generation_latency_seconds=0.0,
                is_grounded=False,
                has_confident_context=False,
                model_name=self.config.gemini_model_name,
            )

        # Grounded generation with retrieved context
        gen_res = self.llm_service.generate_grounded(
            query=query,
            chunks=retrieval_result.chunks,
        )
        total_elapsed = time.perf_counter() - total_start

        return PipelineAnswer(
            query=query,
            mode="RAG",
            answer=gen_res.answer,
            citations=gen_res.citations,
            retrieved_chunks=retrieval_result.chunks,
            total_latency_seconds=total_elapsed,
            retrieval_latency_seconds=retrieval_elapsed,
            generation_latency_seconds=gen_res.latency_seconds,
            is_grounded=not gen_res.is_fallback,
            has_confident_context=retrieval_result.has_confident_results,
            error=gen_res.error,
            model_name=gen_res.model_name,
        )
