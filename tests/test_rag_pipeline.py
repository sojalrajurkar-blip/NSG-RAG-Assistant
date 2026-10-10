"""Integration and unit tests for RAGPipeline orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
import pymupdf

from src.config import AppConfig
from src.embeddings import EmbeddingService
from src.vector_store import ChromaVectorStore
from src.retriever import SemanticRetriever, RetrievedChunk
from src.llm_service import LLMService, GenerationResult
from src.rag_pipeline import RAGPipeline, PipelineAnswer


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Creates a temporary 2-page test PDF for end-to-end pipeline indexing."""
    pdf_path = tmp_path / "test_course.pdf"
    doc = pymupdf.open()

    p1 = doc.new_page()
    p1.insert_text((50, 50), "Module 1: Vector Databases. ChromaDB stores dense vector representations.")

    p2 = doc.new_page()
    p2.insert_text((50, 50), "Module 2: LLM Evaluation. Compare Baseline LLM against Grounded RAG.")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def mock_llm_service() -> MagicMock:
    """Mocked LLMService to test pipeline without invoking live APIs."""
    mock = MagicMock(spec=LLMService)
    mock.config = AppConfig(gemini_api_key="test_key", gemini_model_name="gemini-mock")

    # Mock baseline response
    mock.generate_baseline.return_value = GenerationResult(
        answer="Baseline response from model.",
        mode="LLM-Only",
        model_name="gemini-mock",
        latency_seconds=0.5,
        citations=[],
    )

    # Mock grounded response
    mock.generate_grounded.return_value = GenerationResult(
        answer="Grounded answer based on ChromaDB [Page 1].",
        mode="RAG",
        model_name="gemini-mock",
        latency_seconds=0.7,
        citations=["Page 1"],
    )

    return mock


def test_rag_pipeline_indexing(tmp_path: Path, sample_pdf: Path, mock_llm_service: MagicMock):
    """Verifies document indexing creates chunks, updates vector store, and returns timing."""
    store_dir = str(tmp_path / "pipeline_chroma")
    store = ChromaVectorStore(persist_directory=store_dir, collection_name="test_indexing")
    emb = EmbeddingService()

    pipeline = RAGPipeline(
        vector_store=store,
        embedding_service=emb,
        llm_service=mock_llm_service,
    )

    metrics = pipeline.index_document(str(sample_pdf), chunk_size=100, chunk_overlap=20)

    assert metrics["doc_name"] == "test_course.pdf"
    assert metrics["pages_loaded"] == 2
    assert metrics["chunks_indexed"] >= 2
    assert metrics["total_index_count"] >= 2
    assert metrics["indexing_latency_seconds"] >= 0.0


def test_rag_pipeline_answer_rag_mode(tmp_path: Path, sample_pdf: Path, mock_llm_service: MagicMock):
    """Verifies complete RAG execution: retrieval -> context construction -> generation."""
    store_dir = str(tmp_path / "pipeline_chroma_rag")
    store = ChromaVectorStore(persist_directory=store_dir, collection_name="test_rag")
    emb = EmbeddingService()

    pipeline = RAGPipeline(
        vector_store=store,
        embedding_service=emb,
        llm_service=mock_llm_service,
    )

    pipeline.index_document(str(sample_pdf), chunk_size=100, chunk_overlap=20)

    # Ask question in RAG mode
    ans: PipelineAnswer = pipeline.answer_question("Tell me about ChromaDB", mode="RAG", top_k=2)

    assert ans.mode == "RAG"
    assert "Grounded answer" in ans.answer
    assert "Page 1" in ans.citations
    assert len(ans.retrieved_chunks) > 0
    assert ans.retrieval_latency_seconds > 0.0
    assert ans.generation_latency_seconds > 0.0
    assert ans.total_latency_seconds >= ans.retrieval_latency_seconds


def test_rag_pipeline_answer_llm_only_mode(tmp_path: Path, mock_llm_service: MagicMock):
    """Verifies baseline execution: skips retrieval and directly calls baseline LLM."""
    store_dir = str(tmp_path / "pipeline_chroma_base")
    store = ChromaVectorStore(persist_directory=store_dir, collection_name="test_base")
    emb = EmbeddingService()

    pipeline = RAGPipeline(
        vector_store=store,
        embedding_service=emb,
        llm_service=mock_llm_service,
    )

    ans: PipelineAnswer = pipeline.answer_question("What is machine learning?", mode="LLM-Only")

    assert ans.mode == "LLM-Only"
    assert "Baseline response" in ans.answer
    assert len(ans.citations) == 0
    assert len(ans.retrieved_chunks) == 0
    assert ans.retrieval_latency_seconds == 0.0
    assert ans.generation_latency_seconds > 0.0


def test_rag_pipeline_empty_index(tmp_path: Path, mock_llm_service: MagicMock):
    """Verifies that asking RAG on an empty index returns honest instructions without crash."""
    empty_store = ChromaVectorStore(
        persist_directory=str(tmp_path / "empty_pipe"),
        collection_name="empty",
    )
    emb = EmbeddingService()

    pipeline = RAGPipeline(
        vector_store=empty_store,
        embedding_service=emb,
        llm_service=mock_llm_service,
    )

    ans: PipelineAnswer = pipeline.answer_question("Where is the data?", mode="RAG")
    assert ans.mode == "RAG"
    assert "No documents are currently indexed" in ans.answer
    assert len(ans.retrieved_chunks) == 0
