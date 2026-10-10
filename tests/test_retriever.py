"""Tests for semantic retrieval module."""

from pathlib import Path
import pytest

from src.embeddings import EmbeddingService
from src.vector_store import ChromaVectorStore
from src.retriever import SemanticRetriever, RetrievalResult, RetrievedChunk
from src.chunker import TextChunk


@pytest.fixture(scope="module")
def embedding_service() -> EmbeddingService:
    """Fixture providing initialized EmbeddingService."""
    return EmbeddingService(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def populated_retriever(tmp_path: Path, embedding_service: EmbeddingService) -> SemanticRetriever:
    """Fixture providing a retriever populated with representative curriculum chunks."""
    store_dir = str(tmp_path / "chroma_retriever_test")
    vector_store = ChromaVectorStore(persist_directory=store_dir, collection_name="test_retriever")

    sample_chunks = [
        TextChunk(
            chunk_id="chunk_asgn1",
            doc_name="curriculum.pdf",
            page_numbers=[3],
            text="ASSIGNMENT 01: DO I EVEN NEED RAG? The goal is to learn to recognize the problem RAG solves before touching a vector database. Compare LLM-only baseline vs RAG.",
            token_count=35,
            char_count=180,
            metadata={"section_heading": "ASSIGNMENT 01"},
        ),
        TextChunk(
            chunk_id="chunk_asgn2",
            doc_name="curriculum.pdf",
            page_numbers=[5],
            text="ASSIGNMENT 02: LANGCHAIN: ABSTRACTION OR DEPENDENCY TRAP? Compare direct API calls with LangChain implementation on maintainability and latency.",
            token_count=30,
            char_count=160,
            metadata={"section_heading": "ASSIGNMENT 02"},
        ),
        TextChunk(
            chunk_id="chunk_asgn5",
            doc_name="curriculum.pdf",
            page_numbers=[11],
            text="ASSIGNMENT 05: BUILD A PRODUCTION AI PRODUCT. Architect an end-to-end production AI system with evaluation, incident reports, and guardrails.",
            token_count=30,
            char_count=160,
            metadata={"section_heading": "ASSIGNMENT 05"},
        ),
    ]

    vecs = embedding_service.embed_texts([c.text for c in sample_chunks])
    vector_store.add_chunks(sample_chunks, vecs)

    return SemanticRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
        default_top_k=2,
        min_similarity_threshold=0.30,
    )


def test_retriever_empty_query(populated_retriever: SemanticRetriever):
    """Verifies that whitespace or empty query returns empty result with appropriate flag."""
    result = populated_retriever.retrieve("   ")
    assert result.is_empty_query
    assert len(result.chunks) == 0
    assert "Reason: Question was empty" in result.debug_summary()


def test_retriever_empty_index(tmp_path: Path, embedding_service: EmbeddingService):
    """Verifies that searching an empty index cleanly returns empty result with is_empty_index flag."""
    empty_store = ChromaVectorStore(
        persist_directory=str(tmp_path / "empty_store"),
        collection_name="empty_col",
    )
    retriever = SemanticRetriever(empty_store, embedding_service)
    result = retriever.retrieve("What is RAG?")

    assert result.is_empty_index
    assert len(result.chunks) == 0
    assert "Vector database is empty" in result.debug_summary()


def test_retriever_targeted_query(populated_retriever: SemanticRetriever):
    """Verifies that query about Assignment 01 retrieves chunk_asgn1 within top-k chunks."""
    result = populated_retriever.retrieve("Do I even need RAG in Assignment 01?", top_k=2)

    assert not result.is_empty_query
    assert not result.is_empty_index
    assert len(result.chunks) == 2
    assert result.has_confident_results

    # Verify chunk_asgn1 is retrieved within top results
    chunk_ids = [c.chunk_id for c in result.chunks]
    assert "chunk_asgn1" in chunk_ids

    asgn1_chunk = next(c for c in result.chunks if c.chunk_id == "chunk_asgn1")
    assert asgn1_chunk.primary_page == 3
    assert "DO I EVEN NEED RAG" in asgn1_chunk.text
    assert asgn1_chunk.similarity > 0.35


def test_retriever_top_k_parameter(populated_retriever: SemanticRetriever):
    """Verifies top_k limits the number of returned chunks."""
    result_k1 = populated_retriever.retrieve("LangChain comparison", top_k=1)
    assert len(result_k1.chunks) == 1

    result_k3 = populated_retriever.retrieve("LangChain comparison", top_k=3)
    assert len(result_k3.chunks) == 3


def test_retriever_unrelated_query_confidence(populated_retriever: SemanticRetriever):
    """Verifies that a completely irrelevant query has lower similarity."""
    result = populated_retriever.retrieve("How do I bake sourdough bread with olive oil?", top_k=1)

    assert len(result.chunks) == 1
    # Check that unrelated query has significantly lower similarity than targeted query
    assert result.chunks[0].similarity < 0.35


def test_retriever_debug_summary(populated_retriever: SemanticRetriever):
    """Verifies that debug_summary produces readable diagnostic information."""
    result = populated_retriever.retrieve("Explain production architecture", top_k=2)
    summary = result.debug_summary()

    assert "Query: 'Explain production architecture'" in summary
    assert "Retrieved: 2 chunk(s)" in summary
    assert "Similarity:" in summary
    assert "geometric topic proximity" in summary
