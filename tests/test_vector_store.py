"""Tests for embedding service and persistent ChromaDB vector store."""

from pathlib import Path
import pytest

from src.chunker import TextChunk
from src.embeddings import EmbeddingService
from src.vector_store import ChromaVectorStore


@pytest.fixture(scope="module")
def embedding_service() -> EmbeddingService:
    """Module-level fixture to load embedding model once for tests."""
    return EmbeddingService(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def temp_vector_store(tmp_path: Path) -> ChromaVectorStore:
    """Fixture providing an isolated ChromaVectorStore instance in temp directory."""
    store_dir = str(tmp_path / "test_chroma_db")
    return ChromaVectorStore(persist_directory=store_dir, collection_name="test_collection")


def test_embedding_service_dimensions_and_batch(embedding_service: EmbeddingService):
    """Verifies that EmbeddingService generates 384-dimensional normalized vectors."""
    assert embedding_service.dimension == 384

    texts = [
        "What is RAG in AI engineering?",
        "Vector databases store embeddings for similarity search.",
    ]
    vectors = embedding_service.embed_texts(texts)
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384

    # Test query embedding
    q_vec = embedding_service.embed_query("How does indexing work?")
    assert len(q_vec) == 384

    # Test empty batch
    empty_vecs = embedding_service.embed_texts([])
    assert empty_vecs == []


def test_vector_store_add_count_and_deduplication(
    temp_vector_store: ChromaVectorStore,
    embedding_service: EmbeddingService,
):
    """Verifies adding chunks, count accuracy, and upsert deduplication."""
    chunks = [
        TextChunk(
            chunk_id="doc1_p1_c0",
            doc_name="doc1.pdf",
            page_numbers=[1],
            text="Assignment 1 teaches the fundamental concept of RAG.",
            token_count=10,
            char_count=52,
            metadata={"section_heading": "Assignment 1"},
        ),
        TextChunk(
            chunk_id="doc1_p2_c1",
            doc_name="doc1.pdf",
            page_numbers=[2],
            text="Assignment 2 evaluates LangChain abstraction vs direct API.",
            token_count=10,
            char_count=60,
            metadata={"section_heading": "Assignment 2"},
        ),
    ]

    embeddings = embedding_service.embed_texts([c.text for c in chunks])
    added = temp_vector_store.add_chunks(chunks, embeddings)

    assert added == 2
    assert temp_vector_store.count() == 2

    # Verify deduplication: Upserting same chunks must NOT increase count
    added_again = temp_vector_store.add_chunks(chunks, embeddings)
    assert added_again == 2
    assert temp_vector_store.count() == 2, "Count must remain 2 after re-indexing identical IDs"


def test_vector_store_persistence_and_reload(
    tmp_path: Path,
    embedding_service: EmbeddingService,
):
    """Verifies that ChromaDB index persists to disk and reloads upon client recreation."""
    persist_dir = str(tmp_path / "persistent_db")
    col_name = "persistence_test"

    # Step 1: Write to store
    store1 = ChromaVectorStore(persist_directory=persist_dir, collection_name=col_name)
    chunk = TextChunk(
        chunk_id="chunk_persist_1",
        doc_name="notes.pdf",
        page_numbers=[3],
        text="Persistent storage survives application restarts.",
        token_count=8,
        char_count=50,
        metadata={"section_heading": "Persistence"},
    )
    vec = embedding_service.embed_texts([chunk.text])
    store1.add_chunks([chunk], vec)
    assert store1.count() == 1

    # Step 2: Simulate restart by instantiating new client on same path
    store2 = ChromaVectorStore(persist_directory=persist_dir, collection_name=col_name)
    assert store2.count() == 1
    samples = store2.get_sample_chunks(limit=1)
    assert len(samples) == 1
    assert samples[0]["chunk_id"] == "chunk_persist_1"
    assert "survives application restarts" in samples[0]["document"]


def test_vector_store_query(
    temp_vector_store: ChromaVectorStore,
    embedding_service: EmbeddingService,
):
    """Verifies vector search returns relevant chunks with similarity metrics."""
    chunks = [
        TextChunk(
            chunk_id="python_chunk",
            doc_name="tech.pdf",
            page_numbers=[1],
            text="Python is used for backend development and machine learning.",
            token_count=10,
            char_count=63,
            metadata={},
        ),
        TextChunk(
            chunk_id="cooking_chunk",
            doc_name="food.pdf",
            page_numbers=[5],
            text="Baking bread requires flour, water, yeast, and patience.",
            token_count=10,
            char_count=57,
            metadata={},
        ),
    ]
    vecs = embedding_service.embed_texts([c.text for c in chunks])
    temp_vector_store.add_chunks(chunks, vecs)

    # Query about programming
    q_vec = embedding_service.embed_query("What language is used for machine learning?")
    results = temp_vector_store.query(q_vec, top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "python_chunk"
    assert "Python is used" in results[0]["text"]
    assert results[0]["similarity"] > 0.4
