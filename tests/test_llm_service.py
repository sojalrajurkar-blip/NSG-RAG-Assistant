"""Unit and integration tests for LLMService and configuration."""

from unittest.mock import MagicMock, patch
import pytest

from src.config import AppConfig, load_config
from src.retriever import RetrievedChunk
from src.llm_service import (
    LLMService,
    GenerationResult,
    MissingAPIKeyError,
    RAG_SYSTEM_PROMPT,
)


def test_config_masking():
    """Verifies that API keys are properly masked and not leaked."""
    cfg = AppConfig(gemini_api_key="AIzaSyDummyKeyForTestingOnly1234")
    assert cfg.is_api_key_configured
    masked = cfg.masked_api_key()
    assert "AIza" in masked
    assert "1234" in masked
    assert "DummyKeyForTestingOnly" not in masked

    unconfigured = AppConfig(gemini_api_key="your_gemini_api_key_here")
    assert not unconfigured.is_api_key_configured
    assert unconfigured.masked_api_key() == "(Not Configured)"


def test_missing_api_key_error():
    """Verifies that attempting generation without API key raises MissingAPIKeyError."""
    cfg = AppConfig(gemini_api_key="")
    service = LLMService(config=cfg)
    with pytest.raises(MissingAPIKeyError):
        _ = service.client


def test_format_context():
    """Verifies format_context constructs clear, structured excerpts."""
    service = LLMService(config=AppConfig(gemini_api_key="test_key"))
    chunks = [
        RetrievedChunk(
            chunk_id="chunk_1",
            doc_name="syllabus.pdf",
            page_numbers=[3],
            primary_page=3,
            text="Assignment 1 teaches local RAG fundamentals.",
            similarity=0.85,
            distance=0.15,
        ),
        RetrievedChunk(
            chunk_id="chunk_2",
            doc_name="syllabus.pdf",
            page_numbers=[5, 6],
            primary_page=5,
            text="Assignment 2 evaluates LangChain framework dependencies.",
            similarity=0.75,
            distance=0.25,
        ),
    ]

    formatted = service.format_context(chunks)
    assert "syllabus.pdf" in formatted
    assert "Page 3" in formatted
    assert "Page 5, Page 6" in formatted
    assert "Assignment 1 teaches" in formatted


def test_generate_grounded_empty_chunks():
    """Verifies that RAG with empty chunks returns honest fallback without making API calls."""
    service = LLMService(config=AppConfig(gemini_api_key="test_key"))
    res = service.generate_grounded("What is Assignment 1?", chunks=[])
    assert res.mode == "RAG"
    assert "this information is not available" in res.answer.lower()
    assert res.is_fallback


@patch("src.llm_service.genai.Client")
def test_generate_baseline_mocked(mock_client_cls):
    """Verifies baseline generation flow with mocked API response."""
    mock_instance = MagicMock()
    mock_client_cls.return_value = mock_instance

    # Mock response object
    mock_response = MagicMock()
    mock_response.text = "RAG stands for Retrieval-Augmented Generation."
    mock_response.usage_metadata.prompt_token_count = 12
    mock_response.usage_metadata.candidates_token_count = 8
    mock_instance.models.generate_content.return_value = mock_response

    service = LLMService(config=AppConfig(gemini_api_key="dummy_key"))
    service._client = mock_instance

    res = service.generate_baseline("What is RAG?")

    assert res.mode == "LLM-Only"
    assert "Retrieval-Augmented Generation" in res.answer
    assert res.error is None
    assert res.latency_seconds >= 0.0


@patch("src.llm_service.genai.Client")
def test_generate_grounded_mocked(mock_client_cls):
    """Verifies grounded generation flow and citation extraction with mocked response."""
    mock_instance = MagicMock()
    mock_client_cls.return_value = mock_instance

    mock_response = MagicMock()
    mock_response.text = "Assignment 1 focuses on RAG fundamentals [Page 3]."
    mock_response.usage_metadata.prompt_token_count = 45
    mock_response.usage_metadata.candidates_token_count = 15
    mock_instance.models.generate_content.return_value = mock_response

    service = LLMService(config=AppConfig(gemini_api_key="dummy_key"))
    service._client = mock_instance

    chunk = RetrievedChunk(
        chunk_id="chunk_1",
        doc_name="asgn.pdf",
        page_numbers=[3],
        primary_page=3,
        text="Assignment 1 teaches RAG.",
        similarity=0.9,
        distance=0.1,
    )

    res = service.generate_grounded("Explain Assignment 1", chunks=[chunk])

    assert res.mode == "RAG"
    assert "Page 3" in res.citations
    assert "Assignment 1 focuses on RAG" in res.answer
    assert res.error is None


def test_live_gemini_api_call_if_configured():
    """LIVE TEST: Executes 1 controlled call to Gemini API using local .env key."""
    cfg = load_config()
    if not cfg.is_api_key_configured:
        pytest.skip("Gemini API key not configured in .env, skipping live test.")

    service = LLMService(config=cfg)
    # Fast test with small prompt
    res = service.generate_baseline("Say exactly 'Hello NSG' and nothing else.")

    assert res.mode == "LLM-Only"
    assert res.error is None
    assert "hello" in res.answer.lower()
    assert res.latency_seconds > 0.0
