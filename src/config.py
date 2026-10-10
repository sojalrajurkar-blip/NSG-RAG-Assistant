"""Configuration Module for NSG RAG Assistant.

Loads environment variables, validates parameters, and provides secure
access to application settings without exposing secrets.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Automatically load .env from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class ConfigError(Exception):
    """Raised when critical configuration is missing or invalid."""
    pass


@dataclass(frozen=True)
class AppConfig:
    """Immutable application settings container."""
    gemini_api_key: str
    gemini_model_name: str = "gemini-3.5-flash-lite"
    chroma_persist_directory: str = "chroma_db"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    default_chunk_size: int = 500
    default_chunk_overlap: int = 80
    default_top_k: int = 3
    min_similarity_threshold: float = 0.25
    request_timeout_seconds: int = 30

    @property
    def is_api_key_configured(self) -> bool:
        """Checks if a valid, non-placeholder API key is set."""
        return bool(
            self.gemini_api_key
            and self.gemini_api_key.strip()
            and self.gemini_api_key != "your_gemini_api_key_here"
        )

    def masked_api_key(self) -> str:
        """Returns safe masked representation of API key for status displays."""
        if not self.is_api_key_configured:
            return "(Not Configured)"
        k = self.gemini_api_key.strip()
        if len(k) <= 8:
            return "****"
        return f"{k[:4]}...{k[-4:]}"


def load_config() -> AppConfig:
    """Reads settings from environment variables with sensible defaults.

    Returns:
        AppConfig instance.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash").strip()
    chroma_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db").strip()
    emb_model = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2").strip()

    try:
        chunk_size = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
    except ValueError:
        chunk_size = 500

    try:
        chunk_overlap = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "80"))
    except ValueError:
        chunk_overlap = 80

    try:
        top_k = int(os.getenv("DEFAULT_TOP_K", "3"))
    except ValueError:
        top_k = 3

    try:
        timeout = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    except ValueError:
        timeout = 30

    return AppConfig(
        gemini_api_key=api_key,
        gemini_model_name=model_name,
        chroma_persist_directory=chroma_dir,
        embedding_model_name=emb_model,
        default_chunk_size=chunk_size,
        default_chunk_overlap=chunk_overlap,
        default_top_k=top_k,
        request_timeout_seconds=timeout,
    )
