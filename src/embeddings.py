"""Embedding Generation Module for NSG RAG Assistant.

Wraps Sentence Transformers (default: all-MiniLM-L6-v2) to generate dense
vector embeddings for document chunks and user queries locally.
"""

from typing import List, Optional
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Manages local embedding model loading and vector encoding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initializes embedding service.

        Args:
            model_name: HuggingFace model identifier. Defaults to 'all-MiniLM-L6-v2'
                (384 dimensions, compact, fast, local).
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loads the SentenceTransformer model upon first request."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            if hasattr(self._model, "get_embedding_dimension"):
                self._dimension = self._model.get_embedding_dimension()
            else:
                self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        """Returns embedding vector dimension (384 for all-MiniLM-L6-v2)."""
        if self._dimension is None:
            # Trigger lazy load
            _ = self.model
        return self._dimension or 384

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generates embedding vectors for a batch of text chunks.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of items processed per batch.

        Returns:
            List of float vector lists, each of length `dimension`.
        """
        if not texts:
            return []

        # Convert empty strings to placeholder or clean them
        cleaned_texts = [t if t.strip() else " " for t in texts]
        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # Cosine similarity ready
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Generates an embedding vector for a single search question or query.

        Args:
            query: User's question or search query string.

        Returns:
            Single float vector list of length `dimension`.
        """
        cleaned = query.strip()
        if not cleaned:
            # Return zero vector of appropriate dimension for empty queries
            return [0.0] * self.dimension

        embedding = self.model.encode(
            cleaned,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding.tolist()
