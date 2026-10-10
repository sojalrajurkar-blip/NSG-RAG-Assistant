"""Persistent Vector Store Module for NSG RAG Assistant.

Uses ChromaDB to persist chunk embeddings, text content, and metadata locally.
Supports index inspection, upsert deduplication, and similarity queries.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings

from src.chunker import TextChunk


class ChromaVectorStore:
    """Manages persistent ChromaDB vector index and operations."""

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "nsg_academy_knowledge",
    ):
        """Initializes ChromaDB persistent client.

        Args:
            persist_directory: Folder path where SQLite & vector indices are stored.
            collection_name: Name of ChromaDB collection.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Ensure persistence directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection with cosine similarity space
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: List[TextChunk],
        embeddings: List[List[float]],
    ) -> int:
        """Stores or updates chunks and their corresponding embeddings in the vector index.

        Uses upsert to prevent accidental duplicates when documents are re-indexed.

        Args:
            chunks: List of TextChunk objects.
            embeddings: Parallel list of vector embeddings.

        Returns:
            Number of chunks upserted.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks provided with {len(embeddings)} embeddings."
            )

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)

            # ChromaDB requires metadata values to be primitive types: str, int, float, bool
            first_page = chunk.page_numbers[0] if chunk.page_numbers else 1
            pages_str = ",".join(str(p) for p in chunk.page_numbers)
            section_heading = str(chunk.metadata.get("section_heading") or "")

            sanitized_metadata = {
                "doc_name": chunk.doc_name,
                "primary_page": first_page,
                "pages": pages_str,
                "token_count": chunk.token_count,
                "char_count": chunk.char_count,
                "section_heading": section_heading,
            }
            metadatas.append(sanitized_metadata)

        # Upsert ensures that existing chunks are updated, avoiding duplicate entries
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(ids)

    def count(self) -> int:
        """Returns total number of chunks currently stored in the collection."""
        return self.collection.count()

    def get_sample_chunks(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Returns sample stored chunks and metadata for inspection."""
        count = self.count()
        if count == 0:
            return []

        limit = min(limit, count)
        result = self.collection.peek(limit=limit)

        samples: List[Dict[str, Any]] = []
        if result and "ids" in result and result["ids"]:
            for i in range(len(result["ids"])):
                meta = result["metadatas"][i] if result.get("metadatas") else {}
                doc = result["documents"][i] if result.get("documents") else ""
                chunk_id = result["ids"][i]
                samples.append({
                    "chunk_id": chunk_id,
                    "document": doc,
                    "metadata": meta,
                })
        return samples

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Searches index for nearest chunks matching query embedding.

        Args:
            query_embedding: Vector of query question.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of result dicts containing id, text, metadata, and distance.
        """
        if self.count() == 0:
            return []

        top_k = min(top_k, self.count())
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[Dict[str, Any]] = []
        if results and "ids" in results and results["ids"] and results["ids"][0]:
            num_found = len(results["ids"][0])
            for i in range(num_found):
                cid = results["ids"][0][i]
                doc = results["documents"][0][i] if results.get("documents") else ""
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                distance = results["distances"][0][i] if results.get("distances") else 0.0

                # In cosine space, cosine similarity = 1 - distance
                similarity = round(1.0 - float(distance), 4)

                retrieved.append({
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": meta,
                    "distance": round(float(distance), 4),
                    "similarity": similarity,
                })

        return retrieved

    def reset_index(self) -> None:
        """Deletes collection to cleanly rebuild index when documents change."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
