"""Abstract base class for knowledge store backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..chunking.base import Chunk


class BaseKnowledgeStore(ABC):
    """Interface for persistent document storage and retrieval.

    Implementations store chunked documents with embeddings and support
    similarity-based retrieval. Metadata (source, title, doc_type) is
    preserved for citation and filtering.
    """

    @abstractmethod
    async def add_documents(self, chunks: list[Chunk], doc_type: str = "local") -> int:
        """Add chunked documents to the knowledge store.

        Args:
            chunks: List of Chunk objects with content and metadata.
            doc_type: Document origin — "local" (file) or "manual" (user-added).

        Returns:
            Number of chunks actually stored.
        """

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10) -> list[Chunk]:
        """Retrieve relevant chunks for a query using similarity search.

        Args:
            query: The search query.
            top_k: Maximum number of chunks to return.

        Returns:
            List of Chunk objects with vector_score populated.
        """

    @abstractmethod
    async def delete(self, source: str) -> int:
        """Delete all chunks from a specific source.

        Args:
            source: The source identifier (URL or filename).

        Returns:
            Number of chunks deleted.
        """

    @abstractmethod
    def list_sources(self) -> list[dict]:
        """List all unique sources in the knowledge store.

        Returns:
            List of dicts with keys: source, title, doc_type, chunk_count.
        """

    @abstractmethod
    async def count(self) -> int:
        """Return total number of chunks in the store."""
