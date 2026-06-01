"""Base classes and data contracts for the chunking stage."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Core data unit flowing through the RAG pipeline.

    Each pipeline stage writes to its own score field so downstream
    stages can inspect multiple dimensions (e.g. compression can
    compare hybrid_score vs rerank_score for dedup decisions).

    The ``embedding`` field caches the vector representation so that
    downstream stages (reranker, compressor) can reuse it without
    making redundant API calls.
    """

    content: str
    metadata: dict = field(default_factory=dict)
    vector_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    embedding: list[float] | None = None


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    async def chunk(self, documents: list[dict]) -> list[Chunk]:
        """Split a list of raw documents into Chunks.

        Args:
            documents: Each dict has at least 'raw_content' (str) and
                       optionally 'url', 'title'.

        Returns:
            List of Chunk objects with metadata preserved.
        """
        ...
