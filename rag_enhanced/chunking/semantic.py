"""Semantic chunking strategy."""

from .base import BaseChunker, Chunk


class SemanticChunker(BaseChunker):
    async def chunk(self, documents: list[dict]) -> list[Chunk]:
        raise NotImplementedError
