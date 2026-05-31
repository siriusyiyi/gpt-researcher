"""Adaptive chunking strategy."""

from .base import BaseChunker, Chunk


class AdaptiveChunker(BaseChunker):
    async def chunk(self, documents: list[dict]) -> list[Chunk]:
        raise NotImplementedError
