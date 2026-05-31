"""RAG Pipeline orchestration — chunk → retrieve → rerank → compress."""

from __future__ import annotations

from typing import Callable, Coroutine

from .chunking.base import Chunk
from .chunking.adaptive import AdaptiveChunker
from .chunking.semantic import SemanticChunker
from .compression.context_aware import ContextAwareCompressor
from .config import RAGConfig
from .retrieval.hybrid import HybridRetriever
from .retrieval.query_rewriter import QueryRewriter
from .reranking.embedding_rerank import EmbeddingReranker


class RAGPipeline:
    """Orchestrate the full RAG pipeline.

    Flow:
        documents → [Chunker] → chunks → [QueryRewriter] → queries
        → [HybridRetriever] → retrieved → [Reranker] → reranked
        → [Compressor] → context_string
    """

    def __init__(
        self,
        config: RAGConfig,
        embeddings,
        llm_func: Callable[..., Coroutine] | None = None,
    ):
        self.config = config
        self.embeddings = embeddings
        self.llm_func = llm_func

        self.chunker = self._build_chunker()
        self.query_rewriter = self._build_query_rewriter()
        self.retriever = self._build_retriever()
        self.reranker = self._build_reranker()
        self.compressor = self._build_compressor()

    async def process(self, query: str, documents: list[dict]) -> str:
        """Run the full RAG pipeline on documents for the given query."""
        if not documents:
            return ""

        chunks = await self.chunker.chunk(documents)
        if not chunks:
            return ""

        if self.config.enable_query_rewrite and self.query_rewriter:
            queries = await self.query_rewriter.rewrite(query)
        else:
            queries = [query]

        all_retrieved: list[Chunk] = []
        for q in queries:
            query_chunks = self._clone_chunks(chunks)
            retrieved = await self.retriever.retrieve(q, query_chunks)
            all_retrieved.extend(retrieved)

        deduped = self._deduplicate_by_content(all_retrieved)
        reranked = await self.reranker.rerank(query, deduped)
        context = await self.compressor.compress(query, reranked)
        return context

    def _build_chunker(self):
        match self.config.chunking_strategy:
            case "semantic":
                return SemanticChunker(embeddings=self.embeddings)
            case "adaptive":
                return AdaptiveChunker(
                    default_chunk_size=self.config.chunk_size,
                    default_overlap=self.config.chunk_overlap,
                )
            case _:
                return AdaptiveChunker()

    def _build_query_rewriter(self):
        if not self.config.enable_query_rewrite:
            return None
        return QueryRewriter(
            strategy=self.config.query_rewrite_strategy,
            llm_func=self.llm_func,
            min_retrieval_results=self.config.min_retrieval_results,
            min_top_score=self.config.min_top_score,
        )

    def _build_retriever(self):
        if not self.config.hybrid_search:
            return HybridRetriever(
                embeddings=self.embeddings,
                top_k=self.config.rerank_top_k * 2,
                fusion_mode="weighted",
                bm25_weight=0.0,
                vector_weight=1.0,
            )
        return HybridRetriever(
            embeddings=self.embeddings,
            top_k=self.config.rerank_top_k * 2,
            fusion_mode=self.config.fusion_mode,
            rrf_k=self.config.rrf_k,
            bm25_weight=self.config.bm25_weight,
            vector_weight=self.config.vector_weight,
        )

    def _build_reranker(self):
        match self.config.reranker:
            case "cross_encoder":
                from .reranking.cross_encoder import CrossEncoderReranker
                return CrossEncoderReranker(top_k=self.config.rerank_top_k)
            case _:
                return EmbeddingReranker(
                    embeddings=self.embeddings,
                    top_k=self.config.rerank_top_k,
                )

    def _build_compressor(self):
        return ContextAwareCompressor(
            embeddings=self.embeddings,
            similarity_threshold=self.config.similarity_threshold,
            dedup_threshold=self.config.dedup_threshold,
            max_results=self.config.max_results,
        )

    @staticmethod
    def _clone_chunks(chunks: list[Chunk]) -> list[Chunk]:
        return [
            Chunk(
                content=c.content,
                metadata=dict(c.metadata),
                vector_score=c.vector_score,
                bm25_score=c.bm25_score,
                hybrid_score=c.hybrid_score,
                rerank_score=c.rerank_score,
            )
            for c in chunks
        ]

    @staticmethod
    def _deduplicate_by_content(chunks: list[Chunk]) -> list[Chunk]:
        seen: set[str] = set()
        result: list[Chunk] = []
        for c in chunks:
            if c.content not in seen:
                seen.add(c.content)
                result.append(c)
        return result
