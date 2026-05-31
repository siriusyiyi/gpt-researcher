"""Lightweight RAG query tool for knowledge-base Q&A and retrieval.

Usage — standalone tool functions (quick start):
    from rag_enhanced.tools import rag_query, knowledge_search, knowledge_add

    # Add documents
    count = await knowledge_add([{"raw_content": "...", "url": "..."}])

    # Pure retrieval (no LLM)
    chunks = await knowledge_search("query", top_k=5)

    # Full Q&A (with LLM answer generation)
    result = await rag_query("What is Python?")
    print(result["answer"])

Usage — RAGQueryTool class (more control):
    tool = RAGQueryTool(knowledge_store=store, embeddings=emb, llm_func=llm)
    result = await tool.query("question")
"""

from __future__ import annotations

import logging
from typing import Callable, Coroutine, Optional

from ..chunking.adaptive import AdaptiveChunker
from ..chunking.base import Chunk
from ..compression.context_aware import ContextAwareCompressor
from ..config import RAGConfig
from ..knowledge_store.base import BaseKnowledgeStore
from ..knowledge_store.chroma_store import ChromaKnowledgeStore
from ..retrieval.query_rewriter import QueryRewriter
from ..reranking.embedding_rerank import EmbeddingReranker

logger = logging.getLogger(__name__)

_ANSWER_PROMPT = """\
Based on the following context, answer the question. Cite sources using [Source N] format where N is the source number.
If the context doesn't contain enough information to answer the question, say so explicitly.

Context:
{context}

Question: {question}

Answer:"""


class RAGQueryTool:
    """Lightweight RAG Q&A tool backed by a knowledge store.

    Pipeline:
        query → [QueryRewriter] → [KnowledgeStore.retrieve] → [Reranker]
              → [Compressor] → [LLM answer] → {answer, sources, chunks}

    Can be used without llm_func for pure retrieval (no answer generation).

    Args:
        knowledge_store: The persistent knowledge store to query.
        embeddings: LangChain-compatible embeddings instance.
        llm_func: Optional async LLM function for answer generation.
            Signature: async def llm_func(messages, **kwargs) -> str
        config: Optional RAGConfig. Defaults are suitable for Q&A.
    """

    def __init__(
        self,
        knowledge_store: BaseKnowledgeStore,
        embeddings,
        llm_func: Optional[Callable[..., Coroutine]] = None,
        config: Optional[RAGConfig] = None,
    ):
        self.knowledge_store = knowledge_store
        self.embeddings = embeddings
        self.llm_func = llm_func
        self.config = config or RAGConfig(
            enable_query_rewrite=True,
            query_rewrite_strategy="multi",
            knowledge_store_mode="primary",
        )

        self.query_rewriter = self._build_query_rewriter()
        self.reranker = EmbeddingReranker(
            embeddings=embeddings,
            top_k=self.config.rerank_top_k,
        )
        self.compressor = ContextAwareCompressor(
            embeddings=embeddings,
            similarity_threshold=self.config.similarity_threshold,
            dedup_threshold=self.config.dedup_threshold,
            max_results=self.config.max_results,
        )

    def _build_query_rewriter(self):
        if not self.config.enable_query_rewrite or not self.llm_func:
            return None
        return QueryRewriter(
            strategy=self.config.query_rewrite_strategy,
            llm_func=self.llm_func,
            min_retrieval_results=self.config.min_retrieval_results,
            min_top_score=self.config.min_top_score,
        )

    async def query(self, question: str) -> dict:
        """Run the full Q&A pipeline.

        Args:
            question: The user's question.

        Returns:
            dict with keys:
                answer: str — LLM-generated answer (empty string if no llm_func)
                sources: list[dict] — unique sources used, each with {source, title}
                chunks: list[dict] — raw chunks with {content, source, score}
                context: str — formatted context string
        """
        # 1. Query rewriting (optional)
        if self.query_rewriter:
            queries = await self.query_rewriter.rewrite(question)
        else:
            queries = [question]

        # 2. Retrieve from knowledge store for each query variant
        all_chunks: list[Chunk] = []
        for q in queries:
            retrieved = await self.knowledge_store.retrieve(
                q, top_k=self.config.rerank_top_k * 2,
            )
            all_chunks.extend(retrieved)

        if not all_chunks:
            return {
                "answer": "",
                "sources": [],
                "chunks": [],
                "context": "",
            }

        # 3. Deduplicate by content
        seen: set[str] = set()
        deduped: list[Chunk] = []
        for c in all_chunks:
            if c.content not in seen:
                seen.add(c.content)
                deduped.append(c)

        # 4. Rerank
        reranked = await self.reranker.rerank(question, deduped)

        # 5. Compress (get formatted context)
        context = await self.compressor.compress(question, reranked)

        # 6. Extract unique sources
        sources = self._extract_sources(reranked)

        # 7. Build chunk summaries
        chunk_summaries = [
            {
                "content": c.content[:200] + ("..." if len(c.content) > 200 else ""),
                "source": c.metadata.get("source", "unknown"),
                "score": c.rerank_score or c.vector_score,
            }
            for c in reranked[: self.config.max_results]
        ]

        # 8. Generate answer via LLM (optional)
        answer = ""
        if self.llm_func and context:
            answer = await self._generate_answer(question, context)

        return {
            "answer": answer,
            "sources": sources,
            "chunks": chunk_summaries,
            "context": context,
        }

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Pure retrieval without LLM answer generation.

        Args:
            query: Search query.
            top_k: Maximum results.

        Returns:
            List of dicts with {content, source, score}.
        """
        chunks = await self.knowledge_store.retrieve(query, top_k=top_k)
        return [
            {
                "content": c.content,
                "source": c.metadata.get("source", "unknown"),
                "score": c.vector_score,
            }
            for c in chunks
        ]

    async def _generate_answer(self, question: str, context: str) -> str:
        """Use LLM to generate an answer from context."""
        messages = [
            {"role": "user", "content": _ANSWER_PROMPT.format(
                context=context, question=question,
            )},
        ]
        return await self.llm_func(messages)

    @staticmethod
    def _extract_sources(chunks: list[Chunk]) -> list[dict]:
        """Extract unique sources from chunks."""
        seen: set[str] = set()
        sources: list[dict] = []
        for c in chunks:
            source = c.metadata.get("source", "")
            if source and source not in seen:
                seen.add(source)
                sources.append({
                    "source": source,
                    "title": c.metadata.get("title", ""),
                })
        return sources


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

# Global store registry: store_path → ChromaKnowledgeStore
_store_cache: dict[str, ChromaKnowledgeStore] = {}


def _get_or_create_store(
    store_path: str,
    embeddings=None,
    collection_name: str = "default",
) -> ChromaKnowledgeStore:
    """Get a cached or new ChromaKnowledgeStore."""
    cache_key = f"{store_path}:{collection_name}"
    if cache_key not in _store_cache:
        if embeddings is None:
            embeddings = _default_embeddings()
        _store_cache[cache_key] = ChromaKnowledgeStore(
            embeddings=embeddings,
            collection_name=collection_name,
            persist_directory=store_path,
        )
    return _store_cache[cache_key]


def _default_embeddings():
    """Create default OpenAI embeddings for convenience functions."""
    import os
    from langchain_openai import OpenAIEmbeddings

    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    kwargs = {"model": model, "check_embedding_ctx_length": False}
    if base_url:
        kwargs["openai_api_base"] = base_url
    return OpenAIEmbeddings(**kwargs)


async def rag_query(
    question: str,
    store_path: str = "./knowledge_store",
    collection: str = "default",
    llm_func: Optional[Callable[..., Coroutine]] = None,
    embeddings=None,
) -> dict:
    """Ask a question against the knowledge base.

    If llm_func is provided, generates a full answer with citations.
    Otherwise returns retrieval results only.

    Args:
        question: The question to answer.
        store_path: Path to the Chroma knowledge store.
        collection: Chroma collection name.
        llm_func: Optional async LLM function for answer generation.
        embeddings: Optional embeddings instance (defaults to OpenAI).

    Returns:
        dict with {answer, sources, chunks, context}.
    """
    store = _get_or_create_store(store_path, embeddings, collection)
    tool = RAGQueryTool(
        knowledge_store=store,
        embeddings=store.embeddings,
        llm_func=llm_func,
    )
    return await tool.query(question)


async def knowledge_add(
    documents: list[dict],
    store_path: str = "./knowledge_store",
    collection: str = "default",
    embeddings=None,
) -> int:
    """Add documents to the knowledge store.

    Each document should have at least "raw_content" and "url" keys.

    Args:
        documents: List of document dicts with {raw_content, url, title?}.
        store_path: Path to the Chroma knowledge store.
        collection: Chroma collection name.
        embeddings: Optional embeddings instance.

    Returns:
        Number of chunks stored.
    """
    store = _get_or_create_store(store_path, embeddings, collection)
    chunker = AdaptiveChunker()
    chunks = await chunker.chunk(documents)
    if not chunks:
        return 0
    return await store.add_documents(chunks, doc_type="manual")


async def knowledge_search(
    query: str,
    top_k: int = 5,
    store_path: str = "./knowledge_store",
    collection: str = "default",
    embeddings=None,
) -> list[dict]:
    """Pure retrieval from the knowledge store (no LLM).

    Args:
        query: Search query.
        top_k: Maximum number of results.
        store_path: Path to the Chroma knowledge store.
        collection: Chroma collection name.
        embeddings: Optional embeddings instance.

    Returns:
        List of dicts with {content, source, score}.
    """
    store = _get_or_create_store(store_path, embeddings, collection)
    chunks = await store.retrieve(query, top_k=top_k)
    return [
        {
            "content": c.content,
            "source": c.metadata.get("source", "unknown"),
            "score": c.vector_score,
        }
        for c in chunks
    ]
