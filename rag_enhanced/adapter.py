"""Adapter to integrate rag_enhanced with the existing GPTResearcher."""

from __future__ import annotations

from .config import RAGConfig
from .knowledge_store.chroma_store import ChromaKnowledgeStore
from .pipeline import RAGPipeline


class RAGAdapter:
    """Drop-in replacement for ContextManager's compression methods.

    Optionally creates a KnowledgeStore for persistent document storage
    when enable_knowledge_store is True in config.

    Usage:
        from rag_enhanced.adapter import RAGAdapter
        researcher.context_manager = RAGAdapter(researcher)
    """

    def __init__(self, researcher, config: RAGConfig | None = None):
        self.researcher = researcher
        self.config = config or RAGConfig.from_researcher(researcher)

        # Optionally create knowledge store
        self.knowledge_store = self._build_knowledge_store()

        self.pipeline = RAGPipeline(
            config=self.config,
            embeddings=researcher.memory.get_embeddings(),
            llm_func=self._make_llm_func(),
            knowledge_store=self.knowledge_store,
        )

    def _build_knowledge_store(self):
        """Create a KnowledgeStore if enabled in config."""
        if not self.config.enable_knowledge_store:
            return None
        return ChromaKnowledgeStore(
            embeddings=self.researcher.memory.get_embeddings(),
            collection_name=self.config.knowledge_collection,
            persist_directory=self.config.knowledge_store_path,
        )

    async def get_similar_content_by_query(self, query: str, pages: list) -> str:
        """Compatible with ContextManager.get_similar_content_by_query."""
        return await self.pipeline.process(query, pages)

    async def get_similar_content_by_query_with_vectorstore(self, query: str, filter: dict | None) -> str:
        """VectorStore path — delegates to original VectorstoreCompressor."""
        from gpt_researcher.context.compression import VectorstoreCompressor
        from gpt_researcher.actions.utils import stream_output

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "fetching_query_format",
                f"Getting relevant content from vectorstore: {query}...",
                self.researcher.websocket,
            )
        vectorstore_compressor = VectorstoreCompressor(
            self.researcher.vector_store,
            filter=filter,
            prompt_family=self.researcher.prompt_family,
            **self.researcher.kwargs,
        )
        return await vectorstore_compressor.async_get_context(query=query, max_results=8)

    async def ingest_local_docs(self, doc_path: str | None = None) -> int:
        """Load local documents into the knowledge store.

        Args:
            doc_path: Path to file or directory. Defaults to researcher's doc_path.

        Returns:
            Number of chunks stored.

        Raises:
            RuntimeError: If knowledge store is not enabled.
        """
        if not self.knowledge_store:
            raise RuntimeError(
                "Knowledge store is not enabled. "
                "Set enable_knowledge_store=True in RAGConfig."
            )
        path = doc_path or self.researcher.cfg.doc_path
        return await self.knowledge_store.ingest_local_docs(path)

    def _make_llm_func(self):
        """Create an LLM function from the researcher's config."""
        if not self.config.llm_model or not self.config.llm_provider:
            return None

        async def llm_func(messages, **kwargs):
            from gpt_researcher.utils.llm import create_chat_completion
            return await create_chat_completion(
                model=self.config.llm_model,
                messages=messages,
                llm_provider=self.config.llm_provider,
                **kwargs,
            )

        return llm_func
