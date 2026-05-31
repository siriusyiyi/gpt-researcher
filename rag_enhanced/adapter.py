"""Adapter to integrate rag_enhanced with the existing GPTResearcher."""

from __future__ import annotations

from .config import RAGConfig
from .pipeline import RAGPipeline


class RAGAdapter:
    """Drop-in replacement for ContextManager's compression methods.

    Usage:
        from rag_enhanced.adapter import RAGAdapter
        researcher.context_manager = RAGAdapter(researcher)
    """

    def __init__(self, researcher, config: RAGConfig | None = None):
        self.researcher = researcher
        self.config = config or RAGConfig.from_researcher(researcher)
        self.pipeline = RAGPipeline(
            config=self.config,
            embeddings=researcher.memory.get_embeddings(),
            llm_func=self._make_llm_func(),
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
