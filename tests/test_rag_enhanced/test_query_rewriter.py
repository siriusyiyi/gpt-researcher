import pytest

from rag_enhanced.chunking.base import Chunk
from rag_enhanced.retrieval.query_rewriter import QueryRewriter


@pytest.mark.asyncio
async def test_query_rewriter_multi_returns_multiple():
    """Multi-query mode returns the original plus expanded queries."""
    async def fake_llm(messages, **kwargs):
        return '- "expanded query one"\n- "expanded query two"'

    rewriter = QueryRewriter(strategy="multi", llm_func=fake_llm)
    queries = await rewriter.rewrite("original query")
    assert "original query" in queries
    assert len(queries) >= 3


@pytest.mark.asyncio
async def test_query_rewriter_passthrough_when_disabled():
    """When strategy is None, returns the original query as-is."""
    rewriter = QueryRewriter(strategy=None, llm_func=None)
    queries = await rewriter.rewrite("just a query")
    assert queries == ["just a query"]


@pytest.mark.asyncio
async def test_query_rewriter_auto_expand_when_insufficient():
    """Auto mode expands when initial results are below threshold."""
    call_count = 0

    async def fake_llm(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        return '- "expanded one"\n- "expanded two"'

    rewriter = QueryRewriter(
        strategy="auto", llm_func=fake_llm,
        min_retrieval_results=3, min_top_score=0.3,
    )
    initial_chunks = [
        Chunk(content="a", metadata={}, vector_score=0.2),
        Chunk(content="b", metadata={}, vector_score=0.1),
    ]
    queries = await rewriter.rewrite("test", initial_results=initial_chunks)
    assert call_count == 1
    assert len(queries) >= 2


@pytest.mark.asyncio
async def test_query_rewriter_auto_skip_when_sufficient():
    """Auto mode skips expansion when initial results are sufficient."""
    call_count = 0

    async def fake_llm(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        return '- "expanded"'

    rewriter = QueryRewriter(
        strategy="auto", llm_func=fake_llm,
        min_retrieval_results=3, min_top_score=0.3,
    )
    initial_chunks = [Chunk(content=f"c{i}", metadata={}, vector_score=0.8 - i * 0.1) for i in range(5)]
    queries = await rewriter.rewrite("test", initial_results=initial_chunks)
    assert call_count == 0
    assert queries == ["test"]
