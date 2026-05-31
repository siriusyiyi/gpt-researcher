"""LLM-driven query rewriting and expansion."""

from __future__ import annotations

import re
from typing import Callable, Coroutine

from ..chunking.base import Chunk

_MULTI_QUERY_PROMPT = """You are a helpful assistant that generates multiple search queries.
Given the original query, generate 2-3 alternative search queries that would help find relevant information.
Return ONLY a bulleted list, one query per line, prefixed with "- ".

Original query: {query}

Alternative queries:"""

_HYDE_PROMPT = """You are a helpful assistant. Write a brief, informative paragraph (3-5 sentences) that answers the following question. Even if you are not sure, provide a plausible answer based on general knowledge. This will be used as a search query embedding.

Question: {query}

Answer:"""


class QueryRewriter:
    """Rewrite or expand queries using LLM.

    Strategies:
        - "multi": Generate multiple alternative queries (default).
        - "hyde": Generate a hypothetical answer for embedding-based retrieval.
        - "auto": Use raw query first; expand only if initial results are insufficient.
        - None: Passthrough — return original query unchanged.
    """

    def __init__(
        self,
        strategy: str | None = "multi",
        llm_func: Callable[..., Coroutine] | None = None,
        min_retrieval_results: int = 3,
        min_top_score: float = 0.3,
    ):
        self.strategy = strategy
        self.llm_func = llm_func
        self.min_retrieval_results = min_retrieval_results
        self.min_top_score = min_top_score

    async def rewrite(self, query: str, initial_results: list[Chunk] | None = None) -> list[str]:
        """Expand or rewrite the query.

        Args:
            query: The original query string.
            initial_results: Optional results from the first retrieval pass
                             (used by "auto" mode to decide if expansion is needed).

        Returns:
            List of query strings to use for retrieval.
        """
        if self.strategy is None:
            return [query]

        if self.strategy == "auto":
            if initial_results is None or self._is_insufficient(initial_results):
                return await self._expand_multi(query)
            return [query]

        if self.strategy == "multi":
            return await self._expand_multi(query)

        if self.strategy == "hyde":
            hyde_answer = await self._generate_hyde(query)
            return [query, hyde_answer]

        return [query]

    def _is_insufficient(self, results: list[Chunk]) -> bool:
        """Check if initial retrieval results are insufficient for auto mode."""
        if len(results) < self.min_retrieval_results:
            return True
        top_score = max((c.vector_score for c in results), default=0.0)
        return top_score < self.min_top_score

    async def _expand_multi(self, query: str) -> list[str]:
        """Generate multi-query expansions using LLM."""
        if not self.llm_func:
            return [query]

        prompt = _MULTI_QUERY_PROMPT.format(query=query)
        response = await self.llm_func(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=500,
        )
        expanded = self._parse_bullet_list(response)
        return [query] + expanded

    async def _generate_hyde(self, query: str) -> str:
        """Generate a hypothetical answer for HyDE retrieval."""
        if not self.llm_func:
            return query

        prompt = _HYDE_PROMPT.format(query=query)
        response = await self.llm_func(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300,
        )
        return response.strip()

    @staticmethod
    def _parse_bullet_list(text: str) -> list[str]:
        """Parse a bullet-point list from LLM output."""
        lines = text.strip().split("\n")
        queries = []
        for line in lines:
            cleaned = re.sub(r'^[\s\-\*\d.]+', '', line).strip()
            cleaned = cleaned.strip('"\'""''')
            if cleaned:
                queries.append(cleaned)
        return queries
