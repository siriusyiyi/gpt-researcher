"""Query rewriting strategies."""


class QueryRewriter:
    async def rewrite(self, query: str) -> list[str]:
        raise NotImplementedError
