"""RAG Pipeline orchestration."""

from .config import RAGConfig


class RAGPipeline:
    """Orchestrates the full RAG pipeline: chunk → retrieve → rerank → compress."""

    def __init__(self, config: RAGConfig, embeddings=None):
        self.config = config
        self.embeddings = embeddings

    async def process(self, query: str, documents: list[dict]) -> str:
        raise NotImplementedError
