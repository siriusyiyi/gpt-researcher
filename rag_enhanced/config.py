"""Configuration for the RAG enhancement pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Unified configuration for all RAG pipeline stages."""

    # Chunking
    chunking_strategy: str = "semantic"       # "semantic" | "adaptive" | "fixed"
    chunk_size: int = 1000
    chunk_overlap: int = 100

    # Retrieval
    enable_query_rewrite: bool = True
    query_rewrite_strategy: str = "multi"     # "multi" | "hyde" | "auto"
    min_retrieval_results: int = 3            # auto mode: minimum chunks to consider sufficient
    min_top_score: float = 0.3                # auto mode: minimum top chunk score
    hybrid_search: bool = True
    fusion_mode: str = "rrf"                  # "rrf" | "weighted"
    rrf_k: int = 60                           # RRF constant (standard value)
    bm25_weight: float = 0.4                  # only used when fusion_mode="weighted"
    vector_weight: float = 0.6                # only used when fusion_mode="weighted"

    # Reranking
    reranker: str = "embedding"               # "embedding" | "cross_encoder"
    rerank_top_k: int = 10

    # Compression
    similarity_threshold: float = 0.35
    dedup_threshold: float = 0.85
    max_results: int = 10

    # LLM (for query rewriting only)
    llm_model: str | None = None
    llm_provider: str | None = None

    @classmethod
    def from_researcher(cls, researcher) -> RAGConfig:
        """Build config from an existing GPTResearcher instance.

        Field mapping:
            researcher.cfg.fast_llm_model    → llm_model
            researcher.cfg.fast_llm_provider → llm_provider
            researcher.cfg.max_search_results_per_query → rerank_top_k
        Embeddings are passed directly to pipeline stages (stateful instance),
        not stored in config.
        """
        cfg = researcher.cfg
        return cls(
            llm_model=cfg.fast_llm_model,
            llm_provider=cfg.fast_llm_provider,
            rerank_top_k=getattr(cfg, "max_search_results_per_query", 10),
        )
