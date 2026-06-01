"""Chroma-based knowledge store implementation."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from ..chunking.base import Chunk
from .base import BaseKnowledgeStore

logger = logging.getLogger(__name__)

_RRF_K = 60  # RRF constant, matches HybridRetriever default


class ChromaKnowledgeStore(BaseKnowledgeStore):
    """Persistent knowledge store backed by ChromaDB with in-memory BM25.

    Dual retrieval path:
        - Vector search via Chroma (persistent, cosine distance)
        - BM25 keyword search via rank_bm25 (in-memory, rebuilt lazily)
        - Results fused with Reciprocal Rank Fusion (RRF)

    The BM25 index is built lazily on first retrieval and rebuilt
    automatically after any add/delete operation (dirty flag).
    """

    def __init__(
        self,
        embeddings,
        collection_name: str = "default",
        persist_directory: Optional[str] = "./knowledge_store",
    ):
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )

        # In-memory BM25 index (lazy-built, auto-refreshed)
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_docs: list[dict] = []  # parallel array: [{id, text, metadata}]
        self._bm25_dirty: bool = True

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def _invalidate_bm25(self):
        """Mark the BM25 index as stale."""
        self._bm25_dirty = True

    def _ensure_bm25(self):
        """Rebuild the in-memory BM25 index if dirty.

        Loads all documents from Chroma and tokenizes them.
        This is fast (< 100ms for thousands of chunks).
        """
        if not self._bm25_dirty and self._bm25_index is not None:
            return

        all_data = self._store.get()
        ids = all_data.get("ids", [])
        texts = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])

        if not texts:
            self._bm25_index = None
            self._bm25_docs = []
            self._bm25_dirty = False
            return

        self._bm25_docs = [
            {"id": doc_id, "text": text, "metadata": meta or {}}
            for doc_id, text, meta in zip(ids, texts, metadatas)
        ]

        tokenized_corpus = [text.lower().split() for text in texts]
        self._bm25_index = BM25Okapi(tokenized_corpus)
        self._bm25_dirty = False

        logger.debug("BM25 index rebuilt: %d documents", len(self._bm25_docs))

    def _bm25_search(self, query: str) -> list[tuple[int, float]]:
        """Run BM25 search, return (doc_index, score) pairs sorted by score."""
        if self._bm25_index is None:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)
        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )
        return [(idx, float(scores[idx])) for idx in ranked]

    @staticmethod
    def _rrf_fuse(
        bm25_ranked: list[tuple[int, float]],
        vector_ranked: list[tuple[int, float]],
        total_docs: int,
    ) -> dict[int, float]:
        """RRF fusion: combine BM25 and vector ranks.

        Returns dict of doc_index → hybrid_score.
        """
        bm25_rank_map = {idx: rank for rank, (idx, _) in enumerate(bm25_ranked)}
        vec_rank_map = {idx: rank for rank, (idx, _) in enumerate(vector_ranked)}

        scores: dict[int, float] = {}
        for i in range(total_docs):
            rrf_score = 0.0
            if i in bm25_rank_map:
                rrf_score += 1.0 / (_RRF_K + bm25_rank_map[i] + 1)
            if i in vec_rank_map:
                rrf_score += 1.0 / (_RRF_K + vec_rank_map[i] + 1)
            scores[i] = rrf_score

        return scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_documents(self, chunks: list[Chunk], doc_type: str = "local") -> int:
        """Add chunks to the Chroma collection.

        Before adding, any existing chunks with the same source are
        deleted first to prevent duplicates on re-ingestion.
        """
        if not chunks:
            return 0

        # Deduplication: remove old chunks for the same sources
        sources = {c.metadata.get("source") for c in chunks if c.metadata.get("source")}
        for source in sources:
            await self.delete(source)

        texts: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for chunk in chunks:
            texts.append(chunk.content)
            meta = dict(chunk.metadata)
            meta["doc_type"] = doc_type
            sanitized = {}
            for k, v in meta.items():
                if v is None:
                    continue
                sanitized[k] = str(v) if not isinstance(v, (int, float, bool)) else v
            metadatas.append(sanitized)
            ids.append(str(uuid.uuid4()))

        await self._store.aadd_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        self._invalidate_bm25()
        logger.info("Added %d chunks to knowledge store (doc_type=%s)", len(chunks), doc_type)
        return len(chunks)

    async def retrieve(self, query: str, top_k: int = 10) -> list[Chunk]:
        """Hybrid retrieval: Chroma vector search + in-memory BM25, fused via RRF.

        Returns Chunk objects with vector_score, bm25_score, hybrid_score,
        and cached embedding from Chroma for downstream reuse.
        """
        # Ensure BM25 index is fresh
        self._ensure_bm25()

        total = self._store._collection.count()
        if total == 0:
            return []

        # --- Embed query ourselves (ensures consistency + caching) ---
        query_emb = await self.embeddings.aembed_query(query)

        # --- Vector search via Chroma (fetch embeddings too) ---
        n_results = min(top_k * 3, total)
        try:
            raw = self._store._collection.query(
                query_embeddings=[query_emb],
                n_results=n_results,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
        except Exception:
            # Fallback: query without embeddings
            raw = self._store._collection.query(
                query_embeddings=[query_emb],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

        vec_docs = raw.get("documents", [[]])[0]
        vec_dists = raw.get("distances", [[]])[0]
        vec_embs = raw.get("embeddings", [None])[0]

        # Build index mapping: Chroma results → position in _bm25_docs
        text_to_bm25_idx = {
            doc["text"]: i for i, doc in enumerate(self._bm25_docs)
        }

        # Vector ranked list: (bm25_index, vector_score, embedding|None)
        vector_ranked: list[tuple[int, float]] = []
        vec_emb_map: dict[int, list[float]] = {}
        for i, (text, dist) in enumerate(zip(vec_docs, vec_dists)):
            idx = text_to_bm25_idx.get(text)
            if idx is not None:
                vector_ranked.append((idx, 1.0 - dist))
                if vec_embs is not None and i < len(vec_embs) and vec_embs[i] is not None:
                    vec_emb_map[idx] = vec_embs[i]

        # --- BM25 search ---
        bm25_ranked = self._bm25_search(query)

        # --- RRF fusion ---
        fused = self._rrf_fuse(bm25_ranked, vector_ranked, total)

        # Sort by hybrid_score, take top_k
        top_indices = sorted(fused, key=fused.get, reverse=True)[:top_k]

        # Build result chunks
        chunks: list[Chunk] = []
        for idx in top_indices:
            if idx >= len(self._bm25_docs):
                continue
            doc = self._bm25_docs[idx]
            bm25_score = 0.0
            vec_score = 0.0
            for bi, bs in bm25_ranked:
                if bi == idx:
                    bm25_score = bs
                    break
            for vi, vs in vector_ranked:
                if vi == idx:
                    vec_score = vs
                    break

            chunks.append(Chunk(
                content=doc["text"],
                metadata=doc["metadata"],
                vector_score=vec_score,
                bm25_score=bm25_score,
                hybrid_score=fused[idx],
                embedding=vec_emb_map.get(idx),  # cached from Chroma
            ))

        logger.debug("Retrieved %d chunks (hybrid) for query: %s", len(chunks), query[:50])
        return chunks

    async def delete(self, source: str) -> int:
        """Delete all chunks matching a source."""
        results = self._store.get(
            where={"source": source},
        )

        ids_to_delete = results.get("ids", [])
        if not ids_to_delete:
            return 0

        self._store._collection.delete(ids=ids_to_delete)
        self._invalidate_bm25()
        logger.info("Deleted %d chunks from source: %s", len(ids_to_delete), source)
        return len(ids_to_delete)

    def list_sources(self) -> list[dict]:
        """List unique sources with their metadata and chunk counts."""
        all_data = self._store.get()

        metadatas = all_data.get("metadatas", [])

        source_map: dict[str, dict] = {}
        for meta in metadatas:
            source = meta.get("source", "unknown")
            if source not in source_map:
                source_map[source] = {
                    "source": source,
                    "title": meta.get("title", ""),
                    "doc_type": meta.get("doc_type", "unknown"),
                    "chunk_count": 0,
                }
            source_map[source]["chunk_count"] += 1

        return list(source_map.values())

    async def count(self) -> int:
        """Return total number of stored chunks."""
        return self._store._collection.count()

    async def ingest_local_docs(self, doc_path: str) -> int:
        """Load local documents, chunk them, and store in the knowledge base."""
        from gpt_researcher.document.document import DocumentLoader

        from ..chunking.adaptive import AdaptiveChunker

        loader = DocumentLoader(doc_path)
        documents = await loader.load()

        if not documents:
            logger.warning("No documents found at: %s", doc_path)
            return 0

        chunker = AdaptiveChunker()
        chunks = await chunker.chunk(documents)

        if not chunks:
            logger.warning("No chunks produced from: %s", doc_path)
            return 0

        count = await self.add_documents(chunks, doc_type="local")
        logger.info("Ingested %d chunks from %s", count, doc_path)
        return count
