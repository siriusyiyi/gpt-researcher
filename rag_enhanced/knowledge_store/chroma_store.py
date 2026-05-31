"""Chroma-based knowledge store implementation."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from langchain_chroma import Chroma

from ..chunking.base import Chunk
from .base import BaseKnowledgeStore

logger = logging.getLogger(__name__)


class ChromaKnowledgeStore(BaseKnowledgeStore):
    """Persistent knowledge store backed by ChromaDB.

    Stores chunked documents with embeddings in a Chroma collection.
    Supports similarity search, source-based deletion, and ingestion
    from local document directories.

    Args:
        embeddings: LangChain-compatible embeddings instance.
        collection_name: Chroma collection name.
        persist_directory: Directory for Chroma data persistence.
            If None, runs in-memory (no persistence).
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
        )

    async def add_documents(self, chunks: list[Chunk], doc_type: str = "local") -> int:
        """Add chunks to the Chroma collection.

        Each chunk's content becomes the document text, and metadata
        is enriched with doc_type. A unique id is assigned per chunk.
        """
        if not chunks:
            return 0

        texts: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for chunk in chunks:
            texts.append(chunk.content)
            meta = dict(chunk.metadata)
            meta["doc_type"] = doc_type
            # Ensure all metadata values are str/int/float/bool (Chroma requirement)
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

        logger.info("Added %d chunks to knowledge store (doc_type=%s)", len(chunks), doc_type)
        return len(chunks)

    async def retrieve(self, query: str, top_k: int = 10) -> list[Chunk]:
        """Retrieve chunks by similarity search.

        Returns Chunk objects with vector_score populated from Chroma's
        relevance score.
        """
        results = await self._store.asimilarity_search_with_score(
            query=query,
            k=top_k,
        )

        chunks: list[Chunk] = []
        for doc, score in results:
            chunk = Chunk(
                content=doc.page_content,
                metadata=doc.metadata,
                vector_score=1.0 - score,  # Chroma returns L2 distance; convert to similarity
            )
            chunks.append(chunk)

        logger.debug("Retrieved %d chunks for query: %s", len(chunks), query[:50])
        return chunks

    async def delete(self, source: str) -> int:
        """Delete all chunks matching a source.

        Uses Chroma's metadata filtering to find and remove chunks.
        """
        # Query all chunks with matching source to get their IDs
        results = self._store.get(
            where={"source": source},
        )

        ids_to_delete = results.get("ids", [])
        if not ids_to_delete:
            return 0

        self._store._collection.delete(ids=ids_to_delete)
        logger.info("Deleted %d chunks from source: %s", len(ids_to_delete), source)
        return len(ids_to_delete)

    def list_sources(self) -> list[dict]:
        """List unique sources with their metadata and chunk counts."""
        all_data = self._store.get()

        ids = all_data.get("ids", [])
        metadatas = all_data.get("metadatas", [])

        # Group by source
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
        """Load local documents, chunk them, and store in the knowledge base.

        This is a convenience method that:
        1. Loads documents via DocumentLoader
        2. Chunks them with AdaptiveChunker
        3. Stores the chunks

        Args:
            doc_path: Path to a file or directory.

        Returns:
            Number of chunks stored.
        """
        from gpt_researcher.document.document import DocumentLoader

        from ..chunking.adaptive import AdaptiveChunker

        # Load raw documents
        loader = DocumentLoader(doc_path)
        documents = await loader.load()

        if not documents:
            logger.warning("No documents found at: %s", doc_path)
            return 0

        # Chunk documents
        chunker = AdaptiveChunker()
        chunks = await chunker.chunk(documents)

        if not chunks:
            logger.warning("No chunks produced from: %s", doc_path)
            return 0

        # Store in knowledge base
        count = await self.add_documents(chunks, doc_type="local")
        logger.info("Ingested %d chunks from %s", count, doc_path)
        return count
