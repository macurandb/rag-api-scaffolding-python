"""Embedding service for RAG operations."""

from abc import ABC, abstractmethod
from typing import Any

import structlog

from ...domain.rag import DocumentChunk, ProcessedQuery

logger = structlog.get_logger()


class EmbeddingService(ABC):
    """Abstract service for generating embeddings."""

    @abstractmethod
    async def embed_query(self, query: ProcessedQuery) -> list[float]:
        """Generate embedding for a processed query."""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for raw text."""
        pass

    @abstractmethod
    async def embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Generate embeddings for document chunks."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the embedding model."""
        pass


class EmbeddingMetrics:
    """Metrics for embedding operations."""

    def __init__(self):
        self.total_embeddings_generated = 0
        self.total_tokens_processed = 0
        self.average_embedding_time = 0.0
        self.batch_sizes = []

    def record_embedding(
        self, tokens: int, processing_time: float, batch_size: int = 1
    ):
        """Record metrics for an embedding operation."""
        self.total_embeddings_generated += batch_size
        self.total_tokens_processed += tokens
        self.batch_sizes.append(batch_size)

        # Update average processing time
        total_time = self.average_embedding_time * (
            self.total_embeddings_generated - batch_size
        )
        self.average_embedding_time = (
            total_time + processing_time
        ) / self.total_embeddings_generated

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        return {
            "total_embeddings": self.total_embeddings_generated,
            "total_tokens": self.total_tokens_processed,
            "average_time": self.average_embedding_time,
            "average_batch_size": (
                sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
            ),
        }
