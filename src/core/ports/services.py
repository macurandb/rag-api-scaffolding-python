"""Service interfaces for RAG application."""

from abc import ABC, abstractmethod
from typing import Any

from ..domain.rag import (
    DocumentChunk,
    ProcessedQuery,
    RAGContext,
    RAGResponse,
    RetrievalResult,
)


class EmbeddingService(ABC):
    """Interface for embedding generation."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for text."""
        pass

    @abstractmethod
    async def embed_query(self, query: ProcessedQuery) -> list[float]:
        """Generate embedding for processed query."""
        pass

    @abstractmethod
    async def embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Generate embeddings for document chunks."""
        pass


class RetrievalService(ABC):
    """Interface for document retrieval."""

    @abstractmethod
    async def retrieve(
        self, query: ProcessedQuery, limit: int = 5
    ) -> list[RetrievalResult]:
        """Retrieve relevant document chunks for a query."""
        pass


class PromptService(ABC):
    """Interface for prompt construction."""

    @abstractmethod
    async def build_prompt(self, context: RAGContext) -> str:
        """Build prompt from query and retrieved documents."""
        pass


class GenerationService(ABC):
    """Interface for text generation."""

    @abstractmethod
    async def generate(self, context: RAGContext) -> str:
        """Generate response from RAG context."""
        pass


class FormatterService(ABC):
    """Interface for response formatting."""

    @abstractmethod
    async def format_response(self, response: RAGResponse) -> dict[str, Any]:
        """Format RAG response for output."""
        pass
