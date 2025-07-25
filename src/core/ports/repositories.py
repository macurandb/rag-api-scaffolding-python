"""Repository interfaces for RAG application."""

from abc import ABC, abstractmethod

from ..domain.rag import DocumentChunk, ProcessedQuery, RAGResponse, RetrievalResult


class DocumentRepository(ABC):
    """Interface for document storage and retrieval."""

    @abstractmethod
    async def save_document_chunk(self, chunk: DocumentChunk) -> str:
        """Save a document chunk and return its ID."""
        pass

    @abstractmethod
    async def find_chunk_by_id(self, chunk_id: str) -> DocumentChunk | None:
        """Find a document chunk by ID."""
        pass

    @abstractmethod
    async def search_similar_chunks(
        self, query: ProcessedQuery, limit: int = 5
    ) -> list[RetrievalResult]:
        """Search for similar document chunks."""
        pass

    @abstractmethod
    async def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a document chunk."""
        pass


class QueryRepository(ABC):
    """Interface for query persistence and analytics."""

    @abstractmethod
    async def save_query(self, query: ProcessedQuery) -> str:
        """Save a processed query and return its ID."""
        pass

    @abstractmethod
    async def find_similar_queries(self, query: ProcessedQuery) -> list[ProcessedQuery]:
        """Find similar queries for optimization."""
        pass

    @abstractmethod
    async def get_query_analytics(self, time_range: str | None = None) -> dict:
        """Get query analytics and patterns."""
        pass


class ResponseRepository(ABC):
    """Interface for response persistence and caching."""

    @abstractmethod
    async def save_response(self, response: RAGResponse) -> str:
        """Save a RAG response and return its ID."""
        pass

    @abstractmethod
    async def find_cached_response(self, query: ProcessedQuery) -> RAGResponse | None:
        """Find a cached response for similar query."""
        pass

    @abstractmethod
    async def get_response_metrics(self, time_range: str | None = None) -> dict:
        """Get response quality and performance metrics."""
        pass
