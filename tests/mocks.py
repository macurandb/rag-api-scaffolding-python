"""Mock implementations for testing."""

from collections.abc import AsyncGenerator
from typing import Any

from src.core.domain.rag import (
    DocumentChunk,
    GenerationResult,
    ProcessedQuery,
    RAGContext,
    RetrievalResult,
    RetrievalStrategy,
)
from src.core.services.rag.context_service import ContextService
from src.core.services.rag.embedding_service import EmbeddingService
from src.core.services.rag.generation_service import GenerationService
from src.core.services.rag.prompt_service import PromptService
from src.core.services.rag.retrieval_service import RetrievalService


class MockEmbeddingService(EmbeddingService):
    """Mock embedding service for testing."""

    async def embed_query(self, query: ProcessedQuery) -> list[float]:
        """Generate mock embedding for query."""
        return [0.1, 0.2, 0.3, 0.4, 0.5]

    async def embed_text(self, text: str) -> list[float]:
        """Generate mock embedding for text."""
        return [0.1, 0.2, 0.3, 0.4, 0.5]

    async def embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Generate mock embeddings for chunks."""
        for chunk in chunks:
            chunk.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        return chunks

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate mock batch embeddings."""
        return [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in texts]

    def get_embedding_dimension(self) -> int:
        """Get mock embedding dimension."""
        return 5

    def get_model_info(self) -> dict[str, Any]:
        """Get mock model info."""
        return {"model": "mock-embedding-model", "provider": "mock", "dimension": 5}


class MockRetrievalService(RetrievalService):
    """Mock retrieval service for testing."""

    async def retrieve(
        self,
        query: ProcessedQuery,
        strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY,
        max_results: int = 5,
        min_score: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve mock documents."""
        mock_chunks = [
            DocumentChunk(
                id="doc1",
                content="Mock document content 1",
                metadata={"source": "mock1"},
            ),
            DocumentChunk(
                id="doc2",
                content="Mock document content 2",
                metadata={"source": "mock2"},
            ),
        ]

        return [
            RetrievalResult(
                chunk=chunk,
                relevance_score=0.9 - i * 0.1,
                retrieval_method=strategy.value,
                rank=i + 1,
            )
            for i, chunk in enumerate(mock_chunks[:max_results])
        ]

    async def retrieve_by_ids(self, document_ids: list[str]) -> list[RetrievalResult]:
        """Retrieve mock documents by IDs."""
        return []

    async def get_similar_documents(
        self, document_id: str, max_results: int = 5
    ) -> list[RetrievalResult]:
        """Get mock similar documents."""
        return []

    async def health_check(self) -> dict[str, Any]:
        """Mock health check."""
        return {"status": "healthy"}


class MockGenerationService(GenerationService):
    """Mock generation service for testing."""

    async def generate(
        self,
        context: RAGContext,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> GenerationResult:
        """Generate mock response."""
        return GenerationResult(
            generated_text="Mock generated response",
            confidence_score=0.8,
            tokens_used=50,
            model_used="mock-model",
            generation_time=0.5,
            metadata={"temperature": temperature},
        )

    async def generate_stream(
        self,
        context: RAGContext,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate mock streaming response."""
        words = ["Mock", "streaming", "response"]
        for word in words:
            yield word + " "

    async def generate_with_citations(
        self,
        context: RAGContext,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate mock response with citations."""
        return GenerationResult(
            generated_text="Mock response with [Source 1] citations",
            confidence_score=0.8,
            tokens_used=60,
            model_used="mock-model",
            generation_time=0.6,
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get mock model info."""
        return {"model": "mock-generation-model", "provider": "mock"}

    async def health_check(self) -> dict[str, Any]:
        """Mock health check."""
        return {"status": "healthy"}


class MockPromptService(PromptService):
    """Mock prompt service for testing."""

    async def build_prompt(self, context: RAGContext) -> str:
        """Build mock prompt."""
        return f"Mock prompt for query: {context.query.original_text}"

    async def build_system_prompt(self, context: RAGContext) -> str:
        """Build mock system prompt."""
        return "Mock system prompt"

    async def optimize_context_window(
        self, context: RAGContext, max_tokens: int
    ) -> RAGContext:
        """Mock context optimization."""
        return context

    def get_prompt_templates(self) -> dict[str, str]:
        """Get mock templates."""
        return {"default": "Mock template: {query}"}


class MockContextService(ContextService):
    """Mock context service for testing."""

    async def build_context(
        self,
        query: ProcessedQuery,
        retrieved_results: list[RetrievalResult],
        generation_strategy,
        max_context_size: int = 4000,
    ) -> RAGContext:
        """Build mock context."""
        return RAGContext(
            query=query,
            retrieved_chunks=retrieved_results,
            generation_strategy=generation_strategy,
            context_window_size=max_context_size,
            max_chunks=len(retrieved_results),
        )

    async def enrich_context(
        self, context: RAGContext, additional_info: dict[str, Any]
    ) -> RAGContext:
        """Mock context enrichment."""
        return context

    async def validate_context(self, context: RAGContext) -> bool:
        """Mock context validation."""
        return True
