"""Context service for RAG operations."""

from abc import ABC, abstractmethod
from typing import Any

import structlog

from ...domain.rag import (
    GenerationStrategy,
    ProcessedQuery,
    QueryType,
    RAGContext,
    RetrievalResult,
)

logger = structlog.get_logger()


class ContextService(ABC):
    """Abstract service for RAG context management."""

    @abstractmethod
    async def build_context(
        self,
        query: ProcessedQuery,
        retrieved_results: list[RetrievalResult],
        generation_strategy: GenerationStrategy = GenerationStrategy.STANDARD,
        max_context_size: int = 4000,
    ) -> RAGContext:
        """Build RAG context from query and retrieved results."""
        pass

    @abstractmethod
    async def enrich_context(
        self, context: RAGContext, additional_info: dict[str, Any]
    ) -> RAGContext:
        """Enrich context with additional information."""
        pass

    @abstractmethod
    async def validate_context(self, context: RAGContext) -> bool:
        """Validate that context is suitable for generation."""
        pass


class DefaultContextService(ContextService):
    """Default implementation of context service."""

    def __init__(self):
        self.context_strategies = {
            QueryType.QUESTION_ANSWERING: self._build_qa_context,
            QueryType.SUMMARIZATION: self._build_summary_context,
            QueryType.ANALYSIS: self._build_analysis_context,
            QueryType.COMPARISON: self._build_comparison_context,
            QueryType.EXTRACTION: self._build_extraction_context,
        }

    async def build_context(
        self,
        query: ProcessedQuery,
        retrieved_results: list[RetrievalResult],
        generation_strategy: GenerationStrategy = GenerationStrategy.STANDARD,
        max_context_size: int = 4000,
    ) -> RAGContext:
        """Build RAG context from components."""
        try:
            # Apply query-type specific context building
            context_builder = self.context_strategies.get(
                query.query_type, self._build_default_context
            )

            context = await context_builder(
                query, retrieved_results, generation_strategy, max_context_size
            )

            # Validate context
            if not await self.validate_context(context):
                logger.warning("Context validation failed", query=query.original_text)

            return context

        except Exception as e:
            logger.error("Error building context", error=str(e))
            raise

    async def enrich_context(
        self, context: RAGContext, additional_info: dict[str, Any]
    ) -> RAGContext:
        """Enrich context with additional information."""
        # Add to query metadata
        if context.query.metadata is None:
            context.query.metadata = {}

        context.query.metadata.update(additional_info)

        # Update system prompt if needed
        if "system_instructions" in additional_info:
            context.system_prompt = additional_info["system_instructions"]

        # Update generation strategy if specified
        if "generation_strategy" in additional_info:
            try:
                context.generation_strategy = GenerationStrategy(
                    additional_info["generation_strategy"]
                )
            except ValueError:
                logger.warning(
                    "Invalid generation strategy",
                    strategy=additional_info["generation_strategy"],
                )

        return context

    async def validate_context(self, context: RAGContext) -> bool:
        """Validate context for generation."""
        validations = [
            self._validate_query(context.query),
            self._validate_retrieved_results(context.retrieved_chunks),
            self._validate_context_size(context),
        ]

        return all(validations)

    def _validate_query(self, query: ProcessedQuery) -> bool:
        """Validate query component."""
        if not query.original_text or not query.original_text.strip():
            logger.error("Empty query text")
            return False

        if len(query.original_text) > 2000:
            logger.warning("Query text very long", length=len(query.original_text))

        return True

    def _validate_retrieved_results(self, results: list[RetrievalResult]) -> bool:
        """Validate retrieved results."""
        if not results:
            logger.warning("No retrieved results")
            return True  # Not necessarily invalid

        # Check for valid scores
        for result in results:
            if result.relevance_score < 0 or result.relevance_score > 1:
                logger.warning(
                    "Invalid relevance score",
                    score=result.relevance_score,
                    chunk_id=result.chunk.id,
                )

        return True

    def _validate_context_size(self, context: RAGContext) -> bool:
        """Validate context size."""
        total_chars = sum(
            len(result.chunk.content) for result in context.retrieved_chunks
        )

        # Rough token estimation (4 chars ≈ 1 token)
        estimated_tokens = total_chars / 4

        if estimated_tokens > context.context_window_size:
            logger.warning(
                "Context may exceed window size",
                estimated_tokens=estimated_tokens,
                window_size=context.context_window_size,
            )

        return True

    async def _build_default_context(
        self,
        query: ProcessedQuery,
        retrieved_results: list[RetrievalResult],
        generation_strategy: GenerationStrategy,
        max_context_size: int,
    ) -> RAGContext:
        """Build default context."""
        return RAGContext(
            query=query,
            retrieved_chunks=retrieved_results[:5],  # Limit to top 5
            generation_strategy=generation_strategy,
            context_window_size=max_context_size,
            max_chunks=5,
        )

    async def _build_qa_context(
        self,
        query: ProcessedQuery,
        retrieved_results: list[RetrievalResult],
        generation_strategy: GenerationStrategy,
        max_context_size: int,
    ) -> RAGContext:
        """Build context optimized for Q&A."""
        # For Q&A, prioritize high-relevance chunks
        filtered_results = [
            result for result in retrieved_results if result.relevance_score > 0.3
        ]

        return RAGContext(
            query=query,
            retrieved_chunks=filtered_results[:3],  # Fewer, higher quality chunks
            generation_strategy=generation_strategy,
            system_prompt="Answer the question accurately based on the provided context. If the context doesn't contain enough information, say so.",
            context_window_size=max_context_size,
            max_chunks=3,
        )

    async def _build_summary_context(
        self,
        query: ProcessedQuery,
        retrieved_results: list[RetrievalResult],
        generation_strategy: GenerationStrategy,
        max_context_size: int,
    ) -> RAGContext:
        """Build context optimized for summarization."""
        # For summarization, include more diverse sources
        return RAGContext(
            query=query,
            retrieved_chunks=retrieved_results[
                :7
            ],  # More chunks for comprehensive summary
            generation_strategy=generation_strategy,
            system_prompt="Create a comprehensive summary covering all key points from the provided sources.",
            context_window_size=max_context_size,
            max_chunks=7,
        )

    async def _build_analysis_context(
        self,
        query: ProcessedQuery,
        retrieved_results: list[RetrievalResult],
        generation_strategy: GenerationStrategy,
        max_context_size: int,
    ) -> RAGContext:
        """Build context optimized for analysis."""
        return RAGContext(
            query=query,
            retrieved_chunks=retrieved_results[:5],
            generation_strategy=GenerationStrategy.CHAIN_OF_THOUGHT,  # Override for analysis
            system_prompt="Provide detailed analysis with reasoning and evidence from the sources.",
            context_window_size=max_context_size,
            max_chunks=5,
        )

    async def _build_comparison_context(
        self,
        query: ProcessedQuery,
        retrieved_results: list[RetrievalResult],
        generation_strategy: GenerationStrategy,
        max_context_size: int,
    ) -> RAGContext:
        """Build context optimized for comparison."""
        return RAGContext(
            query=query,
            retrieved_chunks=retrieved_results[:6],  # More sources for comparison
            generation_strategy=generation_strategy,
            system_prompt="Compare and contrast the information from different sources, highlighting similarities and differences.",
            context_window_size=max_context_size,
            max_chunks=6,
        )

    async def _build_extraction_context(
        self,
        query: ProcessedQuery,
        retrieved_results: list[RetrievalResult],
        generation_strategy: GenerationStrategy,
        max_context_size: int,
    ) -> RAGContext:
        """Build context optimized for information extraction."""
        # For extraction, prioritize precision over coverage
        high_relevance_results = [
            result for result in retrieved_results if result.relevance_score > 0.5
        ]

        return RAGContext(
            query=query,
            retrieved_chunks=high_relevance_results[:3],
            generation_strategy=GenerationStrategy.STEP_BY_STEP,
            system_prompt="Extract only the specific information requested. Be precise and cite sources.",
            context_window_size=max_context_size,
            max_chunks=3,
        )
