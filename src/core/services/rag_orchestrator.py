"""RAG Orchestrator - Main business logic using new RAG components."""

from typing import Any

import structlog

from ...adapters.output.formatters import JSONResponseFormatter
from ..domain.rag import GenerationStrategy, QueryType
from ..ports.repositories import QueryRepository, ResponseRepository
from .rag.pipeline import RAGPipeline

logger = structlog.get_logger()


class RAGOrchestrator:
    """Orchestrates the RAG process using the new pipeline architecture."""

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        formatter: JSONResponseFormatter | None = None,
        query_repo: QueryRepository | None = None,
        response_repo: ResponseRepository | None = None,
    ):
        self.rag_pipeline = rag_pipeline
        self.formatter = formatter or JSONResponseFormatter()
        self.query_repo = query_repo
        self.response_repo = response_repo

    async def process_query(
        self,
        query_text: str,
        metadata: dict[str, Any] = None,
        query_type: str = "question_answering",
        generation_strategy: str = "standard",
        max_sources: int = 5,
        temperature: float = 0.7,
    ) -> dict:
        """Process a RAG query end-to-end."""
        logger.info("Processing RAG query", query=query_text[:100])

        try:
            # Convert string parameters to enums
            query_type_enum = self._parse_query_type(query_type)
            generation_strategy_enum = self._parse_generation_strategy(
                generation_strategy
            )

            # Process through RAG pipeline
            rag_response = await self.rag_pipeline.process(
                query_text=query_text,
                query_type=query_type_enum,
                generation_strategy=generation_strategy_enum,
                max_sources=max_sources,
                temperature=temperature,
                metadata=metadata,
            )

            # Save query for auditing (optional)
            if self.query_repo:
                await self.query_repo.save_query(rag_response.query)

            # Save response for auditing (optional)
            if self.response_repo:
                await self.response_repo.save_response(rag_response)

            # Format response for API
            formatted_response = await self.formatter.format_response(rag_response)

            logger.info("RAG query processed successfully")
            return formatted_response

        except Exception as e:
            logger.error(
                "Error processing RAG query", error=str(e), query=query_text[:100]
            )
            raise

    async def health_check(self) -> dict[str, Any]:
        """Check health of RAG components."""
        try:
            return await self.rag_pipeline.health_check()
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {"status": "error", "error": str(e)}

    def _parse_query_type(self, query_type: str) -> QueryType:
        """Parse query type string to enum."""
        try:
            return QueryType(query_type.lower())
        except ValueError:
            logger.warning("Invalid query type, using default", query_type=query_type)
            return QueryType.QUESTION_ANSWERING

    def _parse_generation_strategy(self, strategy: str) -> GenerationStrategy:
        """Parse generation strategy string to enum."""
        try:
            return GenerationStrategy(strategy.lower())
        except ValueError:
            logger.warning(
                "Invalid generation strategy, using default", strategy=strategy
            )
            return GenerationStrategy.STANDARD
