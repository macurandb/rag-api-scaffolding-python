"""RAG Pipeline - Main orchestration of RAG components."""

import time
from typing import Any

import structlog

from ...domain.rag import (
    GenerationStrategy,
    ProcessedQuery,
    QueryType,
    RAGResponse,
    RetrievalStrategy,
)
from .context_service import ContextService
from .embedding_service import EmbeddingService
from .generation_service import GenerationService
from .prompt_service import PromptService
from .retrieval_service import RetrievalService

logger = structlog.get_logger()


class RAGPipeline:
    """Main RAG processing pipeline."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        retrieval_service: RetrievalService,
        generation_service: GenerationService,
        prompt_service: PromptService,
        context_service: ContextService,
    ):
        self.embedding_service = embedding_service
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service
        self.prompt_service = prompt_service
        self.context_service = context_service

    async def process(
        self,
        query_text: str,
        query_type: QueryType = QueryType.QUESTION_ANSWERING,
        generation_strategy: GenerationStrategy = GenerationStrategy.STANDARD,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY,
        max_sources: int = 5,
        temperature: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """Process a complete RAG request."""
        start_time = time.time()

        try:
            logger.info(
                "Starting RAG pipeline",
                query=query_text[:100],
                query_type=query_type.value,
                generation_strategy=generation_strategy.value,
            )

            # Step 1: Process query
            processed_query = await self._process_query(
                query_text, query_type, metadata
            )

            # Step 2: Retrieve relevant documents
            retrieval_results = await self.retrieval_service.retrieve(
                processed_query,
                strategy=retrieval_strategy,
                max_results=max_sources,
                min_score=0.1,
            )

            logger.info(
                "Documents retrieved",
                count=len(retrieval_results),
                avg_score=(
                    sum(r.relevance_score for r in retrieval_results)
                    / len(retrieval_results)
                    if retrieval_results
                    else 0
                ),
            )

            # Step 3: Build context
            context = await self.context_service.build_context(
                processed_query, retrieval_results, generation_strategy
            )

            # Step 4: Optimize context for token limits
            context = await self.prompt_service.optimize_context_window(
                context, max_tokens=4000
            )

            # Step 5: Generate response
            generation_result = await self.generation_service.generate(
                context, temperature=temperature
            )

            # Step 6: Build final response
            processing_time = time.time() - start_time

            response = RAGResponse(
                query=processed_query,
                answer=generation_result.generated_text,
                sources=context.retrieved_chunks,
                generation_result=generation_result,
                processing_metadata={
                    "processing_time": processing_time,
                    "retrieval_strategy": retrieval_strategy.value,
                    "generation_strategy": generation_strategy.value,
                    "sources_used": len(context.retrieved_chunks),
                    "total_sources_found": len(retrieval_results),
                },
            )

            logger.info(
                "RAG pipeline completed",
                processing_time=processing_time,
                sources_used=len(context.retrieved_chunks),
                answer_length=len(generation_result.generated_text),
            )

            return response

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "RAG pipeline failed",
                error=str(e),
                processing_time=processing_time,
                query=query_text[:100],
            )
            raise

    async def _process_query(
        self, query_text: str, query_type: QueryType, metadata: dict[str, Any] | None
    ) -> ProcessedQuery:
        """Process raw query into structured format."""
        # Basic query processing - can be enhanced with NLP
        processed_text = query_text.strip()

        # Extract keywords (simple implementation)
        keywords = [
            word.lower()
            for word in processed_text.split()
            if len(word) > 3 and word.isalpha()
        ]

        return ProcessedQuery(
            original_text=query_text,
            processed_text=processed_text,
            query_type=query_type,
            keywords=keywords,
            metadata=metadata or {},
        )

    async def health_check(self) -> dict[str, Any]:
        """Check health of all pipeline components."""
        health_status = {"pipeline": "healthy", "timestamp": time.time()}

        try:
            # Check each service
            services = {
                "embedding_service": self.embedding_service,
                "retrieval_service": self.retrieval_service,
                "generation_service": self.generation_service,
            }

            for service_name, service in services.items():
                try:
                    if hasattr(service, "health_check"):
                        service_health = await service.health_check()
                        health_status[service_name] = service_health
                    else:
                        health_status[service_name] = "no_health_check"
                except Exception as e:
                    health_status[service_name] = {"status": "error", "error": str(e)}

            # Overall status
            failed_services = [
                name
                for name, status in health_status.items()
                if isinstance(status, dict) and status.get("status") == "error"
            ]

            if failed_services:
                health_status["pipeline"] = "degraded"
                health_status["failed_services"] = failed_services

        except Exception as e:
            health_status["pipeline"] = "error"
            health_status["error"] = str(e)

        return health_status


class RAGPipelineBuilder:
    """Builder for creating RAG pipelines with different configurations."""

    def __init__(self):
        self.embedding_service = None
        self.retrieval_service = None
        self.generation_service = None
        self.prompt_service = None
        self.context_service = None

    def with_embedding_service(self, service: EmbeddingService) -> "RAGPipelineBuilder":
        """Set embedding service."""
        self.embedding_service = service
        return self

    def with_retrieval_service(self, service: RetrievalService) -> "RAGPipelineBuilder":
        """Set retrieval service."""
        self.retrieval_service = service
        return self

    def with_generation_service(
        self, service: GenerationService
    ) -> "RAGPipelineBuilder":
        """Set generation service."""
        self.generation_service = service
        return self

    def with_prompt_service(self, service: PromptService) -> "RAGPipelineBuilder":
        """Set prompt service."""
        self.prompt_service = service
        return self

    def with_context_service(self, service: ContextService) -> "RAGPipelineBuilder":
        """Set context service."""
        self.context_service = service
        return self

    def build(self) -> RAGPipeline:
        """Build the RAG pipeline."""
        required_services = [
            ("embedding_service", self.embedding_service),
            ("retrieval_service", self.retrieval_service),
            ("generation_service", self.generation_service),
            ("prompt_service", self.prompt_service),
            ("context_service", self.context_service),
        ]

        missing_services = [
            name for name, service in required_services if service is None
        ]

        if missing_services:
            raise ValueError(f"Missing required services: {missing_services}")

        return RAGPipeline(
            embedding_service=self.embedding_service,
            retrieval_service=self.retrieval_service,
            generation_service=self.generation_service,
            prompt_service=self.prompt_service,
            context_service=self.context_service,
        )
