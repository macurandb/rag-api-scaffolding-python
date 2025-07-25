"""Retrieval service for RAG operations."""

from abc import ABC, abstractmethod
from typing import Any

import structlog

from ...domain.rag import ProcessedQuery, RetrievalResult, RetrievalStrategy

logger = structlog.get_logger()


class RetrievalService(ABC):
    """Abstract service for document retrieval."""

    @abstractmethod
    async def retrieve(
        self,
        query: ProcessedQuery,
        strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY,
        max_results: int = 5,
        min_score: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query."""
        pass

    @abstractmethod
    async def retrieve_by_ids(self, document_ids: list[str]) -> list[RetrievalResult]:
        """Retrieve documents by their IDs."""
        pass

    @abstractmethod
    async def get_similar_documents(
        self, document_id: str, max_results: int = 5
    ) -> list[RetrievalResult]:
        """Get documents similar to a given document."""
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check the health of the retrieval system."""
        pass


class HybridRetrievalService(RetrievalService):
    """Hybrid retrieval combining multiple strategies."""

    def __init__(
        self,
        semantic_retriever: RetrievalService,
        keyword_retriever: RetrievalService | None = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

    async def retrieve(
        self,
        query: ProcessedQuery,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        max_results: int = 5,
        min_score: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve using hybrid approach."""
        results = []

        # Semantic retrieval
        semantic_results = await self.semantic_retriever.retrieve(
            query, RetrievalStrategy.SEMANTIC, max_results, min_score, filters
        )

        # Weight semantic results
        for result in semantic_results:
            result.relevance_score *= self.semantic_weight
            result.retrieval_method = "hybrid_semantic"

        results.extend(semantic_results)

        # Keyword retrieval (if available)
        if self.keyword_retriever:
            keyword_results = await self.keyword_retriever.retrieve(
                query, RetrievalStrategy.KEYWORD, max_results, min_score, filters
            )

            # Weight keyword results
            for result in keyword_results:
                result.relevance_score *= self.keyword_weight
                result.retrieval_method = "hybrid_keyword"

            results.extend(keyword_results)

        # Deduplicate and re-rank
        results = self._deduplicate_and_rerank(results, max_results)

        return results

    def _deduplicate_and_rerank(
        self, results: list[RetrievalResult], max_results: int
    ) -> list[RetrievalResult]:
        """Remove duplicates and re-rank results."""
        seen_ids = set()
        deduplicated = []

        for result in results:
            if result.chunk.id not in seen_ids:
                seen_ids.add(result.chunk.id)
                deduplicated.append(result)

        # Sort by relevance score and limit results
        deduplicated.sort(key=lambda x: x.relevance_score, reverse=True)

        # Update ranks
        for i, result in enumerate(deduplicated[:max_results]):
            result.rank = i + 1

        return deduplicated[:max_results]

    async def retrieve_by_ids(self, document_ids: list[str]) -> list[RetrievalResult]:
        """Retrieve documents by IDs using primary retriever."""
        return await self.semantic_retriever.retrieve_by_ids(document_ids)

    async def get_similar_documents(
        self, document_id: str, max_results: int = 5
    ) -> list[RetrievalResult]:
        """Get similar documents using primary retriever."""
        return await self.semantic_retriever.get_similar_documents(
            document_id, max_results
        )

    async def health_check(self) -> dict[str, Any]:
        """Check health of all retrievers."""
        health = {"hybrid_retrieval": "healthy"}

        semantic_health = await self.semantic_retriever.health_check()
        health["semantic_retriever"] = semantic_health

        if self.keyword_retriever:
            keyword_health = await self.keyword_retriever.health_check()
            health["keyword_retriever"] = keyword_health

        return health
