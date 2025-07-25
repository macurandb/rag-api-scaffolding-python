"""Output formatters for RAG responses."""

from typing import Any

from ...core.domain.rag import RAGResponse
from ...core.ports.services import FormatterService


class JSONResponseFormatter(FormatterService):
    """Formats RAG responses as JSON."""

    async def format_response(self, response: RAGResponse) -> dict[str, Any]:
        """Format RAG response as JSON."""
        return {
            "query": {
                "original_text": response.query.original_text,
                "processed_text": response.query.processed_text,
                "type": response.query.query_type.value,
                "intent": response.query.intent,
                "entities": response.query.entities,
                "keywords": response.query.keywords,
            },
            "answer": response.answer,
            "sources": [
                {
                    "id": result.chunk.id,
                    "content": (
                        result.chunk.content[:200] + "..."
                        if len(result.chunk.content) > 200
                        else result.chunk.content
                    ),
                    "metadata": result.chunk.metadata,
                    "relevance_score": result.relevance_score,
                    "retrieval_method": result.retrieval_method,
                    "rank": result.rank,
                }
                for result in response.sources
            ],
            "generation": {
                "confidence_score": response.generation_result.confidence_score,
                "tokens_used": response.generation_result.tokens_used,
                "model_used": response.generation_result.model_used,
                "generation_time": response.generation_result.generation_time,
            },
            "metadata": {
                **response.processing_metadata,
                "timestamp": response.timestamp.isoformat(),
                "source_count": len(response.sources),
                "total_sources": response.total_sources,
            },
        }
