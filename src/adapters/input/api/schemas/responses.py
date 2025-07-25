"""Response schemas for API endpoints."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ResponseStatus(str, Enum):
    """Response status types."""

    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class SourceDocument(BaseModel):
    """Source document in response."""

    id: str = Field(..., description="Document identifier")
    content: str = Field(..., description="Document content (truncated)")
    score: float | None = Field(None, description="Relevance score")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "doc_1",
                "content": "La inteligencia artificial es una rama de la informática...",
                "score": 0.95,
                "metadata": {
                    "source": "ai_basics",
                    "category": "definition",
                    "date": "2024-01-15",
                },
            }
        }
    )


class QueryResponse(BaseModel):
    """Response model for RAG queries."""

    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS, description="Response status"
    )
    sources: list[SourceDocument] = Field(
        default_factory=list, description="Source documents"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Response metadata"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    processing_time: float | None = Field(
        None, description="Processing time in seconds"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "¿Qué es la inteligencia artificial?",
                "answer": "La inteligencia artificial (IA) es una rama de la informática...",
                "status": "success",
                "sources": [
                    {
                        "id": "doc_1",
                        "content": "La inteligencia artificial es...",
                        "score": 0.95,
                        "metadata": {"source": "ai_basics"},
                    }
                ],
                "metadata": {
                    "source_count": 3,
                    "model_used": "gpt-3.5-turbo",
                    "tokens_used": 150,
                },
                "timestamp": "2024-01-15T10:30:00Z",
                "processing_time": 1.25,
            }
        }
    )


class BatchQueryResponse(BaseModel):
    """Response model for batch queries."""

    results: list[QueryResponse] = Field(..., description="List of query results")
    total_queries: int = Field(..., description="Total number of queries processed")
    successful_queries: int = Field(..., description="Number of successful queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    processing_time: float | None = Field(None, description="Total processing time")

    def __init__(self, **data):
        if "results" in data:
            results = data["results"]
            data["total_queries"] = len(results)
            data["successful_queries"] = sum(
                1 for r in results if r.status == ResponseStatus.SUCCESS
            )
            data["failed_queries"] = data["total_queries"] - data["successful_queries"]
        super().__init__(**data)


class StreamingQueryResponse(BaseModel):
    """Response model for streaming queries."""

    type: str = Field(
        ..., description="Message type (start, chunk, result, error, end)"
    )
    data: dict[str, Any] | None = Field(None, description="Message data")
    message: str | None = Field(None, description="Status message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "result",
                "data": {
                    "query": "¿Qué es la IA?",
                    "answer": "La inteligencia artificial...",
                    "sources": [],
                },
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] | None = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str | None = Field(None, description="Request identifier for tracking")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "message": "Query cannot be empty",
                "details": {"field": "query", "value": ""},
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_123456",
            }
        }
    )
