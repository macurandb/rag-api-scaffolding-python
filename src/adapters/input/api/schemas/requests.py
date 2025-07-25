"""Request schemas for API endpoints."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryType(str, Enum):
    """Types of queries supported."""

    QUESTION = "question"
    SUMMARY = "summary"
    ANALYSIS = "analysis"


class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    query: str = Field(
        ..., min_length=1, max_length=2000, description="User query text"
    )
    query_type: QueryType = Field(
        default=QueryType.QUESTION, description="Type of query being made"
    )
    max_sources: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of source documents to retrieve",
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional metadata for the query"
    )
    user_id: str | None = Field(
        default=None, description="Optional user identifier for tracking"
    )
    session_id: str | None = Field(
        default=None, description="Optional session identifier"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate query content."""
        if not v.strip():
            raise ValueError("Query cannot be empty or only whitespace")
        return v.strip()

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v):
        """Validate metadata structure."""
        if v is not None and not isinstance(v, dict):
            raise ValueError("Metadata must be a dictionary")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "¿Qué es la inteligencia artificial?",
                "query_type": "question",
                "max_sources": 5,
                "metadata": {"language": "es", "domain": "technology"},
                "user_id": "user123",
                "session_id": "session456",
            }
        }
    )


class BatchQueryRequest(BaseModel):
    """Request model for batch RAG queries."""

    queries: list[QueryRequest] = Field(
        ..., min_length=1, max_length=10, description="List of queries to process"
    )
    parallel: bool = Field(
        default=True, description="Whether to process queries in parallel"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "queries": [
                    {
                        "query": "¿Qué es la inteligencia artificial?",
                        "query_type": "question",
                    },
                    {"query": "Explica el machine learning", "query_type": "summary"},
                ],
                "parallel": True,
            }
        }
    )
