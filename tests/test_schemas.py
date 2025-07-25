"""Tests for API schemas."""

import pytest
from pydantic import ValidationError

from src.adapters.input.api.schemas.requests import (
    BatchQueryRequest,
    QueryRequest,
    QueryType,
)
from src.adapters.input.api.schemas.responses import (
    QueryResponse,
    ResponseStatus,
    SourceDocument,
)


class TestQueryRequest:
    """Tests for QueryRequest schema."""

    def test_valid_query_request(self):
        """Test valid query request creation."""
        request = QueryRequest(
            query="What is artificial intelligence?",
            query_type=QueryType.QUESTION,
            max_sources=5,
        )
        assert request.query == "What is artificial intelligence?"
        assert request.query_type == QueryType.QUESTION
        assert request.max_sources == 5

    def test_query_request_with_metadata(self):
        """Test query request with metadata."""
        metadata = {"language": "en", "domain": "technology"}
        request = QueryRequest(query="Test query", metadata=metadata, user_id="user123")
        assert request.metadata == metadata
        assert request.user_id == "user123"

    def test_empty_query_validation(self):
        """Test that empty query raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="")

        errors = exc_info.value.errors()
        # Check for the actual Pydantic V2 error message
        assert any(
            "String should have at least 1 character" in str(error) for error in errors
        )

    def test_whitespace_query_validation(self):
        """Test that whitespace-only query raises validation error."""
        with pytest.raises(ValidationError):
            QueryRequest(query="   ")

    def test_query_too_long(self):
        """Test that overly long query raises validation error."""
        long_query = "x" * 2001  # Exceeds max_length of 2000
        with pytest.raises(ValidationError):
            QueryRequest(query=long_query)

    def test_max_sources_validation(self):
        """Test max_sources validation."""
        # Test minimum
        with pytest.raises(ValidationError):
            QueryRequest(query="test", max_sources=0)

        # Test maximum
        with pytest.raises(ValidationError):
            QueryRequest(query="test", max_sources=21)

    def test_query_strip(self):
        """Test that query is stripped of whitespace."""
        request = QueryRequest(query="  test query  ")
        assert request.query == "test query"


class TestBatchQueryRequest:
    """Tests for BatchQueryRequest schema."""

    def test_valid_batch_request(self):
        """Test valid batch request creation."""
        queries = [QueryRequest(query="Query 1"), QueryRequest(query="Query 2")]
        batch_request = BatchQueryRequest(queries=queries)
        assert len(batch_request.queries) == 2
        assert batch_request.parallel is True

    def test_empty_batch_validation(self):
        """Test that empty batch raises validation error."""
        with pytest.raises(ValidationError):
            BatchQueryRequest(queries=[])

    def test_batch_too_large(self):
        """Test that batch with too many queries raises validation error."""
        queries = [
            QueryRequest(query=f"Query {i}") for i in range(11)
        ]  # Exceeds max of 10
        with pytest.raises(ValidationError):
            BatchQueryRequest(queries=queries)


class TestQueryResponse:
    """Tests for QueryResponse schema."""

    def test_valid_response(self):
        """Test valid response creation."""
        sources = [
            SourceDocument(
                id="doc1",
                content="Test content",
                score=0.95,
                metadata={"source": "test"},
            )
        ]

        response = QueryResponse(
            query="Test query",
            answer="Test answer",
            sources=sources,
            metadata={"test": True},
        )

        assert response.query == "Test query"
        assert response.answer == "Test answer"
        assert len(response.sources) == 1
        assert response.status == ResponseStatus.SUCCESS

    def test_response_with_defaults(self):
        """Test response with default values."""
        response = QueryResponse(query="Test query", answer="Test answer")

        assert response.sources == []
        assert response.metadata == {}
        assert response.status == ResponseStatus.SUCCESS
        assert response.timestamp is not None


class TestSourceDocument:
    """Tests for SourceDocument schema."""

    def test_valid_source_document(self):
        """Test valid source document creation."""
        doc = SourceDocument(
            id="doc1",
            content="Test content",
            score=0.95,
            metadata={"source": "test", "category": "example"},
        )

        assert doc.id == "doc1"
        assert doc.content == "Test content"
        assert doc.score == 0.95
        assert doc.metadata["source"] == "test"

    def test_source_document_without_score(self):
        """Test source document without score."""
        doc = SourceDocument(id="doc1", content="Test content")

        assert doc.score is None
        assert doc.metadata == {}
