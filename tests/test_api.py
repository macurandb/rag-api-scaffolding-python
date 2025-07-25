"""Tests for API endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.adapters.input.api import create_app


@pytest.fixture
def client():
    """Test client fixture."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_orchestrator():
    """Mock RAG orchestrator."""
    mock = AsyncMock()
    mock.process_query.return_value = {
        "query": "test query",
        "answer": "test answer",
        "sources": [],
        "metadata": {"test": True},
    }
    return mock


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_detailed_health_check(client):
    """Test detailed health check endpoint."""
    response = client.get("/health/detailed")
    assert response.status_code == 200
    data = response.json()
    assert "checks" in data
    assert "timestamp" in data


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "RAG Application API" in response.json()["service"]


def test_query_endpoint_success_unit():
    """Test successful query processing at unit level."""
    # This test verifies the orchestrator works correctly with mocks
    import asyncio

    from src.core.services.rag.pipeline import RAGPipelineBuilder
    from src.core.services.rag_orchestrator import RAGOrchestrator
    from tests.mocks import (
        MockContextService,
        MockEmbeddingService,
        MockGenerationService,
        MockPromptService,
        MockRetrievalService,
    )

    # Create real orchestrator with mock services
    pipeline = (
        RAGPipelineBuilder()
        .with_embedding_service(MockEmbeddingService())
        .with_retrieval_service(MockRetrievalService())
        .with_generation_service(MockGenerationService())
        .with_prompt_service(MockPromptService())
        .with_context_service(MockContextService())
        .build()
    )

    orchestrator = RAGOrchestrator(rag_pipeline=pipeline)

    # Test the orchestrator directly
    async def test_orchestrator():
        result = await orchestrator.process_query("test query")
        assert result["query"]["original_text"] == "test query"
        assert result["answer"] == "Mock generated response"
        assert "sources" in result
        assert "metadata" in result
        return result

    # Run the async test
    result = asyncio.run(test_orchestrator())
    assert result is not None


def test_batch_query_processing_unit():
    """Test batch query processing at unit level."""
    # This test verifies batch processing works with the new architecture
    import asyncio

    from src.core.services.rag.pipeline import RAGPipelineBuilder
    from src.core.services.rag_orchestrator import RAGOrchestrator
    from tests.mocks import (
        MockContextService,
        MockEmbeddingService,
        MockGenerationService,
        MockPromptService,
        MockRetrievalService,
    )

    # Create real orchestrator with mock services
    pipeline = (
        RAGPipelineBuilder()
        .with_embedding_service(MockEmbeddingService())
        .with_retrieval_service(MockRetrievalService())
        .with_generation_service(MockGenerationService())
        .with_prompt_service(MockPromptService())
        .with_context_service(MockContextService())
        .build()
    )

    orchestrator = RAGOrchestrator(rag_pipeline=pipeline)

    # Test multiple queries
    async def test_multiple_queries():
        queries = ["test query 1", "test query 2"]
        results = []

        for query in queries:
            result = await orchestrator.process_query(query)
            results.append(result)

        assert len(results) == 2
        for result in results:
            assert "query" in result
            assert "answer" in result
            assert result["answer"] == "Mock generated response"

        return results

    # Run the async test
    results = asyncio.run(test_multiple_queries())
    assert len(results) == 2


def test_query_endpoint_validation_error(client):
    """Test query validation error."""
    response = client.post(
        "/api/v1/query", json={"query": ""}  # Empty query should fail validation
    )

    assert response.status_code == 422


@patch("src.infrastructure.dependencies.get_rag_orchestrator")
def test_query_endpoint_processing_error(mock_get_orchestrator, client):
    """Test query processing error."""
    mock_orchestrator = AsyncMock()
    mock_orchestrator.process_query.side_effect = Exception("Processing error")
    mock_get_orchestrator.return_value = mock_orchestrator

    response = client.post("/api/v1/query", json={"query": "test query"})

    assert response.status_code == 500
    response_data = response.json()
    # Check for error in the response structure (could be in 'detail' or direct)
    assert "error" in response_data or "detail" in response_data


def test_admin_endpoints_require_auth(client):
    """Test that admin endpoints require authentication."""
    response = client.get("/admin/info")
    assert response.status_code == 403  # Should require authentication
