"""Test configuration and fixtures."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from src.core.services.rag.pipeline import RAGPipelineBuilder
from tests.mocks import (
    MockContextService,
    MockEmbeddingService,
    MockGenerationService,
    MockPromptService,
    MockRetrievalService,
)


@pytest.fixture(autouse=True)
def mock_openai_api_key():
    """Mock OpenAI API key for all tests."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "mock-api-key"}):
        yield


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service fixture."""
    return MockEmbeddingService()


@pytest.fixture
def mock_retrieval_service():
    """Mock retrieval service fixture."""
    return MockRetrievalService()


@pytest.fixture
def mock_generation_service():
    """Mock generation service fixture."""
    return MockGenerationService()


@pytest.fixture
def mock_prompt_service():
    """Mock prompt service fixture."""
    return MockPromptService()


@pytest.fixture
def mock_context_service():
    """Mock context service fixture."""
    return MockContextService()


@pytest.fixture
def mock_rag_pipeline(
    mock_embedding_service,
    mock_retrieval_service,
    mock_generation_service,
    mock_prompt_service,
    mock_context_service,
):
    """Mock RAG pipeline fixture."""
    return (
        RAGPipelineBuilder()
        .with_embedding_service(mock_embedding_service)
        .with_retrieval_service(mock_retrieval_service)
        .with_generation_service(mock_generation_service)
        .with_prompt_service(mock_prompt_service)
        .with_context_service(mock_context_service)
        .build()
    )


@pytest.fixture
def mock_rag_orchestrator():
    """Mock RAG orchestrator fixture."""
    mock = AsyncMock()
    mock.process_query.return_value = {
        "query": "test query",
        "answer": "Mock generated response",
        "sources": [
            {
                "id": "doc1",
                "content": "Mock document content",
                "metadata": {"source": "mock"},
                "relevance_score": 0.9,
                "rank": 1,
            }
        ],
        "metadata": {
            "processing_time": 0.5,
            "source_count": 1,
            "query_type": "question_answering",
            "confidence_score": 0.8,
            "tokens_used": 50,
            "model_used": "mock-model",
        },
    }
    mock.health_check.return_value = {"pipeline": "healthy", "timestamp": 1234567890}
    return mock


@pytest.fixture
def patch_dependencies(mock_rag_orchestrator):
    """Patch dependency injection for tests."""
    with patch(
        "src.infrastructure.dependencies.get_rag_orchestrator",
        return_value=mock_rag_orchestrator,
    ):
        with patch("src.infrastructure.dependencies.get_rag_pipeline"):
            with patch("src.infrastructure.dependencies.get_embedding_service"):
                with patch("src.infrastructure.dependencies.get_retrieval_service"):
                    with patch(
                        "src.infrastructure.dependencies.get_generation_service"
                    ):
                        yield
