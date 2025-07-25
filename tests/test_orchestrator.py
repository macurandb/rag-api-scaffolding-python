"""Tests for RAG Orchestrator."""

import pytest

from src.core.domain.rag import (
    GenerationStrategy,
    QueryType,
)
from src.core.services.rag_orchestrator import RAGOrchestrator


@pytest.mark.asyncio
async def test_process_query_success(mock_rag_orchestrator):
    """Test successful query processing."""
    # Execute
    result = await mock_rag_orchestrator.process_query("test query")

    # Verify
    assert result["query"] == "test query"
    assert result["answer"] == "Mock generated response"
    assert "sources" in result
    assert "metadata" in result


@pytest.mark.asyncio
async def test_process_query_with_metadata(mock_rag_orchestrator):
    """Test query processing with metadata."""
    # Execute with metadata
    metadata = {"user_id": "123", "session": "abc"}
    result = await mock_rag_orchestrator.process_query("test query", metadata)

    # Verify
    assert result["query"] == "test query"
    assert result["answer"] == "Mock generated response"


@pytest.mark.asyncio
async def test_process_query_with_parameters(mock_rag_orchestrator):
    """Test query processing with different parameters."""
    # Execute with parameters
    result = await mock_rag_orchestrator.process_query(
        "test query",
        query_type="summarization",
        generation_strategy="chain_of_thought",
        max_sources=3,
        temperature=0.5,
    )

    # Verify
    assert result["query"] == "test query"
    assert result["answer"] == "Mock generated response"


@pytest.mark.asyncio
async def test_health_check(mock_rag_orchestrator):
    """Test health check functionality."""
    # Execute
    health = await mock_rag_orchestrator.health_check()

    # Verify
    assert "pipeline" in health
    assert health["pipeline"] == "healthy"


def test_parse_query_type():
    """Test query type parsing with real orchestrator."""
    from src.core.services.rag.pipeline import RAGPipelineBuilder
    from tests.mocks import (
        MockContextService,
        MockEmbeddingService,
        MockGenerationService,
        MockPromptService,
        MockRetrievalService,
    )

    # Create real orchestrator with mocks
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

    # Test valid query type
    query_type = orchestrator._parse_query_type("question_answering")
    assert query_type == QueryType.QUESTION_ANSWERING

    # Test invalid query type (should default)
    query_type = orchestrator._parse_query_type("invalid_type")
    assert query_type == QueryType.QUESTION_ANSWERING


def test_parse_generation_strategy():
    """Test generation strategy parsing with real orchestrator."""
    from src.core.services.rag.pipeline import RAGPipelineBuilder
    from tests.mocks import (
        MockContextService,
        MockEmbeddingService,
        MockGenerationService,
        MockPromptService,
        MockRetrievalService,
    )

    # Create real orchestrator with mocks
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

    # Test valid strategy
    strategy = orchestrator._parse_generation_strategy("chain_of_thought")
    assert strategy == GenerationStrategy.CHAIN_OF_THOUGHT

    # Test invalid strategy (should default)
    strategy = orchestrator._parse_generation_strategy("invalid_strategy")
    assert strategy == GenerationStrategy.STANDARD
