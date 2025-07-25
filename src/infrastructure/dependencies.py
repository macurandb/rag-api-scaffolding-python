"""Dependency injection configuration."""

from functools import lru_cache

from ..adapters.output.formatters import JSONResponseFormatter
from ..core.services.rag.context_service import DefaultContextService
from ..core.services.rag.pipeline import RAGPipelineBuilder
from ..core.services.rag.prompt_service import DefaultPromptService
from ..core.services.rag_orchestrator import RAGOrchestrator
from ..infrastructure.rag.langchain_implementations import (
    LangChainEmbeddingService,
    LangChainGenerationService,
    LangChainRetrievalService,
)


@lru_cache
def get_embedding_service():
    """Get embedding service instance."""
    return LangChainEmbeddingService()


@lru_cache
def get_retrieval_service():
    """Get retrieval service instance."""
    embedding_service = get_embedding_service()
    return LangChainRetrievalService(embedding_service)


@lru_cache
def get_generation_service():
    """Get generation service instance."""
    return LangChainGenerationService()


@lru_cache
def get_prompt_service():
    """Get prompt service instance."""
    return DefaultPromptService()


@lru_cache
def get_context_service():
    """Get context service instance."""
    return DefaultContextService()


@lru_cache
def get_formatter_service():
    """Get formatter service instance."""
    return JSONResponseFormatter()


@lru_cache
def get_rag_pipeline():
    """Get RAG pipeline instance."""
    return (
        RAGPipelineBuilder()
        .with_embedding_service(get_embedding_service())
        .with_retrieval_service(get_retrieval_service())
        .with_generation_service(get_generation_service())
        .with_prompt_service(get_prompt_service())
        .with_context_service(get_context_service())
        .build()
    )


@lru_cache
def get_rag_orchestrator():
    """Get RAG orchestrator instance."""
    return RAGOrchestrator(
        rag_pipeline=get_rag_pipeline(),
        formatter=get_formatter_service(),
    )
