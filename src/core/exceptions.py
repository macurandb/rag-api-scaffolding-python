"""Custom exceptions for the RAG application."""


class RAGException(Exception):
    """Base exception for RAG application."""

    pass


class EmbeddingException(RAGException):
    """Exception raised during embedding generation."""

    pass


class RetrievalException(RAGException):
    """Exception raised during document retrieval."""

    pass


class GenerationException(RAGException):
    """Exception raised during text generation."""

    pass


class VectorStoreException(RAGException):
    """Exception raised during vector store operations."""

    pass


class ConfigurationException(RAGException):
    """Exception raised for configuration errors."""

    pass


class ValidationException(RAGException):
    """Exception raised for validation errors."""

    pass
