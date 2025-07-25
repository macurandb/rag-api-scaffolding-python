"""Structured logging configuration."""

import logging

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structured logging."""

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s", stream=None, level=getattr(logging, log_level.upper())
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


class RAGLogger:
    """Structured logger for RAG operations."""

    def __init__(self):
        self.logger = structlog.get_logger()

    async def log_query(self, query: str, metadata: dict = None) -> None:
        """Log a query."""
        self.logger.info("query_received", query=query, metadata=metadata)

    async def log_retrieval(
        self, query: str, doc_count: int, scores: list = None
    ) -> None:
        """Log document retrieval."""
        self.logger.info(
            "documents_retrieved", query=query, document_count=doc_count, scores=scores
        )

    async def log_generation(self, prompt_length: int, response_length: int) -> None:
        """Log text generation."""
        self.logger.info(
            "text_generated",
            prompt_length=prompt_length,
            response_length=response_length,
        )

    async def log_error(self, error: Exception, context: dict = None) -> None:
        """Log an error."""
        self.logger.error(
            "rag_error",
            error=str(error),
            error_type=type(error).__name__,
            context=context,
        )
