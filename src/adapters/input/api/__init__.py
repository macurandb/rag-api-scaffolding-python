"""FastAPI application factory and main router."""

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from src.adapters.input.api.middleware import ErrorHandlerMiddleware, LoggingMiddleware
from src.adapters.input.api.routers import admin, health, rag
from src.config import settings

logger = structlog.get_logger()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="RAG Application API",
        description="Enterprise RAG (Retrieval-Augmented Generation) API with Hexagonal Architecture",
        version="1.0.0",
        docs_url="/docs" if settings.environment == "development" else None,
        redoc_url="/redoc" if settings.environment == "development" else None,
        openapi_url="/openapi.json" if settings.environment == "development" else None,
    )

    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=(
            ["*"]
            if settings.environment == "development"
            else ["localhost", "127.0.0.1"]
        ),
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.environment == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)

    # Include routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(rag.router, prefix="/api/v1", tags=["RAG"])
    app.include_router(admin.router, prefix="/admin", tags=["Admin"])

    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with API information."""
        return {
            "service": "RAG Application API",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs" if settings.environment == "development" else "disabled",
            "health": "/health",
        }

    return app


__all__ = ["create_app"]
