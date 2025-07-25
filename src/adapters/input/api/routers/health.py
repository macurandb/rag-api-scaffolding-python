"""Health check endpoints."""

from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.core.services.rag_orchestrator import RAGOrchestrator
from src.infrastructure.dependencies import get_rag_orchestrator

logger = structlog.get_logger()
router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: datetime
    service: str
    version: str
    checks: dict[str, Any]


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        service="rag-application",
        version="1.0.0",
        checks={"basic": "ok"},
    )


@router.get("/detailed", response_model=HealthResponse)
async def detailed_health_check(
    orchestrator: RAGOrchestrator = Depends(get_rag_orchestrator),
):
    """Detailed health check with dependency verification."""
    checks = {}
    overall_status = "healthy"

    try:
        # Check orchestrator
        checks["orchestrator"] = "ok" if orchestrator else "error"

        # Check services (basic validation)
        checks["retriever"] = "ok" if orchestrator.retriever else "error"
        checks["generator"] = "ok" if orchestrator.generator else "error"
        checks["formatter"] = "ok" if orchestrator.formatter else "error"

        # If any check failed, mark as unhealthy
        if "error" in checks.values():
            overall_status = "unhealthy"

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        checks["error"] = str(e)
        overall_status = "unhealthy"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        service="rag-application",
        version="1.0.0",
        checks=checks,
    )


@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive"}
