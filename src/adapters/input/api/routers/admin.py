"""Admin endpoints for system management."""

from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.config import settings

logger = structlog.get_logger()
router = APIRouter()
security = HTTPBearer()


class SystemInfo(BaseModel):
    """System information response."""

    service: str
    version: str
    environment: str
    timestamp: datetime
    configuration: dict[str, Any]


class MetricsResponse(BaseModel):
    """Metrics response model."""

    queries_processed: int
    average_response_time: float
    error_rate: float
    uptime: str


def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify admin authentication token."""
    # In production, implement proper JWT validation
    if settings.environment == "development":
        return True

    # Simple token validation for demo
    if credentials.credentials != "admin-token-123":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    return True


@router.get("/info", response_model=SystemInfo)
async def get_system_info(_: bool = Depends(verify_admin_token)):
    """Get system information (admin only)."""
    return SystemInfo(
        service="rag-application",
        version="1.0.0",
        environment=settings.environment,
        timestamp=datetime.utcnow(),
        configuration={
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "vector_store_type": settings.vector_store_type,
            "embedding_provider": settings.embedding_provider,
            "log_level": settings.log_level,
        },
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(_: bool = Depends(verify_admin_token)):
    """Get application metrics (admin only)."""
    # In production, integrate with proper metrics collection
    return MetricsResponse(
        queries_processed=0, average_response_time=0.0, error_rate=0.0, uptime="0h 0m"
    )


@router.post("/reload-config")
async def reload_configuration(_: bool = Depends(verify_admin_token)):
    """Reload application configuration (admin only)."""
    try:
        # In production, implement proper config reloading
        logger.info("Configuration reload requested")
        return {"status": "success", "message": "Configuration reloaded"}
    except Exception as e:
        logger.error("Error reloading configuration", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to reload configuration: {str(e)}"
        ) from e


@router.post("/clear-cache")
async def clear_cache(_: bool = Depends(verify_admin_token)):
    """Clear application cache (admin only)."""
    try:
        # In production, implement proper cache clearing
        logger.info("Cache clear requested")
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        logger.error("Error clearing cache", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to clear cache: {str(e)}"
        ) from e
