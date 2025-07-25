"""Error handling middleware."""

import traceback
from collections.abc import Callable

import structlog
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from src.adapters.input.api.schemas.responses import ErrorResponse

logger = structlog.get_logger()


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle errors and return structured responses."""
        try:
            return await call_next(request)

        except HTTPException as e:
            # FastAPI HTTP exceptions
            return await self._handle_http_exception(request, e)

        except ValidationError as e:
            # Pydantic validation errors
            return await self._handle_validation_error(request, e)

        except Exception as e:
            # Unexpected errors
            return await self._handle_unexpected_error(request, e)

    async def _handle_http_exception(
        self, request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Handle FastAPI HTTP exceptions."""
        request_id = getattr(request.state, "request_id", None)

        logger.warning(
            "http_exception",
            request_id=request_id,
            status_code=exc.status_code,
            detail=exc.detail,
            url=str(request.url),
        )

        error_response = ErrorResponse(
            error="HTTPException",
            message=str(exc.detail),
            details={"status_code": exc.status_code},
            request_id=request_id,
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(),
            headers={"X-Request-ID": request_id} if request_id else {},
        )

    async def _handle_validation_error(
        self, request: Request, exc: ValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors."""
        request_id = getattr(request.state, "request_id", None)

        logger.warning(
            "validation_error",
            request_id=request_id,
            errors=exc.errors(),
            url=str(request.url),
        )

        error_response = ErrorResponse(
            error="ValidationError",
            message="Request validation failed",
            details={"validation_errors": exc.errors()},
            request_id=request_id,
        )

        return JSONResponse(
            status_code=422,
            content=error_response.model_dump(),
            headers={"X-Request-ID": request_id} if request_id else {},
        )

    async def _handle_unexpected_error(
        self, request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle unexpected errors."""
        request_id = getattr(request.state, "request_id", None)

        logger.error(
            "unexpected_error",
            request_id=request_id,
            error=str(exc),
            error_type=type(exc).__name__,
            traceback=traceback.format_exc(),
            url=str(request.url),
        )

        error_response = ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            details={"error_type": type(exc).__name__} if request.app.debug else None,
            request_id=request_id,
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(),
            headers={"X-Request-ID": request_id} if request_id else {},
        )
