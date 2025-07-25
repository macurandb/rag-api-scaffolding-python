"""RAG endpoints router."""

import json
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.adapters.input.api.schemas.requests import BatchQueryRequest, QueryRequest
from src.adapters.input.api.schemas.responses import (
    BatchQueryResponse,
    QueryResponse,
)
from src.core.services.rag_orchestrator import RAGOrchestrator
from src.infrastructure.dependencies import get_rag_orchestrator

logger = structlog.get_logger()
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    orchestrator: RAGOrchestrator = Depends(get_rag_orchestrator),
):
    """Process a single RAG query."""
    try:
        logger.info("Processing RAG query", query=request.query[:100])

        result = await orchestrator.process_query(
            query_text=request.query, metadata=request.metadata
        )

        # Add background task for analytics (optional)
        background_tasks.add_task(
            _log_query_analytics, request.query, result.get("metadata", {})
        )

        return QueryResponse(**result)

    except Exception as e:
        logger.error("Error processing query", error=str(e), query=request.query[:100])
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Query processing failed",
                "message": str(e),
                "type": type(e).__name__,
            },
        ) from e


@router.post("/query/batch", response_model=BatchQueryResponse)
async def process_batch_queries(
    request: BatchQueryRequest,
    orchestrator: RAGOrchestrator = Depends(get_rag_orchestrator),
):
    """Process multiple RAG queries in batch."""
    try:
        logger.info("Processing batch queries", count=len(request.queries))

        results = []
        for query_req in request.queries:
            try:
                result = await orchestrator.process_query(
                    query_text=query_req.query, metadata=query_req.metadata
                )
                results.append(QueryResponse(**result))
            except Exception as e:
                logger.error(
                    "Error in batch query", error=str(e), query=query_req.query[:100]
                )
                results.append(
                    QueryResponse(
                        query=query_req.query,
                        answer=f"Error: {str(e)}",
                        sources=[],
                        metadata={"error": True, "error_type": type(e).__name__},
                    )
                )

        return BatchQueryResponse(results=results)

    except Exception as e:
        logger.error("Error processing batch queries", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Batch processing failed: {str(e)}"
        ) from e


@router.post("/query/stream")
async def stream_query(
    request: QueryRequest, orchestrator: RAGOrchestrator = Depends(get_rag_orchestrator)
):
    """Stream RAG query response (for real-time UI)."""

    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # Start processing
            yield f"data: {json.dumps({'type': 'start', 'message': 'Processing query...'})}\n\n"

            # Simulate streaming (in real implementation, this would stream from LLM)
            result = await orchestrator.process_query(
                query_text=request.query, metadata=request.metadata
            )

            # Stream the result
            yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"
            yield f"data: {json.dumps({'type': 'end'})}\n\n"

        except Exception as e:
            logger.error("Error in streaming query", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def _log_query_analytics(query: str, metadata: dict[str, Any]) -> None:
    """Background task to log query analytics."""
    try:
        logger.info(
            "query_analytics",
            query_length=len(query),
            has_metadata=bool(metadata),
            source_count=metadata.get("source_count", 0),
        )
    except Exception as e:
        logger.error("Error logging analytics", error=str(e))
