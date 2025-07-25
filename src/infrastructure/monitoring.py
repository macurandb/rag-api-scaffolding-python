"""Monitoring and metrics collection."""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class QueryMetrics:
    """Metrics for a single query."""

    query_id: str
    query_text: str
    processing_time: float
    source_count: int
    success: bool
    error_type: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MetricsCollector:
    """Collects and aggregates application metrics."""

    def __init__(self):
        self.query_metrics: list[QueryMetrics] = []
        self.start_time = datetime.utcnow()

    def record_query(self, metrics: QueryMetrics) -> None:
        """Record metrics for a query."""
        self.query_metrics.append(metrics)

        # Keep only last 1000 queries to prevent memory issues
        if len(self.query_metrics) > 1000:
            self.query_metrics = self.query_metrics[-1000:]

        logger.info(
            "query_metrics_recorded",
            query_id=metrics.query_id,
            processing_time=metrics.processing_time,
            success=metrics.success,
            source_count=metrics.source_count,
        )

    def get_summary_metrics(self, hours: int = 24) -> dict[str, Any]:
        """Get summary metrics for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [m for m in self.query_metrics if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "average_processing_time": 0.0,
                "error_rate": 0.0,
                "uptime": str(datetime.utcnow() - self.start_time),
            }

        successful = [m for m in recent_metrics if m.success]
        failed = [m for m in recent_metrics if not m.success]

        avg_processing_time = (
            sum(m.processing_time for m in successful) / len(successful)
            if successful
            else 0.0
        )

        return {
            "total_queries": len(recent_metrics),
            "successful_queries": len(successful),
            "failed_queries": len(failed),
            "average_processing_time": round(avg_processing_time, 3),
            "error_rate": round(len(failed) / len(recent_metrics), 3),
            "uptime": str(datetime.utcnow() - self.start_time),
            "queries_per_hour": round(len(recent_metrics) / hours, 1),
        }

    def get_error_breakdown(self, hours: int = 24) -> dict[str, int]:
        """Get breakdown of errors by type."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        failed_metrics = [
            m
            for m in self.query_metrics
            if m.timestamp >= cutoff_time and not m.success and m.error_type
        ]

        error_counts = {}
        for metric in failed_metrics:
            error_type = metric.error_type or "Unknown"
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        return error_counts


# Global metrics collector instance
metrics_collector = MetricsCollector()


class PerformanceMonitor:
    """Context manager for monitoring performance."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.debug("operation_started", operation=self.operation_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        if exc_type is None:
            logger.info(
                "operation_completed",
                operation=self.operation_name,
                duration=round(duration, 3),
            )
        else:
            logger.error(
                "operation_failed",
                operation=self.operation_name,
                duration=round(duration, 3),
                error=str(exc_val),
                error_type=exc_type.__name__ if exc_type else None,
            )

    @property
    def duration(self) -> float | None:
        """Get operation duration if completed."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
