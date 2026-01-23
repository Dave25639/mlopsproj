"""
System metrics instrumentation for the API.

This module provides metrics collection for monitoring API performance,
including request counts, latency, error rates, and prediction statistics.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    """Container for API metrics."""

    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Latency metrics (in milliseconds)
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0

    # Prediction metrics
    total_predictions: int = 0
    confidence_sum: float = 0.0
    min_confidence: float = 1.0
    max_confidence: float = 0.0

    # Error tracking
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Endpoint-specific metrics
    endpoint_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Timestamps
    first_request_time: Optional[datetime] = None
    last_request_time: Optional[datetime] = None

    def record_request(
        self,
        endpoint: str,
        success: bool,
        latency_ms: float,
        error: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """Record a request metric."""
        now = datetime.now()

        if self.first_request_time is None:
            self.first_request_time = now
        self.last_request_time = now

        self.total_requests += 1
        self.endpoint_counts[endpoint] += 1

        if success:
            self.successful_requests += 1
            if confidence is not None:
                self.total_predictions += 1
                self.confidence_sum += confidence
                self.min_confidence = min(self.min_confidence, confidence)
                self.max_confidence = max(self.max_confidence, confidence)
        else:
            self.failed_requests += 1
            if error:
                self.error_counts[error] += 1

        # Update latency metrics
        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)

    def get_average_latency(self) -> float:
        """Get average request latency in milliseconds."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    def get_average_confidence(self) -> float:
        """Get average prediction confidence."""
        if self.total_predictions == 0:
            return 0.0
        return self.confidence_sum / self.total_predictions

    def get_success_rate(self) -> float:
        """Get success rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    def get_error_rate(self) -> float:
        """Get error rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(self.get_success_rate(), 2),
            "error_rate": round(self.get_error_rate(), 2),
            "latency": {
                "average_ms": round(self.get_average_latency(), 2),
                "min_ms": round(self.min_latency_ms, 2) if self.min_latency_ms != float('inf') else 0.0,
                "max_ms": round(self.max_latency_ms, 2),
                "total_ms": round(self.total_latency_ms, 2),
            },
            "predictions": {
                "total": self.total_predictions,
                "average_confidence": round(self.get_average_confidence(), 4),
                "min_confidence": round(self.min_confidence, 4) if self.min_confidence != 1.0 else 0.0,
                "max_confidence": round(self.max_confidence, 4),
            },
            "error_counts": dict(self.error_counts),
            "endpoint_counts": dict(self.endpoint_counts),
            "first_request_time": self.first_request_time.isoformat() if self.first_request_time else None,
            "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0
        self.min_latency_ms = float('inf')
        self.max_latency_ms = 0.0
        self.total_predictions = 0
        self.confidence_sum = 0.0
        self.min_confidence = 1.0
        self.max_confidence = 0.0
        self.error_counts.clear()
        self.endpoint_counts.clear()
        self.first_request_time = None
        self.last_request_time = None


# Global metrics instance
_metrics: Optional[Metrics] = None


def get_metrics() -> Metrics:
    """Get or create the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics


class MetricsMiddleware:
    """Middleware to track request metrics."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        """ASGI middleware implementation."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        endpoint = scope.get("path", "unknown")

        # Track request
        metrics = get_metrics()

        try:
            await self.app(scope, receive, send)
            # Success - latency will be recorded in endpoint handlers
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            metrics.record_request(
                endpoint=endpoint,
                success=False,
                latency_ms=latency_ms,
                error=str(e)
            )
            raise
