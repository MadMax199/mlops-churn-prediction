"""In-memory monitoring for API predictions."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class PredictionMonitor:
    """Collect operational metrics without customer features."""

    def __init__(
        self,
        model_name: str,
        model_version: str,
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version

        self.started_at = datetime.now(UTC).isoformat()
        self.last_request_at: str | None = None

        self._total_requests = 0
        self._successful_predictions = 0
        self._failed_predictions = 0
        self._predicted_churn_count = 0

        self._probability_sum = 0.0
        self._latency_sum_ms = 0.0

        self._lock = Lock()

    def record_success(
        self,
        *,
        request_id: str,
        prediction: int,
        churn_probability: float,
        latency_ms: float,
    ) -> None:
        """Record a successful prediction."""

        timestamp = datetime.now(UTC).isoformat()

        with self._lock:
            self._total_requests += 1
            self._successful_predictions += 1
            self._predicted_churn_count += int(prediction == 1)
            self._probability_sum += churn_probability
            self._latency_sum_ms += latency_ms
            self.last_request_at = timestamp

        event = {
            "event": "prediction",
            "status": "success",
            "timestamp": timestamp,
            "request_id": request_id,
            "prediction": prediction,
            "churn_probability": round(
                churn_probability,
                6,
            ),
            "latency_ms": round(latency_ms, 2),
            "model_name": self.model_name,
            "model_version": self.model_version,
        }

        logger.info(
            "prediction_monitoring %s",
            json.dumps(
                event,
                sort_keys=True,
            ),
        )

    def record_failure(
        self,
        *,
        request_id: str,
        latency_ms: float,
        error: Exception,
    ) -> None:
        """Record a failed prediction."""

        timestamp = datetime.now(UTC).isoformat()

        with self._lock:
            self._total_requests += 1
            self._failed_predictions += 1
            self._latency_sum_ms += latency_ms
            self.last_request_at = timestamp

        event = {
            "event": "prediction",
            "status": "failed",
            "timestamp": timestamp,
            "request_id": request_id,
            "latency_ms": round(latency_ms, 2),
            "error_type": type(error).__name__,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }

        logger.warning(
            "prediction_monitoring %s",
            json.dumps(
                event,
                sort_keys=True,
            ),
        )

    def snapshot(self) -> dict[str, Any]:
        """Return the current monitoring metrics."""

        with self._lock:
            total_requests = self._total_requests
            successful_predictions = self._successful_predictions

            if total_requests:
                average_latency_ms = self._latency_sum_ms / total_requests
            else:
                average_latency_ms = 0.0

            if successful_predictions:
                average_probability = self._probability_sum / successful_predictions

                predicted_churn_rate = self._predicted_churn_count / successful_predictions
            else:
                average_probability = 0.0
                predicted_churn_rate = 0.0

            return {
                "total_requests": total_requests,
                "successful_predictions": (successful_predictions),
                "failed_predictions": (self._failed_predictions),
                "predicted_churn_count": (self._predicted_churn_count),
                "predicted_churn_rate": round(
                    predicted_churn_rate,
                    6,
                ),
                "average_churn_probability": round(
                    average_probability,
                    6,
                ),
                "average_latency_ms": round(
                    average_latency_ms,
                    2,
                ),
                "model_name": self.model_name,
                "model_version": self.model_version,
                "started_at": self.started_at,
                "last_request_at": self.last_request_at,
            }
