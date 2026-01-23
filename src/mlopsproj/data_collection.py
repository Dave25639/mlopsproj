"""
Data collection module for logging API predictions and inputs.

This module handles collection and storage of input-output data from the deployed API
for monitoring, analysis, and potential retraining.
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PredictionLogger:
    """
    Logger for API predictions and inputs.

    Stores prediction data to files for later analysis.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        max_file_size_mb: int = 100,
        rotate_daily: bool = True,
    ):
        """
        Initialize the prediction logger.

        Args:
            log_dir: Directory to store log files (default: outputs/predictions)
            max_file_size_mb: Maximum log file size in MB before rotation
            rotate_daily: If True, create new log file each day
        """
        if log_dir is None:
            # Default to outputs/predictions in project root
            project_root = Path(__file__).parent.parent.parent
            log_dir = str(project_root / "outputs" / "predictions")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.rotate_daily = rotate_daily

        logger.info(f"Prediction logger initialized. Log directory: {self.log_dir}")

    def _get_log_file_path(self) -> Path:
        """Get the current log file path."""
        if self.rotate_daily:
            date_str = datetime.now().strftime("%Y-%m-%d")
            return self.log_dir / f"predictions_{date_str}.jsonl"
        else:
            return self.log_dir / "predictions.jsonl"

    def _check_file_size(self, file_path: Path) -> bool:
        """Check if log file exceeds maximum size."""
        if not file_path.exists():
            return False
        return file_path.stat().st_size >= self.max_file_size

    def _get_image_hash(self, image_bytes: bytes) -> str:
        """Calculate SHA256 hash of image for deduplication."""
        return hashlib.sha256(image_bytes).hexdigest()

    def log_prediction(
        self,
        image_bytes: Optional[bytes] = None,
        image_path: Optional[str] = None,
        predictions: List[Dict] = None,
        top_prediction: str = "",
        top_confidence: float = 0.0,
        request_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Log a prediction event.

        Args:
            image_bytes: Image data as bytes (for uploaded images)
            image_path: Path to image file (for local file predictions)
            predictions: List of prediction dictionaries with class_name, confidence, rank
            top_prediction: Top predicted class name
            top_confidence: Confidence score of top prediction
            request_id: Optional request identifier
            latency_ms: Request latency in milliseconds
            error: Error message if prediction failed
        """
        try:
            # Create log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
            }

            # Add image identifier
            if image_bytes:
                image_hash = self._get_image_hash(image_bytes)
                log_entry["image_hash"] = image_hash
                log_entry["image_source"] = "upload"
            elif image_path:
                log_entry["image_path"] = image_path
                log_entry["image_source"] = "local_path"

            # Add prediction results
            if error:
                log_entry["error"] = error
                log_entry["success"] = False
            else:
                log_entry["success"] = True
                log_entry["top_prediction"] = top_prediction
                log_entry["top_confidence"] = top_confidence
                log_entry["predictions"] = predictions or []

            # Add performance metrics
            if latency_ms is not None:
                log_entry["latency_ms"] = latency_ms

            # Write to log file (JSONL format - one JSON object per line)
            log_file = self._get_log_file_path()

            # Check if we need to rotate due to size
            if self._check_file_size(log_file):
                # Rotate by appending timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rotated_file = log_file.parent / f"{log_file.stem}_{timestamp}{log_file.suffix}"
                log_file.rename(rotated_file)
                logger.info(f"Rotated log file to {rotated_file}")

            # Append log entry
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            logger.error(f"Failed to log prediction: {e}", exc_info=True)

    def get_recent_predictions(self, limit: int = 100) -> List[Dict]:
        """
        Get recent predictions from log files.

        Args:
            limit: Maximum number of predictions to return

        Returns:
            List of prediction dictionaries
        """
        predictions = []

        # Get all log files, sorted by modification time (newest first)
        log_files = sorted(
            self.log_dir.glob("predictions*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for log_file in log_files:
            if len(predictions) >= limit:
                break

            try:
                with open(log_file, "r") as f:
                    for line in f:
                        if len(predictions) >= limit:
                            break
                        try:
                            entry = json.loads(line.strip())
                            predictions.append(entry)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"Failed to read log file {log_file}: {e}")

        return predictions


# Global logger instance
_prediction_logger: Optional[PredictionLogger] = None


def get_prediction_logger() -> PredictionLogger:
    """Get or create the global prediction logger instance."""
    global _prediction_logger
    if _prediction_logger is None:
        log_dir = os.getenv("PREDICTION_LOG_DIR")
        _prediction_logger = PredictionLogger(log_dir=log_dir)
    return _prediction_logger
