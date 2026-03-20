"""Calibration module for electrode orientation detection."""

from playagain_pipeline.calibration.calibrator import (
    CalibrationResult,
    CalibrationProcessor,
    AutoCalibrator,
    backfill_session_rotations
)

__all__ = [
    "CalibrationResult",
    "CalibrationProcessor",
    "AutoCalibrator",
    "backfill_session_rotations"
]
