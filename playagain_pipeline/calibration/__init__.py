"""Calibration module for electrode orientation detection."""

from playagain_pipeline.calibration.calibrator_old import (
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
