"""Core components of the gesture pipeline."""

from playagain_pipeline.core.gesture import (
    Gesture,
    GestureSet,
    GestureCategory,
    create_default_gesture_set,
    create_calibration_gesture_set
)
from playagain_pipeline.core.session import (
    RecordingSession,
    RecordingMetadata,
    RecordingTrial
)
from playagain_pipeline.core.data_manager_old import DataManager

__all__ = [
    "Gesture",
    "GestureSet",
    "GestureCategory",
    "create_default_gesture_set",
    "create_extended_gesture_set",
    "create_calibration_gesture_set",
    "RecordingSession",
    "RecordingMetadata",
    "RecordingTrial",
    "DataManager",
]
