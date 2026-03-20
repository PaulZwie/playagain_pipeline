"""
Gesture Pipeline - A modular EMG gesture recording and classification pipeline.

This package provides tools for:
- Recording EMG data with configurable gestures
- Calibration protocols for electrode orientation detection
- Training various ML models for gesture recognition
- Real-time gesture prediction from EMG streams

Quick Start:
    from playagain_pipeline import (
        create_default_gesture_set,
        RecordingSession,
        DataManager,
        RecordingProtocol,
        create_standard_protocol,
        ModelManager
    )
"""

__version__ = "0.1.0"
__author__ = "Paul"

# Core components
from playagain_pipeline.core.gesture import (
    Gesture,
    GestureSet,
    GestureCategory,
    create_default_gesture_set,
    create_calibration_gesture_set
)
from playagain_pipeline.core.session import RecordingSession, RecordingTrial
from playagain_pipeline.core.data_manager_old import DataManager

# Protocols
from playagain_pipeline.protocols.protocol import (
    RecordingProtocol,
    ProtocolConfig,
    ProtocolPhase,
    create_quick_protocol,
    create_standard_protocol,
    create_extended_protocol
)

# Devices
from playagain_pipeline.devices.emg_device import (
    DeviceType,
    DeviceManager,
    SyntheticEMGDevice
)

# Models
from playagain_pipeline.models.classifier import (
    ModelManager,
    SVMClassifier,
    RandomForestClassifier,
    LDAClassifier,
    EMGFeatureExtractor
)

# Calibration
from playagain_pipeline.calibration.calibrator_old import (
    AutoCalibrator,
    CalibrationResult,
    CalibrationProcessor
)

# Configuration
from playagain_pipeline.config.config import (
    PipelineConfig,
    get_default_config
)

__all__ = [
    # Version info
    "__version__",
    "__author__",

    # Core
    "Gesture",
    "GestureSet",
    "GestureCategory",
    "create_default_gesture_set",
    "create_extended_gesture_set",
    "create_calibration_gesture_set",
    "RecordingSession",
    "RecordingTrial",
    "DataManager",

    # Protocols
    "RecordingProtocol",
    "ProtocolConfig",
    "ProtocolPhase",
    "create_quick_protocol",
    "create_standard_protocol",
    "create_extended_protocol",

    # Devices
    "DeviceType",
    "DeviceManager",
    "SyntheticEMGDevice",

    # Models
    "ModelManager",
    "SVMClassifier",
    "RandomForestClassifier",
    "LDAClassifier",
    "EMGFeatureExtractor",

    # Calibration
    "AutoCalibrator",
    "CalibrationResult",
    "CalibrationProcessor",

    # Configuration
    "PipelineConfig",
    "get_default_config",
]
