"""
Configuration management for gesture pipeline.

Provides centralized configuration for all pipeline components.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import json


@dataclass
class DeviceConfig:
    """Configuration for EMG device."""
    device_type: str = "synthetic"
    num_channels: int = 32
    sampling_rate: int = 2000
    samples_per_frame: int = 100
    ip_address: str = "127.0.0.1"
    port: int = 31000


@dataclass
class RecordingConfig:
    """Configuration for recording sessions."""
    window_size_ms: int = 200
    window_stride_ms: int = 50
    default_protocol: str = "standard"
    default_gesture_set: str = "default"
    auto_save: bool = True


@dataclass
class CalibrationConfig:
    """Configuration for calibration."""
    enabled: bool = True
    num_calibration_gestures: int = 5
    calibration_duration: float = 2.0
    min_confidence_threshold: float = 0.7


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    default_model_type: str = "svm"
    test_ratio: float = 0.2
    cross_validation_folds: int = 5

    # SVM parameters
    svm_kernel: str = "rbf"
    svm_c: float = 1.0

    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None

    # LDA parameters
    lda_solver: str = "svd"


@dataclass
class PipelineConfig:
    """Master configuration for the entire pipeline."""
    # Paths
    data_dir: str = "data"

    # Component configs
    device: DeviceConfig = field(default_factory=DeviceConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # GUI settings
    display_time: float = 5.0
    update_rate_hz: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data_dir": self.data_dir,
            "device": {
                "device_type": self.device.device_type,
                "num_channels": self.device.num_channels,
                "sampling_rate": self.device.sampling_rate,
                "samples_per_frame": self.device.samples_per_frame,
                "ip_address": self.device.ip_address,
                "port": self.device.port
            },
            "recording": {
                "window_size_ms": self.recording.window_size_ms,
                "window_stride_ms": self.recording.window_stride_ms,
                "default_protocol": self.recording.default_protocol,
                "default_gesture_set": self.recording.default_gesture_set,
                "auto_save": self.recording.auto_save
            },
            "calibration": {
                "enabled": self.calibration.enabled,
                "num_calibration_gestures": self.calibration.num_calibration_gestures,
                "calibration_duration": self.calibration.calibration_duration,
                "min_confidence_threshold": self.calibration.min_confidence_threshold
            },
            "model": {
                "default_model_type": self.model.default_model_type,
                "test_ratio": self.model.test_ratio,
                "cross_validation_folds": self.model.cross_validation_folds,
                "svm_kernel": self.model.svm_kernel,
                "svm_c": self.model.svm_c,
                "rf_n_estimators": self.model.rf_n_estimators,
                "rf_max_depth": self.model.rf_max_depth,
                "lda_solver": self.model.lda_solver
            },
            "display_time": self.display_time,
            "update_rate_hz": self.update_rate_hz
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create from dictionary."""
        config = cls()

        config.data_dir = data.get("data_dir", config.data_dir)

        if "device" in data:
            d = data["device"]
            config.device = DeviceConfig(
                device_type=d.get("device_type", config.device.device_type),
                num_channels=d.get("num_channels", config.device.num_channels),
                sampling_rate=d.get("sampling_rate", config.device.sampling_rate),
                samples_per_frame=d.get("samples_per_frame", config.device.samples_per_frame),
                ip_address=d.get("ip_address", config.device.ip_address),
                port=d.get("port", config.device.port)
            )

        if "recording" in data:
            r = data["recording"]
            config.recording = RecordingConfig(
                window_size_ms=r.get("window_size_ms", config.recording.window_size_ms),
                window_stride_ms=r.get("window_stride_ms", config.recording.window_stride_ms),
                default_protocol=r.get("default_protocol", config.recording.default_protocol),
                default_gesture_set=r.get("default_gesture_set", config.recording.default_gesture_set),
                auto_save=r.get("auto_save", config.recording.auto_save)
            )

        if "calibration" in data:
            c = data["calibration"]
            config.calibration = CalibrationConfig(
                enabled=c.get("enabled", config.calibration.enabled),
                num_calibration_gestures=c.get("num_calibration_gestures", config.calibration.num_calibration_gestures),
                calibration_duration=c.get("calibration_duration", config.calibration.calibration_duration),
                min_confidence_threshold=c.get("min_confidence_threshold", config.calibration.min_confidence_threshold)
            )

        if "model" in data:
            m = data["model"]
            config.model = ModelConfig(
                default_model_type=m.get("default_model_type", config.model.default_model_type),
                test_ratio=m.get("test_ratio", config.model.test_ratio),
                cross_validation_folds=m.get("cross_validation_folds", config.model.cross_validation_folds),
                svm_kernel=m.get("svm_kernel", config.model.svm_kernel),
                svm_c=m.get("svm_c", config.model.svm_c),
                rf_n_estimators=m.get("rf_n_estimators", config.model.rf_n_estimators),
                rf_max_depth=m.get("rf_max_depth", config.model.rf_max_depth),
                lda_solver=m.get("lda_solver", config.model.lda_solver)
            )

        config.display_time = data.get("display_time", config.display_time)
        config.update_rate_hz = data.get("update_rate_hz", config.update_rate_hz)

        return config

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# Global default configuration
DEFAULT_CONFIG = PipelineConfig()


def get_default_config() -> PipelineConfig:
    """Get the default pipeline configuration."""
    return PipelineConfig()
