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
    svm_gamma: str = "scale"

    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = 10
    rf_min_samples_split: int = 2

    # LDA parameters
    lda_solver: str = "svd"
    lda_shrinkage: Optional[str] = None

    # CatBoost parameters
    catboost_iterations: int = 500
    catboost_learning_rate: float = 0.1
    catboost_depth: int = 6
    catboost_l2_leaf_reg: float = 3.0
    catboost_early_stopping: bool = True

    # MLP parameters
    mlp_hidden_layers: tuple = (128, 64)
    mlp_epochs: int = 1000
    mlp_batch_size: int = 32
    mlp_learning_rate: float = 0.0001
    mlp_optimizer: str = "adam"
    mlp_early_stopping: bool = True
    mlp_patience: int = 20
    mlp_dropout: float = 0.2

    # CNN parameters
    cnn_filters: str = "32, 64, 128"
    cnn_kernels: str = "5, 3, 3"
    cnn_fc_layers: str = "128"
    cnn_epochs: int = 100
    cnn_batch_size: int = 32
    cnn_learning_rate: float = 0.0005
    cnn_optimizer: str = "adam"
    cnn_early_stopping: bool = True
    cnn_patience: int = 5

    # Inception parameters
    inception_filters: str = "32, 64, 128"
    inception_kernels: str = "1, 3, 5"
    inception_fc_layers: str = "128"
    inception_epochs: int = 100
    inception_batch_size: int = 32
    inception_learning_rate: float = 0.0005
    inception_optimizer: str = "adam"
    inception_early_stopping: bool = True
    inception_patience: int = 8


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
                "svm_gamma": self.model.svm_gamma,
                "rf_n_estimators": self.model.rf_n_estimators,
                "rf_max_depth": self.model.rf_max_depth,
                "rf_min_samples_split": self.model.rf_min_samples_split,
                "lda_solver": self.model.lda_solver,
                "lda_shrinkage": self.model.lda_shrinkage,
                "catboost_iterations": self.model.catboost_iterations,
                "catboost_learning_rate": self.model.catboost_learning_rate,
                "catboost_depth": self.model.catboost_depth,
                "catboost_l2_leaf_reg": self.model.catboost_l2_leaf_reg,
                "catboost_early_stopping": self.model.catboost_early_stopping,
                "mlp_hidden_layers": list(self.model.mlp_hidden_layers),
                "mlp_epochs": self.model.mlp_epochs,
                "mlp_batch_size": self.model.mlp_batch_size,
                "mlp_learning_rate": self.model.mlp_learning_rate,
                "mlp_optimizer": self.model.mlp_optimizer,
                "mlp_early_stopping": self.model.mlp_early_stopping,
                "mlp_patience": self.model.mlp_patience,
                "mlp_dropout": self.model.mlp_dropout,
                "cnn_filters": self.model.cnn_filters,
                "cnn_kernels": self.model.cnn_kernels,
                "cnn_fc_layers": self.model.cnn_fc_layers,
                "cnn_epochs": self.model.cnn_epochs,
                "cnn_batch_size": self.model.cnn_batch_size,
                "cnn_learning_rate": self.model.cnn_learning_rate,
                "cnn_optimizer": self.model.cnn_optimizer,
                "cnn_early_stopping": self.model.cnn_early_stopping,
                "cnn_patience": self.model.cnn_patience,
                "inception_filters": self.model.inception_filters,
                "inception_kernels": self.model.inception_kernels,
                "inception_fc_layers": self.model.inception_fc_layers,
                "inception_epochs": self.model.inception_epochs,
                "inception_batch_size": self.model.inception_batch_size,
                "inception_learning_rate": self.model.inception_learning_rate,
                "inception_optimizer": self.model.inception_optimizer,
                "inception_early_stopping": self.model.inception_early_stopping,
                "inception_patience": self.model.inception_patience
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
                svm_gamma=m.get("svm_gamma", config.model.svm_gamma),
                rf_n_estimators=m.get("rf_n_estimators", config.model.rf_n_estimators),
                rf_max_depth=m.get("rf_max_depth", config.model.rf_max_depth),
                rf_min_samples_split=m.get("rf_min_samples_split", config.model.rf_min_samples_split),
                lda_solver=m.get("lda_solver", config.model.lda_solver),
                lda_shrinkage=m.get("lda_shrinkage", config.model.lda_shrinkage),
                catboost_iterations=m.get("catboost_iterations", config.model.catboost_iterations),
                catboost_learning_rate=m.get("catboost_learning_rate", config.model.catboost_learning_rate),
                catboost_depth=m.get("catboost_depth", config.model.catboost_depth),
                catboost_l2_leaf_reg=m.get("catboost_l2_leaf_reg", config.model.catboost_l2_leaf_reg),
                catboost_early_stopping=m.get("catboost_early_stopping", config.model.catboost_early_stopping),
                mlp_hidden_layers=tuple(m.get("mlp_hidden_layers", config.model.mlp_hidden_layers)),
                mlp_epochs=m.get("mlp_epochs", config.model.mlp_epochs),
                mlp_batch_size=m.get("mlp_batch_size", config.model.mlp_batch_size),
                mlp_learning_rate=m.get("mlp_learning_rate", config.model.mlp_learning_rate),
                mlp_optimizer=m.get("mlp_optimizer", config.model.mlp_optimizer),
                mlp_early_stopping=m.get("mlp_early_stopping", config.model.mlp_early_stopping),
                mlp_patience=m.get("mlp_patience", config.model.mlp_patience),
                mlp_dropout=m.get("mlp_dropout", config.model.mlp_dropout),
                cnn_filters=m.get("cnn_filters", config.model.cnn_filters),
                cnn_kernels=m.get("cnn_kernels", config.model.cnn_kernels),
                cnn_fc_layers=m.get("cnn_fc_layers", config.model.cnn_fc_layers),
                cnn_epochs=m.get("cnn_epochs", config.model.cnn_epochs),
                cnn_batch_size=m.get("cnn_batch_size", config.model.cnn_batch_size),
                cnn_learning_rate=m.get("cnn_learning_rate", config.model.cnn_learning_rate),
                cnn_optimizer=m.get("cnn_optimizer", config.model.cnn_optimizer),
                cnn_early_stopping=m.get("cnn_early_stopping", config.model.cnn_early_stopping),
                cnn_patience=m.get("cnn_patience", config.model.cnn_patience),
                inception_filters=m.get("inception_filters", config.model.inception_filters),
                inception_kernels=m.get("inception_kernels", config.model.inception_kernels),
                inception_fc_layers=m.get("inception_fc_layers", config.model.inception_fc_layers),
                inception_epochs=m.get("inception_epochs", config.model.inception_epochs),
                inception_batch_size=m.get("inception_batch_size", config.model.inception_batch_size),
                inception_learning_rate=m.get("inception_learning_rate", config.model.inception_learning_rate),
                inception_optimizer=m.get("inception_optimizer", config.model.inception_optimizer),
                inception_early_stopping=m.get("inception_early_stopping", config.model.inception_early_stopping),
                inception_patience=m.get("inception_patience", config.model.inception_patience)
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
