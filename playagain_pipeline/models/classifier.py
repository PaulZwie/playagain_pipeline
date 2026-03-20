"""
Machine learning models for gesture classification.

This module provides a modular framework for training and using
different ML models for EMG gesture recognition.
"""

import json
import logging
import pickle
import shutil
import time
import numpy as np

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from catboost import CatBoostClassifier as CatBoostWrapper
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier as RFClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from playagain_pipeline.models.feature_pipeline import get_registered_features

logger = logging.getLogger(__name__)

# Cache for resolved devices — avoids repeated detection and logging
_device_cache: dict = {}


def get_best_device() -> torch.device:
    """
    Automatically detect the best available compute device.

    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.
    Result is cached so detection (and logging) only happens once.
    """
    if "auto" in _device_cache:
        return _device_cache["auto"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Classifier] Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Classifier] Using Apple MPS (Metal Performance Shaders) device")
    else:
        device = torch.device("cpu")
        print("[Classifier] Using CPU device")

    _device_cache["auto"] = device
    return device


def resolve_device(requested: str = "auto") -> torch.device:
    """
    Resolve a device string to an actual torch.device.

    Results are cached per device string so detection only happens once.

    Args:
        requested: "auto", "cuda", "mps", or "cpu"

    Returns:
        A valid torch.device, falling back to CPU if the requested
        accelerator is unavailable.
    """
    if requested in _device_cache:
        return _device_cache[requested]

    if requested == "auto":
        device = get_best_device()
    elif requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif requested not in ("cuda", "mps", "cpu"):
        print(f"[Classifier] Unknown device '{requested}', falling back to auto-detection")
        device = get_best_device()
    elif requested != "cpu":
        print(f"[Classifier] Requested device '{requested}' is not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    _device_cache[requested] = device
    return device


def _build_optimizer(params, opt_name: str, lr: float, weight_decay: float = 0.0):
    """Create an optimizer from the project's supported optimizer names."""
    opt_name = (opt_name or "adam").lower()
    if opt_name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)


def _find_learning_rate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    opt_name: str,
    weight_decay: float = 0.0,
    start_lr: float = 1e-6,
    end_lr: float = 1.0,
    max_steps: int = 100,
    max_grad_norm: float = 0.0,
) -> float:
    """Run a lightweight LR range test and return a suggested learning rate."""
    if len(dataloader) == 0:
        return 1e-3

    original_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.train()
    model.to(device)

    optimizer = _build_optimizer(model.parameters(), opt_name, start_lr, weight_decay)
    steps = min(max_steps, len(dataloader))
    if steps <= 1:
        model.load_state_dict(original_state)
        model.to(device)
        return start_lr

    lr_mult = (end_lr / start_lr) ** (1 / (steps - 1))
    lr = start_lr
    smoothed_loss = None
    best_loss = float("inf")
    best_lr = start_lr
    beta = 0.98

    for step_idx, (inputs, targets) in enumerate(dataloader):
        if step_idx >= steps:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if not torch.isfinite(loss):
            break

        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        loss_value = float(loss.item())
        smoothed_loss = loss_value if smoothed_loss is None else beta * smoothed_loss + (1 - beta) * loss_value
        corrected_loss = smoothed_loss / (1 - beta ** (step_idx + 1))

        if corrected_loss < best_loss:
            best_loss = corrected_loss
            best_lr = lr

        if corrected_loss > 4 * best_loss:
            break

        lr *= lr_mult
        for group in optimizer.param_groups:
            group["lr"] = lr

    model.load_state_dict(original_state)
    model.to(device)
    return max(best_lr, start_lr)


def apply_bad_channel_strategy(
    data: np.ndarray,
    bad_channels: Optional[List[int]] = None,
    mode: str = "interpolate",
) -> np.ndarray:
    """Apply bad-channel handling to the last channel axis of an EMG array."""
    if bad_channels is None or len(bad_channels) == 0:
        return data
    if data.ndim < 2:
        return data

    n_ch = data.shape[-1]
    bad = sorted({int(ch) for ch in bad_channels if 0 <= int(ch) < n_ch})
    if not bad:
        return data

    out = data.copy()
    mode = (mode or "interpolate").lower()
    if mode == "zero":
        out[..., bad] = 0.0
        return out

    bad_set = set(bad)

    def _nearest_valid(ch_idx: int, step: int) -> Optional[int]:
        for offset in range(1, n_ch):
            cand = (ch_idx + step * offset) % n_ch
            if cand not in bad_set:
                return cand
        return None

    for ch_idx in bad:
        left = _nearest_valid(ch_idx, -1)
        right = _nearest_valid(ch_idx, 1)
        if left is None and right is None:
            out[..., ch_idx] = 0.0
        elif left is None:
            out[..., ch_idx] = data[..., right]
        elif right is None:
            out[..., ch_idx] = data[..., left]
        else:
            out[..., ch_idx] = 0.5 * (data[..., left] + data[..., right])
    return out




@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    name: str
    model_type: str
    created_at: datetime
    num_classes: int
    num_channels: int
    window_size_ms: int
    sampling_rate: int
    training_accuracy: float = 0.0
    validation_accuracy: float = 0.0
    class_names: Dict[int, str] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_history: Dict[str, List[float]] = field(default_factory=dict)
    bad_channels: List[int] = field(default_factory=list)  # Channels excluded during training
    bad_channel_mode: str = "interpolate"  # interpolate or zero
    features_extracted: bool = False  # Whether dataset had features pre-extracted
    feature_config: Optional[Dict[str, Any]] = None  # Feature config used for pre-extraction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "created_at": self.created_at.isoformat(),
            "num_classes": self.num_classes,
            "num_channels": self.num_channels,
            "window_size_ms": self.window_size_ms,
            "sampling_rate": self.sampling_rate,
            "training_accuracy": self.training_accuracy,
            "validation_accuracy": self.validation_accuracy,
            "class_names": self.class_names,
            "hyperparameters": self.hyperparameters,
            "training_history": self.training_history,
            "bad_channels": self.bad_channels,
            "bad_channel_mode": self.bad_channel_mode,
            "features_extracted": self.features_extracted,
            "feature_config": self.feature_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        # Backward compat: older models may not have bad_channels
        if "bad_channels" not in data:
            data["bad_channels"] = []
        if "bad_channel_mode" not in data:
            data["bad_channel_mode"] = "interpolate"
        # Backward compat: older models may not have feature fields
        if "features_extracted" not in data:
            data["features_extracted"] = False
        if "feature_config" not in data:
            data["feature_config"] = None
        # Convert string keys back to int for class_names
        data["class_names"] = {int(k): v for k, v in data["class_names"].items()}
        return cls(**data)


class BaseClassifier(ABC):
    """
    Abstract base class for gesture classifiers.

    Provides a unified interface for different ML models.
    """

    def __init__(self, name: str = "base_classifier"):
        self.name = name
        self.metadata: Optional[ModelMetadata] = None
        self._model = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @abstractmethod
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract features from raw EMG windows.

        Args:
            X: Raw EMG data (windows, samples, channels)

        Returns:
            Feature array
        """
        pass

    def _adapt_features_for_scaler(self, features: np.ndarray, scaler: Any) -> np.ndarray:
        """
        Adapt feature dimensions to match a fitted scaler when a
        mismatch occurs. Zero-pads missing features or truncates excess.
        """
        if scaler is None or not hasattr(scaler, 'n_features_in_'):
            return features
        expected = scaler.n_features_in_
        actual = features.shape[1]
        if actual == expected:
            return features
        if actual < expected:
            # Zero-pad to match expected dimension
            padded = np.zeros((features.shape[0], expected), dtype=features.dtype)
            padded[:, :actual] = features
            return padded
        else:
            # Truncate
            return features[:, :expected]

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            **kwargs: Additional training parameters

        Returns:
            Training results/history
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input data

        Returns:
            Predicted labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Input data

        Returns:
            Class probabilities
        """
        pass

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        if self.metadata:
            with open(path / "metadata.json", 'w') as f:
                json.dump(self.metadata.to_dict(), f, indent=2)

        # Save model
        with open(path / "model.pkl", 'wb') as f:
            pickle.dump(self._model, f)

    def load(self, path: Path) -> None:
        """Load model from disk."""
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            self.metadata = ModelMetadata.from_dict(json.load(f))

        # Load model
        with open(path / "model.pkl", 'rb') as f:
            self._model = pickle.load(f)

        self._is_trained = True


class EMGFeatureExtractor:
    """
    Extracts time-domain and frequency-domain features from EMG signals.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature extractor.

        Args:
            config: Feature configuration dictionary with keys:
                    - mode: "default", "raw", or "custom"
                    - features: list of feature names if mode is "custom"
        """
        self.config = config or {"mode": "default", "features": []}

    @staticmethod
    def compute_rms(data: np.ndarray, axis: int = 1) -> np.ndarray:
        """Compute Root Mean Square."""
        return np.sqrt(np.mean(data ** 2, axis=axis))

    @staticmethod
    def compute_mav(data: np.ndarray, axis: int = 1) -> np.ndarray:
        """Compute Mean Absolute Value."""
        return np.mean(np.abs(data), axis=axis)

    @staticmethod
    def compute_var(data: np.ndarray, axis: int = 1) -> np.ndarray:
        """Compute Variance."""
        return np.var(data, axis=axis)

    @staticmethod
    def compute_wl(data: np.ndarray, axis: int = 1) -> np.ndarray:
        """Compute Waveform Length."""
        return np.sum(np.abs(np.diff(data, axis=axis)), axis=axis)

    @staticmethod
    def compute_zc(data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Compute Zero Crossings (fully vectorized)."""
        if data.ndim == 3:
            sign_diff = np.abs(np.diff(np.sign(data), axis=1))
            val_diff = np.abs(np.diff(data, axis=1))
            return np.sum((sign_diff > 0) & (val_diff > threshold), axis=1)
        else:
            return np.sum(
                (np.abs(np.diff(np.sign(data), axis=0)) > 0) &
                (np.abs(np.diff(data, axis=0)) > threshold),
                axis=0
            )

    @staticmethod
    def compute_ssc(data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Compute Slope Sign Changes (fully vectorized)."""
        if data.ndim == 3:
            diff = np.diff(data, axis=1)
            return np.sum(
                (diff[:, :-1, :] * diff[:, 1:, :] < 0) &
                (np.abs(diff[:, :-1, :] - diff[:, 1:, :]) > threshold),
                axis=1
            )
        else:
            diff = np.diff(data, axis=0)
            return np.sum(
                (diff[:-1] * diff[1:] < 0) &
                (np.abs(diff[:-1] - diff[1:]) > threshold),
                axis=0
            )

    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract features based on configuration.

        Args:
            data: EMG data (windows, samples, channels) or pre-extracted (windows, features)

        Returns:
            Feature array (windows, features)
        """
        # If data is already 2D, assume features were pre-extracted
        if data.ndim == 2:
            return data

        if data.ndim != 3:
            raise ValueError("Expected 3D array (windows, samples, channels) or 2D (windows, features)")

        mode = self.config.get("mode", "default")

        if mode == "raw":
            # Flatten the window: (windows, samples, channels) -> (windows, samples*channels)
            return data.reshape(data.shape[0], -1)

        elif mode == "custom":
            features = []
            selected = self.config.get("features", [])

            # Map of available features
            feature_map = {
                "rms": self.compute_rms,
                "mav": self.compute_mav,
                "var": self.compute_var,
                "wl": self.compute_wl,
                "zc": self.compute_zc,
                "ssc": self.compute_ssc
            }

            # If no features selected but custom mode, fallback to RMS
            if not selected:
                return self.compute_rms(data)

            for name in selected:
                name_lower = name.lower()
                if name_lower in feature_map:
                    features.append(feature_map[name_lower](data))
                else:
                    # Maybe it's a registered custom feature from feature_pipeline
                    registered = get_registered_features()
                    if name in registered:
                        extractor = registered[name]()
                        features.append(extractor.compute(data))

            if not features:
                return self.compute_rms(data)

            return np.hstack(features)

        else: # Default
            return self.extract_all_features(data)

    @classmethod
    def extract_all_features(cls, data: np.ndarray) -> np.ndarray:
        """
        Extract all time-domain features.

        Args:
            data: EMG data (windows, samples, channels) or pre-extracted (windows, features)

        Returns:
            Feature array (windows, features)
        """
        if data.ndim == 2:
            return data  # Already extracted
        if data.ndim != 3:
            raise ValueError("Expected 3D array (windows, samples, channels)")

        features = []

        # Time-domain features
        features.append(cls.compute_rms(data))
        features.append(cls.compute_mav(data))
        features.append(cls.compute_var(data))
        features.append(cls.compute_wl(data))
        features.append(cls.compute_zc(data))
        features.append(cls.compute_ssc(data))

        # Stack features
        return np.hstack(features)


class SVMClassifier(BaseClassifier):
    """
    Support Vector Machine classifier for gesture recognition.
    """

    def __init__(self, name: str = "svm_classifier", **kwargs):
        super().__init__(name)
        self.hyperparameters = {
            "kernel": kwargs.get("kernel", "rbf"),
            "C": kwargs.get("C", 1.0),
            "gamma": kwargs.get("gamma", "scale"),
            "probability": True,
            "feature_config": kwargs.get("feature_config", None)
        }
        self._feature_extractor = EMGFeatureExtractor(self.hyperparameters.get("feature_config"))

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features from raw EMG."""
        return self._feature_extractor.extract_features(X)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the SVM model."""
        start_time = time.time()

        # Extract features (skip if already 2D = pre-extracted)
        if X_train.ndim == 2:
            X_train_features = X_train
        else:
            X_train_features = self.extract_features(X_train)
        feature_count = X_train_features.shape[1]

        # Normalize features
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train_features)

        # Create and train model
        svm_params = {k: v for k, v in self.hyperparameters.items() if k != "feature_config"}
        self._model = SVC(**svm_params)
        self._model.fit(X_train_scaled, y_train)

        # Compute training accuracy
        train_pred = self._model.predict(X_train_scaled)
        train_acc = np.mean(train_pred == y_train)

        # Compute validation accuracy if provided
        val_acc = 0.0
        if X_val is not None and y_val is not None:
            X_val_features = X_val if X_val.ndim == 2 else self.extract_features(X_val)
            X_val_scaled = self._scaler.transform(X_val_features)
            val_pred = self._model.predict(X_val_scaled)
            val_acc = np.mean(val_pred == y_val)

        training_time = time.time() - start_time
        self._is_trained = True

        # Create metadata
        self.metadata = ModelMetadata(
            name=self.name,
            model_type="SVM",
            created_at=datetime.now(),
            num_classes=len(np.unique(y_train)),
            num_channels=kwargs.get("num_channels", X_train.shape[2] if X_train.ndim > 2 else 0),
            window_size_ms=kwargs.get("window_size_ms", 200),
            sampling_rate=kwargs.get("sampling_rate", 2000),
            training_accuracy=train_acc,
            validation_accuracy=val_acc,
            hyperparameters=self.hyperparameters
        )

        return {
            "training_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "feature_count": feature_count,
            "training_time": training_time
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._is_trained:
            raise ValueError("Model not trained")

        X_features = self.extract_features(X)
        X_features = self._adapt_features_for_scaler(X_features, self._scaler)
        X_scaled = self._scaler.transform(X_features)
        return self._model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self._is_trained:
            raise ValueError("Model not trained")

        X_features = self.extract_features(X)
        X_features = self._adapt_features_for_scaler(X_features, self._scaler)
        X_scaled = self._scaler.transform(X_features)
        return self._model.predict_proba(X_scaled)

    def save(self, path: Path) -> None:
        """Save model and scaler."""
        super().save(path)
        with open(Path(path) / "scaler.pkl", 'wb') as f:
            pickle.dump(self._scaler, f)

    def load(self, path: Path) -> None:
        """Load model and scaler."""
        super().load(path)
        with open(Path(path) / "scaler.pkl", 'rb') as f:
            self._scaler = pickle.load(f)


class RandomForestClassifier(BaseClassifier):
    """
    Random Forest classifier for gesture recognition.
    """

    def __init__(self, name: str = "rf_classifier", **kwargs):
        super().__init__(name)
        self.hyperparameters = {
            "n_estimators": kwargs.get("n_estimators", 100),
            "max_depth": kwargs.get("max_depth", None),
            "min_samples_split": kwargs.get("min_samples_split", 2),
            "min_samples_leaf": kwargs.get("min_samples_leaf", 1),
            "random_state": kwargs.get("random_state", 42),
            "n_jobs": kwargs.get("n_jobs", -1),  # Use all CPU cores
            "feature_config": kwargs.get("feature_config", None)
        }
        self._feature_extractor = EMGFeatureExtractor(self.hyperparameters.get("feature_config"))

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features from raw EMG."""
        return self._feature_extractor.extract_features(X)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the Random Forest model."""
        start_time = time.time()

        # Extract features (skip if already 2D = pre-extracted)
        if X_train.ndim == 2:
            X_train_features = X_train
        else:
            X_train_features = self.extract_features(X_train)
        feature_count = X_train_features.shape[1]

        # Create and train model
        rf_params = {k: v for k, v in self.hyperparameters.items() if k != "feature_config"}
        self._model = RFClassifier(**rf_params)
        self._model.fit(X_train_features, y_train)

        # Compute training accuracy
        train_pred = self._model.predict(X_train_features)
        train_acc = np.mean(train_pred == y_train)

        # Compute validation accuracy if provided
        val_acc = 0.0
        if X_val is not None and y_val is not None:
            X_val_features = X_val if X_val.ndim == 2 else self.extract_features(X_val)
            val_pred = self._model.predict(X_val_features)
            val_acc = np.mean(val_pred == y_val)

        training_time = time.time() - start_time
        self._is_trained = True

        # Create metadata
        self.metadata = ModelMetadata(
            name=self.name,
            model_type="RandomForest",
            created_at=datetime.now(),
            num_classes=len(np.unique(y_train)),
            num_channels=kwargs.get("num_channels", X_train.shape[2] if X_train.ndim > 2 else 0),
            window_size_ms=kwargs.get("window_size_ms", 200),
            sampling_rate=kwargs.get("sampling_rate", 2000),
            training_accuracy=train_acc,
            validation_accuracy=val_acc,
            hyperparameters=self.hyperparameters
        )

        return {
            "training_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "feature_count": feature_count,
            "training_time": training_time,
            "feature_importances": self._model.feature_importances_.tolist()
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained")
        X_features = self.extract_features(X)
        expected = getattr(self._model, 'n_features_in_', X_features.shape[1])
        if X_features.shape[1] != expected:
            X_features = self._adapt_features_for_scaler(X_features, self._model)
        return self._model.predict(X_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained")
        X_features = self.extract_features(X)
        expected = getattr(self._model, 'n_features_in_', X_features.shape[1])
        if X_features.shape[1] != expected:
            X_features = self._adapt_features_for_scaler(X_features, self._model)
        return self._model.predict_proba(X_features)


class LDAClassifier(BaseClassifier):
    """Linear Discriminant Analysis classifier for gesture recognition."""

    def __init__(self, name: str = "lda_classifier", **kwargs):
        super().__init__(name)
        self.hyperparameters = {
            "solver": kwargs.get("solver", "svd"),
            "shrinkage": kwargs.get("shrinkage", None),
            "feature_config": kwargs.get("feature_config", None)
        }
        self._feature_extractor = EMGFeatureExtractor(self.hyperparameters.get("feature_config"))

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        return self._feature_extractor.extract_features(X)

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        start_time = time.time()

        X_train_features = X_train if X_train.ndim == 2 else self.extract_features(X_train)
        feature_count = X_train_features.shape[1]

        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train_features)

        lda_params = {k: v for k, v in self.hyperparameters.items() if k != "feature_config"}
        self._model = LinearDiscriminantAnalysis(**lda_params)
        self._model.fit(X_train_scaled, y_train)

        train_pred = self._model.predict(X_train_scaled)
        train_acc = np.mean(train_pred == y_train)

        val_acc = 0.0
        if X_val is not None and y_val is not None:
            X_val_features = X_val if X_val.ndim == 2 else self.extract_features(X_val)
            X_val_scaled = self._scaler.transform(X_val_features)
            val_pred = self._model.predict(X_val_scaled)
            val_acc = np.mean(val_pred == y_val)

        training_time = time.time() - start_time
        self._is_trained = True

        self.metadata = ModelMetadata(
            name=self.name, model_type="LDA", created_at=datetime.now(),
            num_classes=len(np.unique(y_train)),
            num_channels=kwargs.get("num_channels", X_train.shape[2] if X_train.ndim > 2 else 0),
            window_size_ms=kwargs.get("window_size_ms", 200),
            sampling_rate=kwargs.get("sampling_rate", 2000),
            training_accuracy=float(train_acc), validation_accuracy=float(val_acc),
            hyperparameters=self.hyperparameters
        )
        return {"training_accuracy": train_acc, "validation_accuracy": val_acc,
                "feature_count": feature_count, "training_time": training_time}

    def predict(self, X):
        if not self._is_trained: raise ValueError("Model not trained")
        X_features = self.extract_features(X)
        X_features = self._adapt_features_for_scaler(X_features, self._scaler)
        return self._model.predict(self._scaler.transform(X_features))

    def predict_proba(self, X):
        if not self._is_trained: raise ValueError("Model not trained")
        X_features = self.extract_features(X)
        X_features = self._adapt_features_for_scaler(X_features, self._scaler)
        return self._model.predict_proba(self._scaler.transform(X_features))

    def save(self, path):
        super().save(path)
        with open(Path(path) / "scaler.pkl", 'wb') as f: pickle.dump(self._scaler, f)

    def load(self, path):
        super().load(path)
        with open(Path(path) / "scaler.pkl", 'rb') as f: self._scaler = pickle.load(f)


class MLPClassifier(BaseClassifier):
    """Multi-Layer Perceptron classifier with advanced training options."""

    def __init__(self, name: str = "mlp_classifier", **kwargs):
        super().__init__(name)
        self.hyperparameters = {
            "hidden_layers": kwargs.get("hidden_layers", (128, 64)),
            "activation": kwargs.get("activation", "relu"),
            "dropout": kwargs.get("dropout", 0.2),
            "optimizer": kwargs.get("optimizer", "adam"),
            "learning_rate": kwargs.get("learning_rate", 0.001),
            "weight_decay": kwargs.get("weight_decay", 0.0),
            "scheduler": kwargs.get("scheduler", "plateau"),  # none, cosine, plateau, step, warmup_cosine
            "batch_size": kwargs.get("batch_size", 32),
            "epochs": kwargs.get("epochs", 100),
            "early_stopping": kwargs.get("early_stopping", True),
            "patience": kwargs.get("patience", 10),
            "max_grad_norm": kwargs.get("max_grad_norm", 1.0),
            "device": kwargs.get("device", "auto"),
            "feature_config": kwargs.get("feature_config", None)
        }
        self._feature_extractor = EMGFeatureExtractor(self.hyperparameters.get("feature_config"))
        self._scaler = None
        self._model = None
        self._input_dim = 0
        self._output_dim = 0

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        return self._feature_extractor.extract_features(X)

    def _build_model(self, input_dim: int, output_dim: int):
        layers = []
        in_dim = input_dim
        for hidden_dim in self.hyperparameters["hidden_layers"]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if self.hyperparameters["activation"] == "relu":
                layers.append(nn.ReLU())
            elif self.hyperparameters["activation"] == "tanh":
                layers.append(nn.Tanh())
            if self.hyperparameters["dropout"] > 0:
                layers.append(nn.Dropout(self.hyperparameters["dropout"]))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        start_time = time.time()
        callback = kwargs.get("callback", None)

        X_train_features = X_train if X_train.ndim == 2 else self.extract_features(X_train)
        feature_count = X_train_features.shape[1]

        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train_features)

        X_val_scaled = None
        if X_val is not None:
            X_val_features = X_val if X_val.ndim == 2 else self.extract_features(X_val)
            X_val_scaled = self._scaler.transform(X_val_features)

        device = resolve_device(self.hyperparameters["device"])
        # Keep tensors on CPU for DataLoader with pin_memory=True
        # They will be moved to device inside the training loop
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_scaled) if X_val_scaled is not None else None
        y_val_tensor = torch.LongTensor(y_val) if y_val is not None else None

        self._input_dim = X_train_scaled.shape[1]
        self._output_dim = len(np.unique(y_train))
        self._model = self._build_model(self._input_dim, self._output_dim).to(device)

        opt_name = self.hyperparameters["optimizer"].lower()
        lr = self.hyperparameters["learning_rate"]
        wd = self.hyperparameters.get("weight_decay", 0.0)

        scheduler_name = self.hyperparameters.get("scheduler", "none").lower()
        scheduler = None
        total_epochs = self.hyperparameters["epochs"]
        optimizer = _build_optimizer(self._model.parameters(), opt_name, lr, wd)
        if scheduler_name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        elif scheduler_name == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
        elif scheduler_name == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, total_epochs // 3), gamma=0.1)
        elif scheduler_name == "warmup_cosine":
            warmup_ep = max(1, total_epochs // 10)
            warmup_s = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_ep)
            cosine_s = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_ep)
            scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_s, cosine_s], milestones=[warmup_ep])

        criterion = nn.CrossEntropyLoss()
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=self.hyperparameters["batch_size"], shuffle=True,
                                num_workers=0, pin_memory=(device.type != "cpu"))

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(total_epochs):
            self._model.train()
            train_loss = 0.0; correct = 0; total = 0
            max_grad_norm = self.hyperparameters.get("max_grad_norm", 1.0)
            for inputs, targets in dataloader:
                # Move to device here, not before DataLoader
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_grad_norm)
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_loss /= len(dataloader.dataset)
            train_acc = correct / total
            val_loss = 0.0; val_acc = 0.0

            if X_val_scaled is not None:
                self._model.eval()
                with torch.inference_mode():
                    # Move validation tensors to device
                    X_val_device = X_val_tensor.to(device)
                    y_val_device = y_val_tensor.to(device)
                    outputs = self._model(X_val_device)
                    loss = criterion(outputs, y_val_device)
                    val_loss = loss.item()
                    _, predicted = outputs.max(1)
                    val_acc = predicted.eq(y_val_device).sum().item() / y_val_device.size(0)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            if callback: callback(epoch + 1, train_loss, val_loss, train_acc, val_acc)

            if scheduler is not None:
                if scheduler_name == "plateau":
                    scheduler.step(val_loss if X_val_scaled is not None else train_loss)
                else:
                    scheduler.step()

            if self.hyperparameters["early_stopping"] and X_val_scaled is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss; patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.hyperparameters["patience"]:
                        if best_model_state is not None:
                            self._model.load_state_dict(best_model_state); self._model.to(device)
                        break

        self._is_trained = True
        self.metadata = ModelMetadata(
            name=self.name, model_type="MLP", created_at=datetime.now(),
            num_classes=self._output_dim,
            num_channels=kwargs.get("num_channels", X_train.shape[2] if X_train.ndim > 2 else 0),
            window_size_ms=kwargs.get("window_size_ms", 200),
            sampling_rate=kwargs.get("sampling_rate", 2000),
            training_accuracy=float(history["train_acc"][-1]),
            validation_accuracy=float(history["val_acc"][-1]) if history["val_acc"] else 0.0,
            hyperparameters=self.hyperparameters, training_history=history
        )
        return {"training_accuracy": history["train_acc"][-1],
                "validation_accuracy": history["val_acc"][-1] if history["val_acc"] else 0.0,
                "feature_count": feature_count, "training_time": time.time() - start_time,
                "epochs_trained": len(history["train_loss"])}

    def predict(self, X):
        if not self._is_trained: raise ValueError("Model not trained")
        X_features = self.extract_features(X)
        X_scaled = self._scaler.transform(self._adapt_features_for_scaler(X_features, self._scaler))
        device = resolve_device(self.hyperparameters["device"])
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        self._model.eval()
        with torch.inference_mode():
            _, predicted = self._model(X_tensor).max(1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        if not self._is_trained: raise ValueError("Model not trained")
        X_features = self.extract_features(X)
        X_scaled = self._scaler.transform(self._adapt_features_for_scaler(X_features, self._scaler))
        device = resolve_device(self.hyperparameters["device"])
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        self._model.eval()
        with torch.inference_mode():
            probs = torch.softmax(self._model(X_tensor), dim=1)
        return probs.cpu().numpy()

    def save(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        if self.metadata:
            with open(path / "metadata.json", 'w') as f: json.dump(self.metadata.to_dict(), f, indent=2)
        torch.save(self._model.state_dict(), path / "model.pt")
        with open(path / "scaler.pkl", 'wb') as f: pickle.dump(self._scaler, f)
        with open(path / "params.json", 'w') as f:
            json.dump({"input_dim": self._input_dim, "output_dim": self._output_dim,
                        "hyperparameters": self.hyperparameters}, f)

    def load(self, path):
        path = Path(path)
        with open(path / "metadata.json", 'r') as f: self.metadata = ModelMetadata.from_dict(json.load(f))
        with open(path / "params.json", 'r') as f:
            params = json.load(f)
            self._input_dim = params["input_dim"]; self._output_dim = params["output_dim"]
            self.hyperparameters = params["hyperparameters"]
        with open(path / "scaler.pkl", 'rb') as f: self._scaler = pickle.load(f)
        self._model = self._build_model(self._input_dim, self._output_dim)
        device = resolve_device(self.hyperparameters["device"])
        self._model.load_state_dict(torch.load(path / "model.pt", map_location=device))
        self._model.to(device)
        self._is_trained = True


class CNNClassifier(BaseClassifier):
    def __init__(self, name: str = "cnn_classifier", **kwargs):
        super().__init__(name)
        self.hyperparameters = {
            "filters": kwargs.get("filters", [32, 64, 128]),
            "kernel_sizes": kwargs.get("kernel_sizes", [5, 3, 3]),
            "fc_layers": kwargs.get("fc_layers", (128,)),
            "activation": kwargs.get("activation", "relu"),
            "dropout": kwargs.get("dropout", 0.3),
            "optimizer": kwargs.get("optimizer", "adam"),
            "learning_rate": kwargs.get("learning_rate", 0.0005),
            "weight_decay": kwargs.get("weight_decay", 0.0),
            "scheduler": kwargs.get("scheduler", "plateau"),  # none, cosine, plateau, step, warmup_cosine
            "batch_size": kwargs.get("batch_size", 32),
            "epochs": kwargs.get("epochs", 100),
            "early_stopping": kwargs.get("early_stopping", True),
            "patience": kwargs.get("patience", 10),
            "max_grad_norm": kwargs.get("max_grad_norm", 1.0),
            "device": kwargs.get("device", "auto")
        }
        self._scaler = None; self._model = None; self._input_shape = None; self._output_dim = 0

    def extract_features(self, X):
        if X.ndim == 3: return np.transpose(X, (0, 2, 1))
        return X

    def _build_model(self, input_channels, input_length, output_dim):
        layers = []; in_channels = input_channels; current_length = max(1, input_length)
        for out_channels, k_size in zip(self.hyperparameters["filters"], self.hyperparameters["kernel_sizes"]):
            layers.extend([nn.Conv1d(in_channels, out_channels, kernel_size=k_size, padding=k_size // 2),
                           nn.BatchNorm1d(out_channels)])
            if self.hyperparameters["activation"] == "relu": layers.append(nn.ReLU())
            elif self.hyperparameters["activation"] == "tanh": layers.append(nn.Tanh())
            if current_length >= 2: layers.append(nn.MaxPool1d(2)); current_length //= 2
            if self.hyperparameters["dropout"] > 0: layers.append(nn.Dropout(self.hyperparameters["dropout"]))
            in_channels = out_channels
        layers.extend([nn.AdaptiveAvgPool1d(1), nn.Flatten()])
        current_dim = in_channels
        for hidden_dim in self.hyperparameters["fc_layers"]:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if self.hyperparameters["activation"] == "relu": layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.hyperparameters["dropout"]))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        start_time = time.time()
        callback = kwargs.get("callback", None)
        if X_train.ndim == 2:
            X_train = X_train[:, np.newaxis, :]
            if X_val is not None and X_val.ndim == 2: X_val = X_val[:, np.newaxis, :]
        X_train_proc = self.extract_features(X_train)
        if X_train_proc.ndim == 2: X_train_proc = X_train_proc[:, np.newaxis, :]
        try: N, C, T = X_train_proc.shape
        except ValueError:
            if X_train_proc.ndim == 4:
                N, C, T = X_train_proc.shape[0], X_train_proc.shape[1]*X_train_proc.shape[2], X_train_proc.shape[3]
                X_train_proc = X_train_proc.reshape(N, C, T)
            else: raise ValueError(f"CNN expects 3D input (N, C, T), got {X_train_proc.shape}")

        self._scaler = StandardScaler()
        X_flat = X_train_proc.transpose(0, 2, 1).reshape(-1, C)
        X_scaled = self._scaler.fit_transform(X_flat).reshape(N, T, C).transpose(0, 2, 1)

        X_val_scaled = None
        if X_val is not None:
            X_val_proc = self.extract_features(X_val)
            Nv, Cv, Tv = X_val_proc.shape
            X_val_scaled = self._scaler.transform(X_val_proc.transpose(0, 2, 1).reshape(-1, Cv)).reshape(Nv, Tv, Cv).transpose(0, 2, 1)

        device = resolve_device(self.hyperparameters["device"])
        # Keep tensors on CPU for DataLoader with pin_memory=True
        # They will be moved to device inside the training loop
        X_train_tensor = torch.FloatTensor(X_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_scaled) if X_val_scaled is not None else None
        y_val_tensor = torch.LongTensor(y_val) if y_val is not None else None

        self._input_shape = (C, T); self._output_dim = len(np.unique(y_train))
        self._model = self._build_model(C, T, self._output_dim).to(device)

        lr = self.hyperparameters["learning_rate"]; wd = self.hyperparameters.get("weight_decay", 0.0)
        opt_name = self.hyperparameters["optimizer"]

        sched_name = self.hyperparameters.get("scheduler", "none").lower()
        sched = None; cnn_epochs = self.hyperparameters["epochs"]
        optimizer = _build_optimizer(self._model.parameters(), opt_name, lr, wd)
        if sched_name == "cosine": sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cnn_epochs)
        elif sched_name == "plateau": sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
        elif sched_name == "step": sched = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cnn_epochs // 3), gamma=0.1)
        elif sched_name == "warmup_cosine":
            warmup_ep = max(1, cnn_epochs // 10)
            warmup_s = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_ep)
            cosine_s = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cnn_epochs - warmup_ep)
            sched = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_s, cosine_s], milestones=[warmup_ep])

        criterion = nn.CrossEntropyLoss()
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=self.hyperparameters["batch_size"], shuffle=True,
                                num_workers=0, pin_memory=(device.type != "cpu"))
        best_val_loss = float('inf'); best_model_state = None; patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(cnn_epochs):
            self._model.train(); train_loss = 0.0; correct = 0; total = 0
            mgn = self.hyperparameters.get("max_grad_norm", 1.0)
            for inputs, targets in dataloader:
                # Move to device here, not before DataLoader
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self._model(inputs); loss = criterion(outputs, targets)
                loss.backward()
                if mgn > 0: torch.nn.utils.clip_grad_norm_(self._model.parameters(), mgn)
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1); total += targets.size(0); correct += predicted.eq(targets).sum().item()
            train_loss /= len(dataloader.dataset); train_acc = correct / total
            val_loss = 0.0; val_acc = 0.0
            if X_val_scaled is not None:
                self._model.eval()
                with torch.inference_mode():
                    # Move validation tensors to device
                    X_val_device = X_val_tensor.to(device)
                    y_val_device = y_val_tensor.to(device)
                    outputs = self._model(X_val_device); loss = criterion(outputs, y_val_device)
                    val_loss = loss.item()
                    _, predicted = outputs.max(1)
                    val_acc = predicted.eq(y_val_device).sum().item() / y_val_device.size(0)
            history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc); history["val_acc"].append(val_acc)
            if callback: callback(epoch + 1, train_loss, val_loss, train_acc, val_acc)
            if sched is not None:
                if sched_name == "plateau": sched.step(val_loss if X_val_scaled is not None else train_loss)
                else: sched.step()
            if self.hyperparameters["early_stopping"] and X_val_scaled is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss; patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.hyperparameters["patience"]:
                        if best_model_state: self._model.load_state_dict(best_model_state); self._model.to(device)
                        break

        self._is_trained = True
        self.metadata = ModelMetadata(
            name=self.name, model_type="CNN", created_at=datetime.now(),
            num_classes=self._output_dim,
            num_channels=kwargs.get("num_channels", X_train.shape[2] if X_train.ndim > 2 else 0),
            window_size_ms=kwargs.get("window_size_ms", 200),
            sampling_rate=kwargs.get("sampling_rate", 2000),
            training_accuracy=float(history["train_acc"][-1]),
            validation_accuracy=float(history["val_acc"][-1]) if history["val_acc"] else 0.0,
            hyperparameters=self.hyperparameters, training_history=history
        )
        return {"training_accuracy": history["train_acc"][-1],
                "validation_accuracy": history["val_acc"][-1] if history["val_acc"] else 0.0,
                "feature_count": C, "training_time": time.time() - start_time,
                "epochs_trained": len(history["train_loss"])}

    def predict(self, X):
        if not self._is_trained: raise ValueError("Model not trained")
        X_proc = self.extract_features(X); N, C, T = X_proc.shape
        expected_C = self._scaler.n_features_in_ if hasattr(self._scaler, 'n_features_in_') else C
        if C == expected_C:
            X_scaled = self._scaler.transform(X_proc.transpose(0, 2, 1).reshape(-1, C)).reshape(N, T, C).transpose(0, 2, 1)
        else: X_scaled = X_proc
        device = resolve_device(self.hyperparameters["device"])
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        self._model.eval()
        with torch.inference_mode(): _, predicted = self._model(X_tensor).max(1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        if not self._is_trained: raise ValueError("Model not trained")
        X_proc = self.extract_features(X); N, C, T = X_proc.shape
        expected_C = self._scaler.n_features_in_ if hasattr(self._scaler, 'n_features_in_') else C
        if C == expected_C:
            X_scaled = self._scaler.transform(X_proc.transpose(0, 2, 1).reshape(-1, C)).reshape(N, T, C).transpose(0, 2, 1)
        else: X_scaled = X_proc
        device = resolve_device(self.hyperparameters["device"])
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        self._model.eval()
        with torch.inference_mode(): probs = torch.softmax(self._model(X_tensor), dim=1)
        return probs.cpu().numpy()

    def save(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        if self.metadata:
            with open(path / "metadata.json", 'w') as f: json.dump(self.metadata.to_dict(), f, indent=2)
        torch.save(self._model.state_dict(), path / "model.pt")
        with open(path / "scaler.pkl", 'wb') as f: pickle.dump(self._scaler, f)
        with open(path / "params.json", 'w') as f:
            json.dump({"input_channels": self._input_shape[0] if self._input_shape else 0,
                        "input_length": self._input_shape[1] if self._input_shape else 0,
                        "output_dim": self._output_dim, "hyperparameters": self.hyperparameters}, f)

    def load(self, path):
        path = Path(path)
        with open(path / "metadata.json", 'r') as f: self.metadata = ModelMetadata.from_dict(json.load(f))
        with open(path / "params.json", 'r') as f:
            params = json.load(f)
            self._input_shape = (params["input_channels"], params["input_length"])
            self._output_dim = params["output_dim"]; self.hyperparameters = params["hyperparameters"]
        with open(path / "scaler.pkl", 'rb') as f: self._scaler = pickle.load(f)
        self._model = self._build_model(self._input_shape[0], self._input_shape[1], self._output_dim)
        device = resolve_device(self.hyperparameters["device"])
        self._model.load_state_dict(torch.load(path / "model.pt", map_location=device))
        self._model.to(device); self._is_trained = True


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        reduced = max(channels // reduction, 1)
        self.fc = nn.Sequential(nn.Linear(channels, reduced, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(reduced, channels, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _ = x.size(); y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1).expand_as(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, base_channels, branch_kernels=(3, 5)):
        super().__init__()
        self._kernels = tuple(branch_kernels) if isinstance(branch_kernels, (list, tuple)) else (branch_kernels,)
        if not self._kernels: self._kernels = (3, 5)
        n_conv = len(self._kernels); base = base_channels
        out_1x1 = max(base // 2, 1)
        self.branch_1x1 = nn.Sequential(nn.Conv1d(in_channels, out_1x1, kernel_size=1), nn.ReLU())
        per_branch_out = max(base // n_conv, 1); per_branch_red = max(base // (2 * n_conv), 1)
        self.conv_branches = nn.ModuleList()
        for k in self._kernels:
            self.conv_branches.append(nn.Sequential(
                nn.Conv1d(in_channels, per_branch_red, kernel_size=1), nn.ReLU(),
                nn.Conv1d(per_branch_red, per_branch_out, kernel_size=k, padding=k // 2), nn.ReLU()))
        pool_proj = max(base // 4, 1)
        self.branch_pool = nn.Sequential(nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                                         nn.Conv1d(in_channels, pool_proj, kernel_size=1), nn.ReLU())
        self.out_channels = out_1x1 + per_branch_out * n_conv + pool_proj
    def forward(self, x):
        outputs = [self.branch_1x1(x)]
        for branch in self.conv_branches: outputs.append(branch(x))
        outputs.append(self.branch_pool(x))
        return torch.cat(outputs, dim=1)


class AttentionNet(nn.Module):
    def __init__(self, input_channels, output_dim, inception_channels=32,
                 branch_kernels=(3, 5), reduction_ratio=8, dropout=0.3):
        super().__init__()
        base = inception_channels
        self.stem = nn.Sequential(nn.Conv1d(input_channels, base, kernel_size=7, stride=2, padding=3),
                                  nn.BatchNorm1d(base), nn.ReLU())
        self.inception = InceptionBlock(in_channels=base, base_channels=base, branch_kernels=branch_kernels)
        total_inception_out = self.inception.out_channels
        self.attention = ChannelAttention(total_inception_out, reduction=reduction_ratio)
        self.pool = nn.AdaptiveAvgPool1d(1); self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout); self.classifier = nn.Linear(total_inception_out, output_dim)
    def forward(self, x):
        x = self.stem(x); x = self.inception(x); x = self.attention(x)
        return self.classifier(self.dropout(self.flatten(self.pool(x))))


class AttentionNetClassifier(CNNClassifier):
    def __init__(self, name="attention_net_classifier", **kwargs):
        super().__init__(name, **kwargs)
        self.hyperparameters.update({
            "reduction_ratio": kwargs.get("reduction_ratio", 8),
            "inception_channels": kwargs.get("inception_channels", 32),
            "branch_kernels": kwargs.get("branch_kernels", [3, 5]),
        })
    def _build_model(self, input_channels, input_length, output_dim):
        bk = self.hyperparameters.get("branch_kernels", [3, 5])
        if isinstance(bk, list): bk = tuple(bk)
        return AttentionNet(input_channels=input_channels, output_dim=output_dim,
                            inception_channels=self.hyperparameters.get("inception_channels", 32),
                            branch_kernels=bk,
                            reduction_ratio=self.hyperparameters.get("reduction_ratio", 8),
                            dropout=self.hyperparameters.get("dropout", 0.3))
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        results = super().train(X_train, y_train, X_val, y_val, **kwargs)
        if self.metadata: self.metadata.model_type = "AttentionNet"
        return results


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Scale Temporal Network (MSTNet)
# ═══════════════════════════════════════════════════════════════════════════

class _SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(channels, mid, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(mid, channels, bias=False), nn.Sigmoid())
    def forward(self, x):
        w = self.pool(x).squeeze(-1); w = self.fc(w).unsqueeze(-1)
        return x * w


class _MultiScaleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernels=(3, 7, 15), dropout=0.2):
        super().__init__()
        n_branches = len(kernels); branch_ch = max(out_ch // n_branches, 1)
        leftover = out_ch - branch_ch * (n_branches - 1)
        self.branches = nn.ModuleList()
        for i, k in enumerate(kernels):
            ch = leftover if i == 0 else branch_ch
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_ch, ch, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(ch), nn.GELU()))
        self.se = _SEBlock(out_ch); self.drop = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches], dim=1)
        out = self.drop(self.se(out))
        target_T = max(1, out.size(2) // 2)
        out = torch.nn.functional.adaptive_max_pool1d(out, target_T)
        res = torch.nn.functional.adaptive_max_pool1d(self.residual(x), target_T)
        return out + res


class _TemporalAttentionBlock(nn.Module):
    """Multi-Head Temporal Attention for 1-D feature sequences.
    Operates on (B, C, T) tensors, treating each temporal position as a token."""

    def __init__(self, embed_dim, num_heads=4, dropout=0.1, max_len=512):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(embed_dim * 2, embed_dim), nn.Dropout(dropout))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,C,T) -> (B,T,C)
        T = x.size(1)
        x = x + self.pos_embed[:, :T, :]
        x_n = self.norm1(x)
        attn_out, _ = self.attn(x_n, x_n, x_n)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x.permute(0, 2, 1)  # back to (B,C,T)


class MSTNet(nn.Module):
    """Multi-Scale Temporal Network with optional Temporal Multi-Head Attention."""

    def __init__(self, input_channels, output_dim, base_filters=48, kernels=(3, 7, 15),
                 num_blocks=3, dropout=0.25, use_temporal_attention=False, attention_heads=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(input_channels, base_filters, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(base_filters), nn.GELU())
        blocks = []; in_ch = base_filters
        for i in range(num_blocks):
            out_ch = base_filters * (2 ** i)
            blocks.append(_MultiScaleBlock(in_ch, out_ch, kernels=kernels, dropout=dropout))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.temporal_attention = None
        if use_temporal_attention:
            self.temporal_attention = _TemporalAttentionBlock(embed_dim=in_ch, num_heads=attention_heads, dropout=dropout)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, output_dim))

    def forward(self, x):
        x = self.stem(x); x = self.blocks(x)
        if self.temporal_attention is not None: x = self.temporal_attention(x)
        return self.head(x)


class MSTNetClassifier(CNNClassifier):
    def __init__(self, name="mstnet_classifier", **kwargs):
        super().__init__(name, **kwargs)
        self.hyperparameters.update({
            "base_filters": kwargs.get("base_filters", 48),
            "ms_kernels": kwargs.get("ms_kernels", [3, 7, 15]),
            "num_blocks": kwargs.get("num_blocks", 3),
            "use_temporal_attention": kwargs.get("use_temporal_attention", False),
            "attention_heads": kwargs.get("attention_heads", 4),
        })
    def _build_model(self, input_channels, input_length, output_dim):
        return MSTNet(input_channels=input_channels, output_dim=output_dim,
                      base_filters=self.hyperparameters.get("base_filters", 48),
                      kernels=tuple(self.hyperparameters.get("ms_kernels", [3, 7, 15])),
                      num_blocks=self.hyperparameters.get("num_blocks", 3),
                      dropout=self.hyperparameters.get("dropout", 0.25),
                      use_temporal_attention=self.hyperparameters.get("use_temporal_attention", False),
                      attention_heads=self.hyperparameters.get("attention_heads", 4))
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        results = super().train(X_train, y_train, X_val, y_val, **kwargs)
        if self.metadata: self.metadata.model_type = "MSTNet"
        return results


class CatBoostClassifier(BaseClassifier):
    def __init__(self, name="catboost_classifier", **kwargs):
        super().__init__(name)
        self.hyperparameters = {
            "iterations": kwargs.get("iterations", 1000),
            "learning_rate": kwargs.get("learning_rate", 0.03),
            "depth": kwargs.get("depth", 6),
            "l2_leaf_reg": kwargs.get("l2_leaf_reg", 3),
            "loss_function": kwargs.get("loss_function", "MultiClass"),
            "verbose": kwargs.get("verbose", False),
            "task_type": kwargs.get("task_type", "auto"),
            "early_stopping_rounds": kwargs.get("early_stopping_rounds", 50),
            "feature_config": kwargs.get("feature_config", None)
        }
        self._feature_extractor = EMGFeatureExtractor(self.hyperparameters.get("feature_config"))
        self._scaler = None

    def extract_features(self, X): return self._feature_extractor.extract_features(X)

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        start_time = time.time()
        X_train_features = X_train if X_train.ndim == 2 else self.extract_features(X_train)
        feature_count = X_train_features.shape[1]
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train_features)
        eval_set = None; X_val_scaled = None
        if X_val is not None and y_val is not None:
            X_val_features = X_val if X_val.ndim == 2 else self.extract_features(X_val)
            X_val_scaled = self._scaler.transform(X_val_features)
            eval_set = (X_val_scaled, y_val)
        catboost_params = {k: v for k, v in self.hyperparameters.items() if k != "feature_config"}
        if catboost_params.get("task_type") == "auto":
            catboost_params["task_type"] = "GPU" if torch.cuda.is_available() else "CPU"
        self._model = CatBoostWrapper(**catboost_params)
        self._model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=self.hyperparameters.get("verbose", False))
        train_pred = self._model.predict(X_train_scaled).flatten()
        train_acc = np.mean(train_pred == y_train)
        val_acc = 0.0
        if eval_set is not None:
            val_pred = self._model.predict(X_val_scaled).flatten()
            val_acc = np.mean(val_pred == y_val)
        self._is_trained = True
        self.metadata = ModelMetadata(
            name=self.name, model_type="CatBoost", created_at=datetime.now(),
            num_classes=len(np.unique(y_train)),
            num_channels=kwargs.get("num_channels", X_train.shape[2] if X_train.ndim > 2 else 0),
            window_size_ms=kwargs.get("window_size_ms", 200),
            sampling_rate=kwargs.get("sampling_rate", 2000),
            training_accuracy=float(train_acc), validation_accuracy=float(val_acc),
            hyperparameters=self.hyperparameters)
        return {"training_accuracy": train_acc, "validation_accuracy": val_acc,
                "feature_count": feature_count, "training_time": time.time() - start_time}

    def predict(self, X):
        if not self._is_trained: raise ValueError("Model not trained")
        X_features = self._adapt_features_for_scaler(self.extract_features(X), self._scaler)
        return self._model.predict(self._scaler.transform(X_features)).flatten().astype(int)

    def predict_proba(self, X):
        if not self._is_trained: raise ValueError("Model not trained")
        X_features = self._adapt_features_for_scaler(self.extract_features(X), self._scaler)
        return self._model.predict_proba(self._scaler.transform(X_features))

    def save(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        if self.metadata:
            with open(path / "metadata.json", 'w') as f: json.dump(self.metadata.to_dict(), f, indent=2)
        self._model.save_model(str(path / "model.cbm"))
        with open(path / "scaler.pkl", 'wb') as f: pickle.dump(self._scaler, f)

    def load(self, path):
        path = Path(path)
        with open(path / "metadata.json", 'r') as f: self.metadata = ModelMetadata.from_dict(json.load(f))
        self._model = CatBoostWrapper()
        if (path / "model.cbm").exists(): self._model.load_model(str(path / "model.cbm"))
        elif (path / "model.pkl").exists():
            with open(path / "model.pkl", 'rb') as f: self._model = pickle.load(f)
        scaler_path = path / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f: self._scaler = pickle.load(f)
        else: self._scaler = StandardScaler()
        self._is_trained = True


class ModelManager:
    AVAILABLE_MODELS = {
        "svm": SVMClassifier, "random_forest": RandomForestClassifier,
        "lda": LDAClassifier, "catboost": CatBoostClassifier,
        "mlp": MLPClassifier, "cnn": CNNClassifier,
        "attention_net": AttentionNetClassifier, "mstnet": MSTNetClassifier,
    }

    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._current_model = None

    def create_model(self, model_type, name=None, **kwargs):
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.AVAILABLE_MODELS.keys())}")
        if name is None: name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model = self.AVAILABLE_MODELS[model_type](name=name, **kwargs)
        self._current_model = model
        return model

    def train_model(self, model, dataset, test_ratio=0.2, save=True):
        X, y = dataset["X"], dataset["y"]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=42)
        results = model.train(X_train, y_train, X_val, y_val,
                              window_size_ms=dataset["metadata"]["window_size_ms"],
                              sampling_rate=dataset["metadata"].get("sampling_rate", 2000),
                              num_channels=dataset["metadata"].get("num_channels", 0))
        if model.metadata and "label_names" in dataset["metadata"]:
            model.metadata.class_names = dataset["metadata"]["label_names"]
        if model.metadata:
            model.metadata.bad_channel_mode = dataset["metadata"].get("bad_channel_mode", "interpolate")
            model.metadata.features_extracted = dataset["metadata"].get("features_extracted", False)
            model.metadata.feature_config = dataset["metadata"].get("feature_config", None)
        if save: model.save(self.models_dir / model.name)
        self._current_model = model
        return results

    def load_model(self, name, model_type=None):
        model_path = self.models_dir / name
        if model_type is None:
            with open(model_path / "metadata.json", 'r') as f: metadata = json.load(f)
            model_type = metadata["model_type"].lower()
            if model_type == "randomforest": model_type = "random_forest"
            elif model_type == "attentionnet": model_type = "attention_net"
        model = self.create_model(model_type, name=name)
        model.load(model_path)
        self._current_model = model
        return model

    def list_models(self): return [d.name for d in self.models_dir.iterdir() if d.is_dir()]

    def delete_model(self, name):
        model_path = self.models_dir / name
        if model_path.exists() and model_path.is_dir():
            if self._current_model and hasattr(self._current_model, 'name') and self._current_model.name == name:
                self._current_model = None
            shutil.rmtree(model_path); return True
        return False

    @property
    def current_model(self): return self._current_model
