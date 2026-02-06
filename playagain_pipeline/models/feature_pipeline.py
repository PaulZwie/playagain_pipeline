"""
Modular Feature Selection Pipeline.

Provides a flexible system for defining, selecting, and creating
EMG features for gesture classification.
"""

from abc import abstractmethod
from typing import Dict, Any, List, Type
from dataclasses import dataclass, field
import numpy as np

# Registry for custom features
_FEATURE_REGISTRY: Dict[str, Type['BaseFeatureExtractor']] = {}


def register_feature(name: str):
    """Decorator to register a custom feature extractor."""
    def decorator(cls):
        _FEATURE_REGISTRY[name] = cls
        return cls
    return decorator


def get_registered_features() -> Dict[str, Type['BaseFeatureExtractor']]:
    """Get all registered feature extractors."""
    return _FEATURE_REGISTRY.copy()


@dataclass
class FeatureConfig:
    """Configuration for a feature extractor."""
    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "params": self.params
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureConfig':
        return cls(**data)


class BaseFeatureExtractor:
    """Abstract base class for feature extractors."""

    def __init__(self, **params):
        self.params = params

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the feature."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the feature."""
        pass

    @property
    def num_features_per_channel(self) -> int:
        """Number of features produced per channel."""
        return 1

    @abstractmethod
    def compute(self, data: np.ndarray) -> np.ndarray:
        """Compute the feature from EMG data."""
        pass


@register_feature("rms")
class RMSFeature(BaseFeatureExtractor):
    """Root Mean Square feature."""
    @property
    def name(self) -> str:
        return "rms"
    @property
    def description(self) -> str:
        return "Root Mean Square - measures signal power"
    def compute(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 3:
            return np.sqrt(np.mean(data ** 2, axis=1))
        else:
            return np.sqrt(np.mean(data ** 2, axis=0))


@register_feature("mav")
class MAVFeature(BaseFeatureExtractor):
    """Mean Absolute Value feature."""
    @property
    def name(self) -> str:
        return "mav"
    @property
    def description(self) -> str:
        return "Mean Absolute Value"
    def compute(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 3:
            return np.mean(np.abs(data), axis=1)
        else:
            return np.mean(np.abs(data), axis=0)


@register_feature("var")
class VarianceFeature(BaseFeatureExtractor):
    """Variance feature."""
    @property
    def name(self) -> str:
        return "var"
    @property
    def description(self) -> str:
        return "Variance - signal variability"
    def compute(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 3:
            return np.var(data, axis=1)
        else:
            return np.var(data, axis=0)


@register_feature("wl")
class WaveformLengthFeature(BaseFeatureExtractor):
    """Waveform Length feature."""
    @property
    def name(self) -> str:
        return "wl"
    @property
    def description(self) -> str:
        return "Waveform Length"
    def compute(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 3:
            return np.sum(np.abs(np.diff(data, axis=1)), axis=1)
        else:
            return np.sum(np.abs(np.diff(data, axis=0)), axis=0)


@register_feature("zc")
class ZeroCrossingFeature(BaseFeatureExtractor):
    """Zero Crossing feature."""
    def __init__(self, threshold: float = 0.01, **params):
        super().__init__(**params)
        self.threshold = threshold
    @property
    def name(self) -> str:
        return "zc"
    @property
    def description(self) -> str:
        return "Zero Crossings"
    def compute(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 3:
            zc = np.zeros((data.shape[0], data.shape[2]))
            for i in range(data.shape[0]):
                for j in range(data.shape[2]):
                    signal = data[i, :, j]
                    zc[i, j] = np.sum((np.abs(np.diff(np.sign(signal))) > 0) & (np.abs(np.diff(signal)) > self.threshold))
            return zc
        else:
            return np.sum((np.abs(np.diff(np.sign(data), axis=0)) > 0) & (np.abs(np.diff(data, axis=0)) > self.threshold), axis=0)


@register_feature("ssc")
class SlopeSignChangeFeature(BaseFeatureExtractor):
    """Slope Sign Change feature."""
    def __init__(self, threshold: float = 0.01, **params):
        super().__init__(**params)
        self.threshold = threshold
    @property
    def name(self) -> str:
        return "ssc"
    @property
    def description(self) -> str:
        return "Slope Sign Changes"
    def compute(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 3:
            ssc = np.zeros((data.shape[0], data.shape[2]))
            for i in range(data.shape[0]):
                for j in range(data.shape[2]):
                    signal = data[i, :, j]
                    diff = np.diff(signal)
                    ssc[i, j] = np.sum((diff[:-1] * diff[1:] < 0) & (np.abs(diff[:-1] - diff[1:]) > self.threshold))
            return ssc
        else:
            diff = np.diff(data, axis=0)
            return np.sum((diff[:-1] * diff[1:] < 0) & (np.abs(diff[:-1] - diff[1:]) > self.threshold), axis=0)


@register_feature("iemg")
class IEMGFeature(BaseFeatureExtractor):
    """Integrated EMG feature."""
    @property
    def name(self) -> str:
        return "iemg"
    @property
    def description(self) -> str:
        return "Integrated EMG"
    def compute(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 3:
            return np.sum(np.abs(data), axis=1)
        else:
            return np.sum(np.abs(data), axis=0)


@register_feature("ssi")
class SSIFeature(BaseFeatureExtractor):
    """Simple Square Integral feature."""
    @property
    def name(self) -> str:
        return "ssi"
    @property
    def description(self) -> str:
        return "Simple Square Integral"
    def compute(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 3:
            return np.sum(data ** 2, axis=1)
        else:
            return np.sum(data ** 2, axis=0)


class FeaturePipeline:
    """Pipeline for managing and computing multiple features."""

    def __init__(self):
        self._features: List[BaseFeatureExtractor] = []
        self._feature_configs: List[FeatureConfig] = []

    def add_feature(self, name: str, enabled: bool = True, **params) -> None:
        """Add a feature to the pipeline."""
        if name not in _FEATURE_REGISTRY:
            raise ValueError(f"Unknown feature: {name}")
        feature_cls = _FEATURE_REGISTRY[name]
        feature = feature_cls(**params)
        self._features.append(feature)
        self._feature_configs.append(FeatureConfig(name=name, enabled=enabled, params=params))

    def remove_feature(self, index: int) -> None:
        """Remove a feature by index."""
        if 0 <= index < len(self._features):
            self._features.pop(index)
            self._feature_configs.pop(index)

    def move_feature(self, from_index: int, to_index: int) -> None:
        """Move a feature from one position to another."""
        if 0 <= from_index < len(self._features) and 0 <= to_index < len(self._features):
            feature = self._features.pop(from_index)
            config = self._feature_configs.pop(from_index)
            self._features.insert(to_index, feature)
            self._feature_configs.insert(to_index, config)

    def set_enabled(self, index: int, enabled: bool) -> None:
        """Enable or disable a feature by index."""
        if 0 <= index < len(self._feature_configs):
            self._feature_configs[index].enabled = enabled

    def get_features(self) -> List[FeatureConfig]:
        """Get list of feature configurations."""
        return self._feature_configs.copy()

    def compute(self, data: np.ndarray) -> np.ndarray:
        """Compute all enabled features."""
        feature_arrays = []
        for feature, config in zip(self._features, self._feature_configs):
            if config.enabled:
                features = feature.compute(data)
                feature_arrays.append(features)
        if not feature_arrays:
            raise ValueError("No features enabled in pipeline")
        return np.hstack(feature_arrays)

    def get_feature_names(self) -> List[str]:
        """Get names of all enabled features."""
        return [config.name for config in self._feature_configs if config.enabled]

    def get_num_features(self, num_channels: int) -> int:
        """Get total number of features that will be produced."""
        total = 0
        for feature, config in zip(self._features, self._feature_configs):
            if config.enabled:
                total += num_channels * feature.num_features_per_channel
        return total

    @classmethod
    def create_default(cls) -> 'FeaturePipeline':
        """Create a pipeline with default time-domain features."""
        pipeline = cls()
        for feature_name in ["rms", "mav", "var", "wl", "zc", "ssc"]:
            pipeline.add_feature(feature_name)
        return pipeline
