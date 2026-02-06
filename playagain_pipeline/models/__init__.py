"""Machine learning models for gesture classification."""

from playagain_pipeline.models.classifier import (
    ModelMetadata,
    BaseClassifier,
    EMGFeatureExtractor,
    SVMClassifier,
    RandomForestClassifier,
    LDAClassifier,
    ModelManager
)

from playagain_pipeline.models.feature_pipeline import (
    FeaturePipeline,
    BaseFeatureExtractor,
    FeatureConfig,
    register_feature,
    get_registered_features
)

__all__ = [
    "ModelMetadata",
    "BaseClassifier",
    "EMGFeatureExtractor",
    "SVMClassifier",
    "RandomForestClassifier",
    "LDAClassifier",
    "ModelManager",
    "FeaturePipeline",
    "BaseFeatureExtractor",
    "FeatureConfig",
    "register_feature",
    "get_registered_features"
]
