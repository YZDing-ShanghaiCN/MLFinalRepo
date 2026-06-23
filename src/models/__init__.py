"""CNN-GAP image classification models."""

from src.models.rgb_classifier import RGBClassifier, build_model_from_config

__all__ = ["RGBClassifier", "build_model_from_config"]
