"""CA-classifications package."""

from .gp_classification import SequenceGPClassifier, SequenceGPModel, create_trainer

__all__ = ["SequenceGPClassifier", "SequenceGPModel", "create_trainer"]
