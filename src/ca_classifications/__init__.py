"""CA-classifications package."""

from .data import (
    SequenceClassificationDataModule,
    SequenceClassificationDataset,
    embed_sequences_to_records,
)
from .gp_classification import SequenceGPClassifier, SequenceGPModel, create_trainer

__all__ = [
    "SequenceGPClassifier",
    "SequenceGPModel",
    "create_trainer",
    "SequenceClassificationDataset",
    "SequenceClassificationDataModule",
    "embed_sequences_to_records",
]
