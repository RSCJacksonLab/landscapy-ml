"""CA-classifications package."""

from .data import (
    SequenceClassificationDataModule,
    SequenceClassificationDataset,
    embed_sequences_to_records,
    embed_sequences,
)
from .gp_classification import SequenceGPClassifier, SequenceGPModel, create_trainer
from .mlp_classification import SequenceMLPClassifier, SequenceMLPEnsembleClassifier
from .trainer import TrainingJob, register_model, register_data
from .inference import predict_landscape_records, predict_sequences
from .data_utils import (
    build_config_from_dataframe,
    build_config_from_csv,
    write_config,
    CSVConfigRequest,
)

__all__ = [
    "SequenceGPClassifier",
    "SequenceGPModel",
    "create_trainer",
    "SequenceClassificationDataset",
    "SequenceClassificationDataModule",
    "embed_sequences_to_records",
    "embed_sequences",
    "SequenceMLPClassifier",
    "SequenceMLPEnsembleClassifier",
    "TrainingJob",
    "register_model",
    "register_data",
    "predict_sequences",
    "predict_landscape_records",
    "build_config_from_dataframe",
    "build_config_from_csv",
    "write_config",
    "CSVConfigRequest",
]
