"""landscapy-ml package."""

from .data import (
    SequenceClassificationDataModule,
    SequenceClassificationDataset,
    embed_sequences_to_records,
    embed_sequences,
)
from .mlp_classification import SequenceMLPClassifier, SequenceMLPEnsembleClassifier
from .trainer import TrainingJob, create_trainer, register_model, register_data
from .adapters import (
    LandscapeInputAdapter,
    LandscapeOutputAdapter,
    ModelAdapter,
    register_input_adapter,
    register_layer_adapter,
    register_model_adapter,
    register_model_layer_mapping,
    register_output_adapter,
)
from .inference import predict_landscape_records, predict_sequences
from .data_utils import (
    build_config_from_dataframe,
    build_config_from_csv,
    write_config,
    CSVConfigRequest,
)
from .landscape_adapter import (
    records_from_landscape,
    datamodule_from_landscape,
)

__all__ = [
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
    "ModelAdapter",
    "register_model_adapter",
    "LandscapeInputAdapter",
    "LandscapeOutputAdapter",
    "register_input_adapter",
    "register_output_adapter",
    "register_model_layer_mapping",
    "register_layer_adapter",
    "build_config_from_dataframe",
    "build_config_from_csv",
    "write_config",
    "CSVConfigRequest",
    "records_from_landscape",
    "datamodule_from_landscape",
]
