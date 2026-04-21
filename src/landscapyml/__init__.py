"""landscapy-ml package."""

from .data import (
    SequenceClassificationDataModule,
    SequenceClassificationDataset,
    embed_sequences_to_records,
    embed_sequences,
)
from .landscape_pipeline import (
    LandscapeDataModule,
    LandscapeDataset,
    LandscapeExport,
    export_landscape_records,
    make_fitness_target_getter,
    make_preferred_input_getter,
)
from .mlp_classification import SequenceMLPClassifier, SequenceMLPEnsembleClassifier
from .core.trainer import (
    TrainingJob,
    create_trainer,
    normalize_split_indices,
    register_model,
    register_data,
)
from .core.adaptor import (
    GraphTensorInputAdapter,
    LandscapeInputAdapter,
    LandscapeOutputAdapter,
    ModelAdapter,
    NodeIndexInputAdapter,
    register_input_adapter,
    register_layer_adapter,
    register_model_adapter,
    register_model_layer_mapping,
    register_output_adapter,
)
from .core.inference import (
    infer_fitness_layer_from_landscape,
    predict_landscape_records,
    predict_sequences,
)
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
from .landscape_regression import (
    LandscapeRegressionConfig,
    SplitIndices,
    available_landscape_regression_runners,
    discover_demo_csvs,
    register_landscape_regression_runner,
    run_landscape_regression,
    run_landscape_regression_csv,
)

__all__ = [
    "create_trainer",
    "SequenceClassificationDataset",
    "SequenceClassificationDataModule",
    "LandscapeDataset",
    "LandscapeDataModule",
    "LandscapeExport",
    "embed_sequences_to_records",
    "embed_sequences",
    "export_landscape_records",
    "make_preferred_input_getter",
    "make_fitness_target_getter",
    "SequenceMLPClassifier",
    "SequenceMLPEnsembleClassifier",
    "TrainingJob",
    "register_model",
    "register_data",
    "normalize_split_indices",
    "predict_sequences",
    "predict_landscape_records",
    "infer_fitness_layer_from_landscape",
    "ModelAdapter",
    "register_model_adapter",
    "GraphTensorInputAdapter",
    "LandscapeInputAdapter",
    "LandscapeOutputAdapter",
    "NodeIndexInputAdapter",
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
    "LandscapeRegressionConfig",
    "SplitIndices",
    "available_landscape_regression_runners",
    "discover_demo_csvs",
    "register_landscape_regression_runner",
    "run_landscape_regression",
    "run_landscape_regression_csv",
]
