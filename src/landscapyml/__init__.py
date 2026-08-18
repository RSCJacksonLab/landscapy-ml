"""landscapy-ml core public API."""

from importlib.metadata import PackageNotFoundError, version

from .core.adaptor import (
    GraphTensorInputAdapter,
    LandscapeExport,
    LandscapeInputAdapter,
    LandscapeOutputAdapter,
    ModelAdapter,
    NodeIndexInputAdapter,
    export_landscape_records,
    register_input_adapter,
    register_layer_adapter,
    register_model_adapter,
    register_model_layer_mapping,
    register_output_adapter,
)
from .core.data import (
    LandscapeDataModule,
    LandscapeDataset,
    LandscapeGraphDataset,
    LandscapeGraphRegressionDataModule,
    build_regression_graph_from_landscape,
    make_fitness_target_getter,
    make_preferred_input_getter,
)
from .core.inference import (
    infer_fitness_layer_from_landscape,
    predict_landscape_records,
    predict_sequences,
)
from .core.model_registry import (
    available_data_builders,
    available_models,
    normalize_split_indices,
    register_data,
    register_model,
)
from .core.trainer import TrainingJob, create_trainer
from .landscape_regression import (
    LandscapeRegressionConfig,
    SplitIndices,
    available_landscape_regression_runners,
    discover_demo_csvs,
    register_landscape_regression_runner,
    run_landscape_regression,
    run_landscape_regression_csv,
)

try:
    __version__ = version("landscapy-ml")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "GraphTensorInputAdapter",
    "LandscapeDataModule",
    "LandscapeDataset",
    "LandscapeExport",
    "LandscapeGraphDataset",
    "LandscapeGraphRegressionDataModule",
    "LandscapeInputAdapter",
    "LandscapeOutputAdapter",
    "LandscapeRegressionConfig",
    "ModelAdapter",
    "NodeIndexInputAdapter",
    "SplitIndices",
    "TrainingJob",
    "available_data_builders",
    "available_landscape_regression_runners",
    "available_models",
    "build_regression_graph_from_landscape",
    "create_trainer",
    "discover_demo_csvs",
    "export_landscape_records",
    "infer_fitness_layer_from_landscape",
    "make_fitness_target_getter",
    "make_preferred_input_getter",
    "normalize_split_indices",
    "predict_landscape_records",
    "predict_sequences",
    "register_data",
    "register_input_adapter",
    "register_landscape_regression_runner",
    "register_layer_adapter",
    "register_model",
    "register_model_adapter",
    "register_model_layer_mapping",
    "register_output_adapter",
    "run_landscape_regression",
    "run_landscape_regression_csv",
]
