# landscapy-ml Overview

landscapy-ml provides a generic interface layer between `landscapy`
`FitnessLandscape` objects and ML pipelines. The package centers around
small, composable helpers for exporting landscapes into records, wiring those
records into datasets and Lightning datamodules, and adapting model outputs
back into landscape fitness layers.

Sequence classifiers and model-specific integrations are examples built on top
of that core rather than the primary abstraction.

## Package layout
- `landscapyml.landscape_pipeline`: Generic record export, dataset, and DataModule utilities for landscape-driven pipelines.
- `landscapyml.config.JobConfig`: Hydra-validated configuration schema for training jobs.
- `landscapyml.data`: Embedding helpers, dataset and Lightning `DataModule` definitions.
- `landscapyml.data_utils`: Convenience builders for creating configs from data frames or CSVs.
- `landscapyml.mlp_classification`: Example MLP classifiers (single model and deep ensemble) for embeddings.
- `landscapyml.core.trainer`: Trainer factory (`create_trainer`) and registries for models/data builders (`TrainingJob`).
- `landscapyml.hydra_runner`: Programmatic Hydra entry point that wires configs to `TrainingJob`.
- `landscapyml.core.inference`: Inference helpers for raw sequences, FitnessLandscape exports, and landscape layers.
- `landscapyml.core.adaptor`: Adapter ABCs and registries for mapping landscapes to models and predictions back to layers.
- `landscapyml.examples`: Optional model-specific examples such as the Boltz-2 adapter, the GAT-based fitness regressor, and the diffusion-prior GP regressor.
- `landscapyml.logging_utils`: Lightweight logging configuration aligned with landscapy.
- `landscapyml.__main__`: CLI entry point (`python -m landscapyml`).

Use the CLI for Hydra-driven training, or import the Python APIs for programmatic training/inference. Other docs in this folder dive into specific topics.
