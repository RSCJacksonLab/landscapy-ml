# landscapy-ml Overview

landscapy-ml provides sequence classification utilities built on landscapy and PyTorch Lightning. The package centers around small, composable helpers for embedding sequences, constructing datasets, training classifiers, and running inference with uncertainty estimates.

## Package layout
- `landscapyml.config.JobConfig`: Hydra-validated configuration schema for training jobs.
- `landscapyml.data`: Embedding helpers, dataset and Lightning `DataModule` definitions.
- `landscapyml.data_utils`: Convenience builders for creating configs from data frames or CSVs.
- `landscapyml.mlp_classification`: MLP classifiers (single model and deep ensemble) for embeddings.
- `landscapyml.core.trainer`: Trainer factory (`create_trainer`) and registries for models/data builders (`TrainingJob`).
- `landscapyml.hydra_runner`: Programmatic Hydra entry point that wires configs to `TrainingJob`.
- `landscapyml.core.inference`: Inference helpers for raw sequences, FitnessLandscape exports, and landscape layers.
- `landscapyml.core.adaptor`: Adapter ABCs and registries for mapping landscapes to models and predictions back to layers.
- `landscapyml.logging_utils`: Lightweight logging configuration aligned with landscapy.
- `landscapyml.__main__`: CLI entry point (`python -m landscapyml`).

Use the CLI for Hydra-driven training, or import the Python APIs for programmatic training/inference. Other docs in this folder dive into specific topics.
