# API reference (high level)

This is a lightweight catalog of the public symbols exported from `landscapyml.__init__` and their source modules.

## Landscape pipeline core
- `LandscapeExport`, `export_landscape_records` (`landscape_pipeline.py`): Task-agnostic export of `FitnessLandscape` objects into ML-ready record dictionaries plus layer metadata.
- `LandscapeDataset`, `LandscapeDataModule` (`landscape_pipeline.py`): Generic dataset/DataModule abstractions for landscape-derived pipelines.
- `make_preferred_input_getter`, `make_fitness_target_getter` (`landscape_pipeline.py`): Helpers for wiring record fields into model inputs and targets.

## Data utilities
- `embed_sequences`, `embed_sequences_to_records` (`data.py`): Embed raw sequences and produce record dicts.
- `SequenceClassificationDataset`, `SequenceClassificationDataModule` (`data.py`): Example classification specializations built on the generic landscape pipeline core.
- `CSVConfigRequest`, `build_config_from_dataframe`, `build_config_from_csv`, `write_config` (`data_utils.py`): Build Hydra-ready configs from tabular data or CSVs.

## Models and trainers
- `SequenceMLPClassifier`, `SequenceMLPEnsembleClassifier` (`mlp_classification.py`).
- `TrainingJob`, `register_model`, `register_data`, `create_trainer` (`core/trainer.py`).

## Inference helpers
- `predict_sequences`, `predict_landscape_records` (`core/inference.py`).

## Adapters
- `ModelAdapter`, `LandscapeInputAdapter`, `LandscapeOutputAdapter` (`core/adaptor.py`).
- `GraphTensorInputAdapter`, `NodeIndexInputAdapter` (`core/adaptor.py`): Generic graph-native landscape input adapters reused by the example models.
- `register_model_adapter`, `register_model_layer_mapping`, `register_input_adapter`, `register_output_adapter`, `register_layer_adapter` (`core/adaptor.py`).

## CLI entry points
- `python -m landscapyml` (via `__main__.py`): exposes `list`, `train`, `config-from-csv` commands.
- `hydra_runner.run_with_hydra(overrides=None)`: programmatic Hydra entry.

## Example integrations
- `landscapyml.examples.boltz2_adapter`: Optional example of a model-specific bridge built on `LandscapeInputAdapter` and `ModelAdapter`.
- `landscapyml.examples.gat_fitness`: Example graph-attention regressor, graph DataModule, and graph input adapter for semi-supervised node fitness prediction on `FitnessLandscape` objects.
- `landscapyml.examples.gp_fitness`: Example diffusion-prior exact GP regressor, node-index input adapter, and fitting helpers for transductive graph-based fitness imputation on `FitnessLandscape` objects.

Imports use the package namespace:
```python
from landscapyml import SequenceMLPClassifier, TrainingJob, embed_sequences
```
