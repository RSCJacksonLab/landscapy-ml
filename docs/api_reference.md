# API reference (high level)

This is a lightweight catalog of the public symbols exported from `landscapyml.__init__` and their source modules.

## Data utilities
- `embed_sequences`, `embed_sequences_to_records` (`data.py`): Embed raw sequences and produce record dicts.
- `SequenceClassificationDataset`, `SequenceClassificationDataModule` (`data.py`): Dataset/DataModule abstractions for classification.
- `CSVConfigRequest`, `build_config_from_dataframe`, `build_config_from_csv`, `write_config` (`data_utils.py`): Build Hydra-ready configs from tabular data or CSVs.

## Models and trainers
- `SequenceMLPClassifier`, `SequenceMLPEnsembleClassifier` (`mlp_classification.py`).
- `TrainingJob`, `register_model`, `register_data`, `create_trainer` (`core/trainer.py`).

## Inference helpers
- `predict_sequences`, `predict_landscape_records` (`core/inference.py`).

## Adapters
- `ModelAdapter`, `LandscapeInputAdapter`, `LandscapeOutputAdapter` (`core/adaptor.py`).
- `register_model_adapter`, `register_model_layer_mapping`, `register_input_adapter`, `register_output_adapter`, `register_layer_adapter` (`core/adaptor.py`).

## CLI entry points
- `python -m landscapyml` (via `__main__.py`): exposes `list`, `train`, `config-from-csv` commands.
- `hydra_runner.run_with_hydra(overrides=None)`: programmatic Hydra entry.

Imports use the package namespace:
```python
from landscapyml import SequenceMLPClassifier, TrainingJob, embed_sequences
```
