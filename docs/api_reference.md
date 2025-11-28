# API reference (high level)

This is a lightweight catalog of the public symbols exported from `landscapyml.__init__` and their source modules.

## Data utilities
- `embed_sequences`, `embed_sequences_to_records` (`data.py`): Embed raw sequences and produce record dicts.
- `SequenceClassificationDataset`, `SequenceClassificationDataModule` (`data.py`): Dataset/DataModule abstractions for classification.
- `CSVConfigRequest`, `build_config_from_dataframe`, `build_config_from_csv`, `write_config` (`data_utils.py`): Build Hydra-ready configs from tabular data or CSVs.

## Models and trainers
- `SequenceGPModel`, `SequenceGPClassifier`, `create_trainer` (`gp_classification.py`).
- `SequenceMLPClassifier`, `SequenceMLPEnsembleClassifier` (`mlp_classification.py`).
- `TrainingJob`, `register_model`, `register_data` (`trainer.py`).

## Inference helpers
- `predict_sequences`, `predict_landscape_records` (`inference.py`).

## CLI entry points
- `python -m landscapyml` (via `__main__.py`): exposes `list`, `train`, `config-from-csv` commands.
- `hydra_runner.run_with_hydra(overrides=None)`: programmatic Hydra entry.

Imports use the package namespace:
```python
from landscapyml import SequenceGPClassifier, TrainingJob, embed_sequences
```
