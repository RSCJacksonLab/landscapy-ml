# Data handling and embeddings

Data utilities live in `landscapyml.data` and `landscapyml.data_utils`.

## Embedding raw sequences
- `embed_sequences(sequences, *, embedding_mode="hard"|"soft", model_name="facebook/esm2_t6_8M_UR50D", device=None, embedding_batch_size=32, include_tokens=True) -> (embeddings, tokens, attention_masks)`
  - Uses landscapy ESM embedders (hard/soft) to convert raw sequences to fixed-size embeddings.
  - When `include_tokens` is true with `hard` mode, returns padded token tensors and attention masks; for `soft`, tokens are disabled.
  - Requires `landscapy` optional dependency; raises `ImportError` if unavailable.
- `embed_sequences_to_records(sequences, labels, *, label_key, ...) -> list[dict]`
  - Embeds sequences and produces records that mirror `FitnessLandscape.to_sequence_tensors(as_batch=False)` output. Each record contains `sequence_tensor` or `embedding`, `fitness_tensors`, and optional `attention_mask`.

## Normalizing data shapes
- `_normalize_records` and `_expand_batch_dict` accept either batched dictionaries or iterables of record dicts and return a list of per-sequence records. Error handling ensures required keys are present.

## Datasets and DataModule
- `SequenceClassificationDataset(records, label_key)` produces `(features, label)` pairs. Features prefer `record['embedding']` falling back to `record['sequence_tensor']`. Labels are drawn from `record['fitness_tensors'][label_key]`; multi-hot labels are argmaxed.
- `SequenceClassificationDataModule` wraps the dataset with train/val/test/predict splits and `DataLoader` construction. Key args:
  - `train_data`, `val_data`, `test_data`, `predict_data`: batch dicts or iterables of record dicts.
  - `label_key`: name of the label inside `fitness_tensors`.
  - `label_mapping`: optional category names; persisted next to checkpoints by `TrainingJob`.
  - `batch_size`, `num_workers`, `pin_memory`, `shuffle`, `val_split`, `val_seed`.
  - Supports splitting `train_records` into train/val if `val_split>0` and no val data provided.
- `SequenceClassificationDataModule.from_sequences(...)` embeds raw sequences and labels before constructing the DataModule. It wires through `embed_sequences_to_records` options (embedding mode/model, batch size, tokens, label key/mapping, and val/test/predict splits).

## Config helpers
`landscapyml.data_utils` provides convenience for generating Hydra configs:
- `CSVConfigRequest`: dataclass bundling CSV paths and hyperparameters.
- `build_config_from_dataframe(df, *, sequence_column, label_column, ...)`: Converts an in-memory dataframe with sequences/labels into a config dict (encodes categorical labels, builds label mapping, sets trainer/data kwargs).
- `build_config_from_csv(req)`: Loads a CSV via pandas and delegates to `build_config_from_dataframe`.
- `write_config(config, path)`: Writes a mapping to JSON for Hydra consumption.

Dependencies: pandas is optional; importing `build_config_from_csv` without pandas installed raises `ImportError` with installation guidance.
