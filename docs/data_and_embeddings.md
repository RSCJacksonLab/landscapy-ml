# Data

The core data path starts from a `FitnessLandscape`, not from a second package
data format.

## Records
`export_landscape_records(...)` preserves landscapy tensor exports as record
dictionaries with:
- `sequence_tensor`
- `embedding` when available
- `fitness_tensors`
- `attention_mask` when available

`LandscapeDataset` and `LandscapeDataModule` are the generic dataset and
Lightning data module wrappers for those records.

## Graph Regression
`LandscapeGraphRegressionDataModule.from_landscape(...)` converts a numeric
fitness layer into node-level regression targets on a single landscape graph.
Known finite targets define train/validation/test masks; missing targets are
available for prediction.

Variable-length sequences are supported through fallback node features:
embeddings when present, otherwise normalized sequence length plus token
composition.
