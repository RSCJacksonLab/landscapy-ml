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

`LandscapeDataModule(val_split=...)` accepts fractions from zero (inclusive)
to one (exclusive). A positive fraction must leave at least one training and
one validation record. Infeasible fractions are rejected during initialization
before the record lists are split. Repeated `setup("fit")` calls reuse the same
partition.

## Graph Regression
`LandscapeGraphRegressionDataModule.from_landscape(...)` converts a numeric
fitness layer into node-level regression targets on a single landscape graph.
Known finite targets define train, validation, and test masks. Missing targets
are available for prediction.

Variable-length sequences are supported through fallback node features:
embeddings when present, otherwise normalized sequence length plus token
composition.
