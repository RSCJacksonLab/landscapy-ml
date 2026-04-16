# landscapy-ml

`landscapy-ml` is a thin bridge between `landscapy` landscape objects and ML
training or inference pipelines.

The package now has a deliberate split:
- Core pipeline utilities export `FitnessLandscape` objects into ML-ready records,
  datasets, and Lightning datamodules.
- Example implementations, such as the sequence MLP classifiers and the Boltz-2
  adapter, sit on top of that generic interface instead of defining the package
  architecture.

## Installation

```bash
pip install landscapy-ml
# or from source
pip install .
# development mode
pip install -e ".[dev]"
# optional graph / GP examples
pip install -e ".[graph]"
pip install -e ".[gp]"
# enable formatting hooks
pre-commit install
```

## Quick start

```python
import torch
from landscapyml import (
    LandscapeDataModule,
    SequenceMLPEnsembleClassifier,  # example model built on the generic core
    TrainingJob,
    create_trainer,                 # example Lightning trainer factory
    export_landscape_records,
    make_fitness_target_getter,
)

# Export a landscape into generic ML records
# bundle = export_landscape_records(landscape, feature_view="embedding")
# dm = LandscapeDataModule(
#     train_data=bundle.records,
#     dataset_kwargs={
#         "target_getter": make_fitness_target_getter(
#             "label",
#             collapse_one_hot=True,
#             dtype=torch.long,
#         )
#     },
#     batch_size=8,
# )
# model = SequenceMLPEnsembleClassifier(num_features=128, num_classes=4, num_models=3)
# trainer = create_trainer(max_epochs=5)
# trainer.fit(model, datamodule=dm)

# Example-only convenience path: embedding raw sequences with landscapy
# from landscapyml import embed_sequences_to_records
# sequences = ["ACDE", "WXYZ"]
# labels = [0, 1]
# records = embed_sequences_to_records(
#     sequences,
#     labels,
#     label_key="label",
#     embedding_mode="hard",  # or "soft"
#     model_name="facebook/esm2_t6_8M_UR50D",
# )
# dm = SequenceClassificationDataModule(train_data=records, label_key="label", batch_size=4)
# model = SequenceMLPEnsembleClassifier(num_features=records[0]["embedding"].shape[-1], num_classes=2)
# trainer.fit(model, datamodule=dm)

# Example-only end-to-end training job wrapper
# job = TrainingJob(
#     model_name="sequence_mlp_ensemble",
#     data_name="raw_sequences",
#     model_kwargs={"num_classes": 2},
#     data_kwargs={
#         "train_sequences": sequences,
#         "train_labels": labels,
#         "label_key": "label",
#         "embedding_mode": "hard",
#     },
#     trainer_kwargs={"max_epochs": 5},
# )
# trainer, model, dm = job.run()

# CLI (python -m landscapyml)
# List available registry entries
# python -m landscapyml list
#
# Hydra-driven training (uses conf/config.yaml by default)
# python -m landscapyml train model_kwargs.num_classes=2 \
#   data_kwargs.train_sequences='[\"ACDE\",\"WXYZ\"]' \
#   data_kwargs.train_labels='[0,1]' \
#   data_kwargs.label_key=label \
#   trainer_kwargs.max_epochs=5
#
# TensorBoard logs and checkpoints are written inside the Hydra run directory
# (outputs/YYYY-MM-DD/HH-MM-SS by default), under logs/ and checkpoints/.
#
# Optional Weights & Biases tracking (install with `pip install .[tracking]`):
# python -m landscapyml train trainer_kwargs.use_wandb=true \
#   trainer_kwargs.wandb_project=my_project trainer_kwargs.wandb_run_name=test_run
#
# Override config directory (if you keep custom configs elsewhere)
# python -m landscapyml train --config-path /path/to/conf model=sequence_mlp_classifier
#
# Hydra multirun (sweep over num_classes)
# python -m landscapyml train -m model_kwargs.num_classes=2,4 trainer_kwargs.max_epochs=5
```

`SequenceMLPEnsembleClassifier` is kept as a reference implementation for
embedding-based sequence classification. For model-specific adapters, see
`landscapyml.examples`, including the optional `boltz2_adapter` example and the
graph-based `gat_fitness` and `gp_fitness` examples for predicting unknown
numeric fitnesses on landscape graphs. The GAT example is inductive over node
features on the fixed graph, while the GP example is a diffusion-prior,
transductive imputation model over the existing landscape nodes.
