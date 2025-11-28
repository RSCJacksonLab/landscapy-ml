# landscapy-ml

Sequence classification utilities for carbonic anhydrase families built on top of
`landscapy`, GPyTorch, and PyTorch Lightning.

## Installation

```bash
pip install landscapy-ml
# or from source
pip install .
# development mode
pip install -e ".[dev]"
```

## Quick start

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from landscapyml import (
    SequenceGPClassifier,
    SequenceClassificationDataModule,
    create_trainer,
    TrainingJob,
)

# Example data: embeddings of shape [batch, embedding_dim] and integer labels
embeddings = torch.randn(16, 128)
labels = torch.randint(0, 4, (16,))

dataset = TensorDataset(embeddings, labels)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = SequenceGPClassifier(num_features=128, num_classes=4, num_inducing=32)
trainer = create_trainer(max_epochs=5)

trainer.fit(model, train_dataloaders=loader)

model.eval()
probs, uncertainty = model.predict_with_uncertainty(embeddings)

# Using a DataModule from FitnessLandscape exports
# records = landscape.to_sequence_tensors(
#     as_batch=False,
#     feature_view="embedding",
#     include_embeddings=False,
# )
# dm = SequenceClassificationDataModule(train_data=records, label_key="label", batch_size=8)
# trainer.fit(model, datamodule=dm)

# Embedding raw sequences with landscapy (hard/soft ESM) before training
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
# model = SequenceGPClassifier(num_features=records[0]["embedding"].shape[-1], num_classes=2)
# trainer.fit(model, datamodule=dm)

# End-to-end training job wrapper (for upcoming CLI)
# job = TrainingJob(
#     model_name="sequence_gp_classifier",
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
# python -m landscapyml train --config-path /path/to/conf model=sequence_gp_classifier
#
# Hydra multirun (sweep over num_classes)
# python -m landscapyml train -m model_kwargs.num_classes=2,4 trainer_kwargs.max_epochs=5
```

`SequenceGPClassifier` wraps a variational GPyTorch model with a softmax
likelihood, enabling classification with predictive uncertainty estimates on
unseen amino-acid sequence embeddings.
