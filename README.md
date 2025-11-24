# CA-classifications

Sequence classification utilities for carbonic anhydrase families built on top of
`landscapy`, GPyTorch, and PyTorch Lightning.

## Installation

```bash
pip install .
# or for development
pip install -e ".[dev]"
```

## Quick start

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from ca_classifications import (
    SequenceGPClassifier,
    SequenceClassificationDataModule,
    create_trainer,
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
# from ca_classifications import embed_sequences_to_records
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
```

`SequenceGPClassifier` wraps a variational GPyTorch model with a softmax
likelihood, enabling classification with predictive uncertainty estimates on
unseen amino-acid sequence embeddings.
