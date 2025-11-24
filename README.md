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
from ca_classifications import SequenceGPClassifier, create_trainer

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
```

`SequenceGPClassifier` wraps a variational GPyTorch model with a softmax
likelihood, enabling classification with predictive uncertainty estimates on
unseen amino-acid sequence embeddings.
