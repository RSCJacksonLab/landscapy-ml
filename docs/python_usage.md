# Python usage

The Python interface converts Landscapy objects into PyTorch-ready data and
adapts model predictions back into landscape fitness layers.

## Records and datasets

The package includes a small aligned landscape at
`landscapyml/data/minimal_landscape.csv`. The following example constructs a
Hamming landscape, exports its numeric target, and wraps the records as a
PyTorch dataset.

```python
from importlib.resources import files

from fitness_landscape.core.landscape import read_csv_landscape
from torch.utils.data import DataLoader

from landscapyml import (
    LandscapeDataset,
    export_landscape_records,
    make_fitness_target_getter,
)

data_path = files("landscapyml").joinpath("data/minimal_landscape.csv")
landscape = read_csv_landscape(
    data_path,
    sequence_col="sequence",
    alphabet=list("AC"),
    graph="hamming",
    numeric_layers=["target"],
    attach_embeddings=False,
)
exported = export_landscape_records(
    landscape,
    fitness_layers=["target"],
    include_embeddings=False,
)
dataset = LandscapeDataset(
    exported.records,
    target_getter=make_fitness_target_getter("target"),
)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

features, targets = next(iter(loader))
print(features.shape, targets.shape)
```

`export_landscape_records` preserves landscape sequence order and selected
fitness layers. `LandscapeDataset` accepts custom input and target getters when
a model requires a different record view.

## Training and inference

`LandscapeDataModule` provides standard PyTorch Lightning loaders for exported
records. `LandscapeGraphRegressionDataModule` provides the graph-native path.
`TrainingJob` resolves registered models and data builders without adding a
second landscape data model.

The inference adapters attach numeric or categorical predictions to a supplied
landscape. See [Inference and landscape integration](inference.md) for adapter
registration, device handling, copied inference, and layer attachment.
