# landscapy-ml

[![CI](https://github.com/RSCJacksonLab/landscapy-ml/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/RSCJacksonLab/landscapy-ml/actions/workflows/ci.yml)

`landscapy-ml` is the small bridge between
[`landscapy`](https://github.com/RSCJacksonLab/landscapy) `FitnessLandscape`
objects and PyTorch models.

More information, documentation and usage instructions can be found on
[`landscapy`](https://github.com/RSCJacksonLab/landscapy).

## Installation

`landscapy-ml` is installed by default with `pip install landscapy` or
`pip install landscapy-ml`. This remains the recommended installation path.

To install directly from a checkout:

```bash
python -m pip install .
```

## Python

Use the Python interface to convert a `FitnessLandscape` into PyTorch-ready
records and datasets, then adapt model outputs back into landscape layers as
described in the [Python usage guide](docs/python_usage.md).

```python
from importlib.resources import files

from fitness_landscape.core.landscape import read_csv_landscape

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

features, target = dataset[0]
print(len(dataset), features.shape, target.shape)
```

## CLI

Use the command-line interface to inspect registered models and data builders
and run the maintained CSV workflows described in the [CLI guide](docs/cli.md).
