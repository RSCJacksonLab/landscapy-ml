# landscapy-ml

`landscapy-ml` is the small bridge between `landscapy` `FitnessLandscape`
objects and PyTorch models. The package focuses on one path:

```text
sequences + fitnesses + splits
-> landscapy FitnessLandscape
-> PyTorch training
-> predicted fitness layer attached back to the landscape
```

The source is intentionally narrow:
- `landscapyml.core.data`: generic landscape record datasets and data modules.
- `landscapyml.core.adaptor`: landscape input/model/output adapters.
- `landscapyml.core.model_registry`: functional model and data registration.
- `landscapyml.core.trainer`: Lightning trainer construction and `TrainingJob`.
- `landscapyml.core.inference`: trained model output back to landscapy fitness layers.
- `landscapyml.examples`: maintained graph-regression data, GAT, and diffusion-prior GP demos.

## Install

```bash
pip install -e ".[dev]"
pip install -e ".[graph]"  # graph attention demo
pip install -e ".[gp]"     # diffusion GP demo
pip install -e ".[tuning]" # Ray Tune demo support
```

## CLI

```bash
python -m landscapyml list

MAX_EPOCHS=1 bash demo/run_gat.sh \
  --csv-path demo/rhomax/by_wild_type/by_wild_type.csv.gz

RAY_TUNE=1 NUM_SAMPLES=4 MAX_EPOCHS=1 bash demo/run_gat.sh \
  --csv-path demo/rhomax/by_wild_type/by_wild_type.csv.gz

FIT_KWARGS='{"training_iters": 2, "learning_rate": 0.05}' \
  bash demo/run_gp.sh \
  --csv-path demo/rhomax/by_wild_type/by_wild_type.csv.gz
```

## Python

```python
from landscapyml.examples.data import LandscapeGraphRegressionDataModule
from landscapyml.examples.gat_fitness import (
    GraphAttentionFitnessRegressor,
    attach_graph_attention_predictions,
)
from landscapyml.examples.logging import create_examples_trainer
from landscapyml.examples.pipeline import prepare_landscape_regression_data_from_dataframe

prepared = prepare_landscape_regression_data_from_dataframe(df)

dm = LandscapeGraphRegressionDataModule.from_landscape(
    landscape=prepared.landscape,
    target_layer="target",
    normalize_features=True,
)
model = GraphAttentionFitnessRegressor(num_features=dm.train_graph.x.shape[-1])

trainer = create_examples_trainer(max_epochs=50)
trainer.fit(model, datamodule=dm)

predicted = attach_graph_attention_predictions(
    prepared.landscape,
    model,
    layer_name="gat_predicted_fitness",
)
```
