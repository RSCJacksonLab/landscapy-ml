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
```

## CLI

```bash
python -m landscapyml list

MAX_EPOCHS=1 bash demo/run_gat.sh \
  --csv-path demo/rhomax/by_wild_type/by_wild_type.csv.gz

FIT_KWARGS='{"training_iters": 2, "learning_rate": 0.05}' \
  bash demo/run_gp.sh \
  --csv-path demo/rhomax/by_wild_type/by_wild_type.csv.gz
```

## Python

```python
from fitness_landscape.models import create_nk_binary_landscape

from landscapyml.core.data import LandscapeGraphRegressionDataModule
from landscapyml.core.trainer import create_trainer
from landscapyml.examples.gat_fitness import (
    GraphAttentionFitnessRegressor,
    attach_graph_attention_predictions,
)

landscape = create_nk_binary_landscape(N=3, K=1, seed=42)

dm = LandscapeGraphRegressionDataModule.from_landscape(
    landscape=landscape,
    target_layer="nk_k=1",
    val_fraction=0.25,
    seed=42,
    normalize_features=True,
)
model = GraphAttentionFitnessRegressor(num_features=dm.train_graph.x.shape[-1])

trainer = create_trainer(max_epochs=50, use_wandb=False)
trainer.fit(model, datamodule=dm)

predicted = attach_graph_attention_predictions(
    landscape,
    model,
    layer_name="gat_predicted_fitness",
)
```

## Testing

```bash
python -m pytest
```

The test command measures branch coverage for the complete `landscapyml`
package, writes `coverage.xml`, and enforces the initial release baseline of
75%. New exclusions should not be added in place of testing core behavior.
