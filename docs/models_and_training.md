# Models And Training

## Maintained Example Models
- `GraphAttentionFitnessRegressor` (`examples/gat_fitness.py`): Lightning graph attention regressor for semi-supervised node fitness prediction. Requires `torch-geometric`.
- `DiffusionPriorExactGP` (`examples/gp_fitness.py`): exact GP regressor over a fixed landscape graph using a diffusion covariance. Requires `gpytorch`.

## Registry
`core/model_registry.py` maps names to factories.

Built-in entries:
- Models: `external`
- Data builders: `landscape_records`, `landscape_graph_regression`

Importing `landscapyml.examples.gat_fitness` registers:
- `graph_attention_regressor`

The diffusion-prior GP is registered as a landscape runner, because it is not a
Lightning training job.

## TrainingJob
`TrainingJob` wires a registered model, registered data builder, and Lightning
trainer:

```python
from landscapyml import TrainingJob
from landscapyml.examples import gat_fitness  # register example model

job = TrainingJob(
    model_name="graph_attention_regressor",
    data_name="landscape_graph_regression",
    data_kwargs={"landscape": landscape, "target_layer": "fitness"},
    trainer_kwargs={"max_epochs": 50, "use_wandb": False},
)
trainer, model, dm = job.run()
```

Use `register_model(...)` and `register_data(...)` for project-specific
Lightning models and data modules.
