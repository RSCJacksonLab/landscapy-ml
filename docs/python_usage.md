# Python Usage

## Graph Attention Regression
```python
from landscapyml import create_trainer
from landscapyml.core.data import LandscapeGraphRegressionDataModule
from landscapyml.examples.gat_fitness import (
    GraphAttentionFitnessRegressor,
    attach_graph_attention_predictions,
)

dm = LandscapeGraphRegressionDataModule.from_landscape(
    landscape=landscape,
    target_layer="measured_fitness",
    normalize_features=True,
)
model = GraphAttentionFitnessRegressor(num_features=dm.train_graph.x.shape[-1])

trainer = create_trainer(max_epochs=50, use_wandb=False)
trainer.fit(model, datamodule=dm)

layer = attach_graph_attention_predictions(
    landscape,
    model,
    layer_name="gat_predicted_fitness",
)
```

## Diffusion-Prior GP
```python
from landscapyml.examples.gp_fitness import (
    attach_diffusion_gp_predictions,
    fit_diffusion_prior_gp,
)

fit = fit_diffusion_prior_gp(
    landscape,
    target_layer="measured_fitness",
    training_iters=100,
    learning_rate=0.05,
)

layer = attach_diffusion_gp_predictions(
    landscape,
    fit.model,
    layer_name="diffusion_gp_predicted_fitness",
)
```

## Registry-Based Training
```python
from landscapyml import TrainingJob

job = TrainingJob(
    model_name="graph_attention_regressor",
    data_name="landscape_graph_regression",
    model_kwargs={"hidden_channels": 64},
    data_kwargs={
        "landscape": landscape,
        "target_layer": "measured_fitness",
        "normalize_features": True,
    },
    trainer_kwargs={"max_epochs": 50, "use_wandb": False},
)
trainer, model, dm = job.run()
```
