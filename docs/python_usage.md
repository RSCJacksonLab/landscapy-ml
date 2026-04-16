# Python usage patterns

This guide stitches together the main APIs exposed in `landscapyml.__init__` for programmatic use.

## Build a generic pipeline from a FitnessLandscape
```python
import torch
from landscapyml import (
    LandscapeDataModule,
    export_landscape_records,
    make_fitness_target_getter,
)

bundle = export_landscape_records(landscape, feature_view="embedding")

dm = LandscapeDataModule(
    train_data=bundle.records,
    dataset_kwargs={
        "target_getter": make_fitness_target_getter(
            "family",
            collapse_one_hot=True,
            dtype=torch.long,
        )
    },
    batch_size=16,
)
```

## Predict unknown fitness values with a graph attention network
This example requires `torch-geometric`.

```python
from landscapyml.examples.gat_fitness import (
    GraphAttentionFitnessRegressor,
    LandscapeGraphRegressionDataModule,
    attach_graph_attention_predictions,
)
from landscapyml import create_trainer

dm = LandscapeGraphRegressionDataModule.from_landscape(
    landscape=landscape,
    target_layer="measured_fitness",
    val_fraction=0.1,
    seed=0,
)

model = GraphAttentionFitnessRegressor(num_features=dm.train_graph.x.shape[-1])
trainer = create_trainer(max_epochs=50, use_wandb=False)
trainer.fit(model, datamodule=dm)

predicted_layer = attach_graph_attention_predictions(
    landscape,
    model,
    layer_name="gat_predicted_fitness",
)
```

## Predict unknown fitness values with a diffusion-prior GP
This example requires `gpytorch`.

Use this when the landscape graph itself is the prior object of interest. The
example fits `t_MAP` on the observed, unmasked part of the landscape, lifts that
diffusion scale to a covariance over all nodes, and then trains an exact GP over
observed node indices.

```python
from landscapyml.examples.gp_fitness import (
    attach_diffusion_gp_predictions,
    fit_diffusion_prior_gp,
)

fit = fit_diffusion_prior_gp(
    landscape,
    target_layer="measured_fitness",
    mask_tokens=("X", "-"),
    training_iters=100,
    learning_rate=0.05,
)

predicted_layer = attach_diffusion_gp_predictions(
    landscape,
    fit.model,
    layer_name="diffusion_gp_predicted_fitness",
)
```

This is a transductive example over a fixed landscape graph. It imputes missing
fitness values for nodes already present in the landscape; it is not meant to
replace an inductive sequence model for out-of-graph generalization.

## Train on existing embeddings
```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from landscapyml import SequenceMLPClassifier, create_trainer

embeddings = torch.randn(32, 128)
labels = torch.randint(0, 3, (32,))
loader = DataLoader(TensorDataset(embeddings, labels), batch_size=8, shuffle=True)

model = SequenceMLPClassifier(num_features=128, num_classes=3)
trainer = create_trainer(max_epochs=5)
trainer.fit(model, train_dataloaders=loader)
```

## Train from raw sequences (auto-embed)
```python
from landscapyml import SequenceClassificationDataModule, SequenceMLPClassifier, create_trainer

sequences = ["ACDE", "MNPK", "WXYZ"]
labels = [0, 1, 2]

dm = SequenceClassificationDataModule.from_sequences(
    train_sequences=sequences,
    train_labels=labels,
    label_key="family",
    embedding_mode="hard",
    model_name="facebook/esm2_t6_8M_UR50D",
    batch_size=4,
)

dm.setup("fit")
num_features = dm.train_dataloader().dataset[0][0].shape[-1]  # inferred automatically by TrainingJob
model = SequenceMLPClassifier(num_features=num_features, num_classes=3)
trainer = create_trainer(max_epochs=5)
trainer.fit(model, datamodule=dm)
```

## TrainingJob convenience
Wraps data/model/trainer wiring and handles seeding and metadata logging. The
MLP model and classification datamodule shown here are examples layered on the
generic landscape pipeline core.
```python
from landscapyml import TrainingJob

job = TrainingJob(
    model_name="sequence_mlp_ensemble",
    data_name="fitness_landscape_records",
    model_kwargs={"num_classes": 3, "num_models": 4},
    data_kwargs={"train_data": records, "label_key": "label", "batch_size": 16},
    trainer_kwargs={"max_epochs": 20, "use_wandb": False},
    seed=123,
)
trainer, model, dm = job.run(fit=True, test=False)
```

## External LightningModule
Load a LightningModule by class path without registering custom code:
```python
from landscapyml import TrainingJob

job = TrainingJob(
    model_name="external",
    data_name="fitness_landscape_records",
    model_kwargs={
        "class_path": "mypkg.models.MyLightningModule",
        "init_kwargs": {"num_classes": 2},
    },
    data_kwargs={"train_data": records, "label_key": "label"},
    trainer_kwargs={"max_epochs": 5},
)
trainer, model, dm = job.run()
```

## Build configs programmatically
Use the helpers in `data_utils` when you need a Hydra config file for CLI runs or reproducibility.
```python
import pandas as pd
from landscapyml import build_config_from_dataframe, write_config

df = pd.read_csv("data.csv")
config = build_config_from_dataframe(
    df,
    sequence_column="sequence",
    label_column="family",
    model="sequence_mlp_classifier",
    embedding_mode="hard",
    max_epochs=10,
)
write_config(config, Path("landscapyml_run_conf/config.yaml"))
```

## Inference with uncertainty
After training, reuse an ensemble model for predictions with uncertainty.
```python
from landscapyml import SequenceMLPEnsembleClassifier, predict_sequences

model = SequenceMLPEnsembleClassifier(num_features=128, num_classes=3, num_models=3)

probs, var = predict_sequences(
    model,
    sequences=["ACDE", "MNPK"],
    embedding_mode="hard",
    model_name="facebook/esm2_t6_8M_UR50D",
)
```

## Logging
`logging_utils.configure_logger(log_file=None, log_level="INFO")` sets up a package-level logger that writes to stdout by default or to a specified file path, creating parent directories if needed.
