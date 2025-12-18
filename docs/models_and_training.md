# Models, trainers, and jobs

## Model implementations
- `SequenceGPClassifier` (`gp_classification.py`): Variational Gaussian process classifier with a softmax likelihood (`gpytorch.likelihoods.SoftmaxLikelihood`).
  - Key args: `num_features`, `num_classes`, `inducing_points` or `num_inducing`, `learning_rate`, `weight_decay`, `num_data` (optional dataset size for ELBO), `embedding_domain`, `embedding_model` (stored for compatibility checks during inference).
  - Training/validation steps log loss and accuracies (overall and per-class). `predict_with_uncertainty` returns `(mean_probs, variance)` over classes.
- `SequenceMLPClassifier` (`mlp_classification.py`): Single MLP classifier for embeddings.
  - Key args: `num_features`, `num_classes`, `hidden_sizes`, `dropout`, optimizer hyperparameters.
- `SequenceMLPEnsembleClassifier` (`mlp_classification.py`): Deep ensemble of MLP classifiers for uncertainty estimation.
  - Key args: same as single MLP plus `num_models`; `predict_with_uncertainty` aggregates mean/variance across ensemble members.

## Trainer factory
- `create_trainer(...)` in `gp_classification.py` builds a `pytorch_lightning.Trainer` with:
  - TensorBoard logging to `log_dir/experiment_name`.
  - Optional W&B logging (when `use_wandb` true and `wandb` available) configured via `wandb_project`, `wandb_entity`, `wandb_run_name`, `wandb_tags`, `wandb_dir`.
  - Checkpointing callback controlled by `checkpoint_dir`, `checkpoint_monitor`, `checkpoint_mode`, `checkpoint_every_n_epochs`, `save_top_k`.
  - Other knobs: `max_epochs`, `accelerator`, `devices`, `log_every_n_steps`, `num_sanity_val_steps`.

## Registries
`trainer.py` maintains registries that map string keys to factories:
- Models: `sequence_gp_classifier`, `sequence_mlp_classifier`, `sequence_mlp_ensemble`, `external`.
- Data builders: `fitness_landscape_records`, `raw_sequences` (`SequenceClassificationDataModule.from_sequences`), `fitness_landscape` (alias for direct record usage).
Use `register_model(name, factory, overwrite=False, requires_num_features=True)` or `register_data(name, factory, overwrite=False)` to extend the set of available components. Set `requires_num_features=False` for models that should not auto-infer feature dimensions. Factories must accept the kwargs passed via Hydra or `TrainingJob`. Example:
```python
from landscapyml import register_model, register_data
from landscapyml.trainer import TrainingJob

# Register a new model factory
def build_custom_model(num_features: int, num_classes: int, **kwargs):
    return MyLightningModule(num_features=num_features, num_classes=num_classes, **kwargs)

register_model("custom_model", build_custom_model)

# Register a custom data builder
def build_custom_data(train_data, label_key, **kwargs):
    return MyDataModule(train_data=train_data, label_key=label_key, **kwargs)

register_data("custom_data", build_custom_data)

job = TrainingJob(
    model_name="custom_model",
    data_name="custom_data",
    model_kwargs={"num_classes": 3},
    data_kwargs={"train_data": records, "label_key": "label"},
    trainer_kwargs={"max_epochs": 10},
)
trainer, model, dm = job.run()
```

## External models
The `external` model entry instantiates a LightningModule by class path and optional adapter:
```python
from landscapyml import TrainingJob

job = TrainingJob(
    model_name="external",
    data_name="fitness_landscape_records",
    model_kwargs={
        "class_path": "mypkg.models.MyLightningModule",
        "init_kwargs": {"num_classes": 3},
    },
    data_kwargs={"train_data": records, "label_key": "label"},
    trainer_kwargs={"max_epochs": 10},
)
trainer, model, dm = job.run()
```
If the external class is not a LightningModule, supply an adapter via `model_kwargs.adapter_path` that wraps the model into a LightningModule.

## TrainingJob workflow
`TrainingJob` (dataclass in `trainer.py`) coordinates building the data module, model, and trainer from provided kwargs and registry keys.
- Performs validation that chosen `model_name` and `data_name` exist in the registries.
- Builds the DataModule and calls `.setup()` before inferring `num_features` from the train loader if not supplied.
- Instantiates the model with `model_kwargs`, trainer with `trainer_factory` (defaults to `create_trainer`), seeds via `pytorch_lightning.seed_everything` when `seed` is set.
- Logs run metadata (label mapping, counts, registry keys) to all configured loggers.
- Writes `label_mapping.json` next to checkpoints when available.
- `run(fit=True, test=False)` will execute `.fit()` and optionally `.test()` on the assembled components.

Typical programmatic training flow:
```python
from landscapyml import TrainingJob

job = TrainingJob(
    model_name="sequence_gp_classifier",
    data_name="raw_sequences",
    model_kwargs={"num_classes": 3},
    data_kwargs={
        "train_sequences": sequences,
        "train_labels": labels,
        "label_key": "family",
        "embedding_mode": "hard",
    },
    trainer_kwargs={"max_epochs": 5},
    seed=0,
)
trainer, model, dm = job.run()
```
