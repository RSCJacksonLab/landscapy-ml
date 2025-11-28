# Python usage patterns

This guide stitches together the main APIs exposed in `landscapyml.__init__` for programmatic use.

## Train on existing embeddings
```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from landscapyml import SequenceGPClassifier, create_trainer

embeddings = torch.randn(32, 128)
labels = torch.randint(0, 3, (32,))
loader = DataLoader(TensorDataset(embeddings, labels), batch_size=8, shuffle=True)

model = SequenceGPClassifier(num_features=128, num_classes=3, num_inducing=32)
trainer = create_trainer(max_epochs=5)
trainer.fit(model, train_dataloaders=loader)
```

## Train from raw sequences (auto-embed)
```python
from landscapyml import SequenceClassificationDataModule, SequenceGPClassifier, create_trainer

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
model = SequenceGPClassifier(num_features=num_features, num_classes=3)
trainer = create_trainer(max_epochs=5)
trainer.fit(model, datamodule=dm)
```

## TrainingJob convenience
Wraps data/model/trainer wiring and handles seeding and metadata logging.
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
    model="sequence_gp_classifier",
    embedding_mode="hard",
    max_epochs=10,
)
write_config(config, Path("landscapyml_run_conf/config.yaml"))
```

## Inference with uncertainty
After training, reuse the same model for predictions.
```python
from landscapyml import predict_sequences

probs, var = predict_sequences(
    model,
    sequences=["ACDE", "MNPK"],
    embedding_mode="hard",
    model_name="facebook/esm2_t6_8M_UR50D",
)
```

## Logging
`logging_utils.configure_logger(log_file=None, log_level="INFO")` sets up a package-level logger that writes to stdout by default or to a specified file path, creating parent directories if needed.
