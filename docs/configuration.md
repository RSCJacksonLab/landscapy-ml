# Configuration and Hydra usage

Hydra drives training and testing. The structured config schema is defined in `config.JobConfig` and registered with Hydra in `hydra_runner.py`.

## Default config
The baseline config ships inside the package at `landscapyml/conf/config.yaml` (also available at the repo root `conf/config.yaml`). Key fields:
- `model`: Registry key for the model factory (see `trainer.py`). Default `sequence_gp_classifier`.
- `data`: Registry key for the data builder. Default `raw_sequences` (embeds raw sequences before training).
- `model_kwargs`: Passed to the selected model factory (e.g., `num_classes`, `num_inducing`).
- `data_kwargs`: Passed to the data builder (e.g., sequence lists, labels, label mapping, embedding options).
- `trainer_kwargs`: Passed to `create_trainer` (or the provided `trainer_factory`), including logging/checkpoint paths and W&B settings.
- `log_file` / `log_level`: Configure the package logger via `logging_utils.configure_logger`.
- `seed`: Optional global seed via `pytorch_lightning.seed_everything`.
- `fit` / `test`: Whether to run `.fit()` and `.test()` on the built trainer/model/datamodule.

Hydra settings under `hydra.run.dir` control the output directory naming (`outputs/YYYY-MM-DD/HH-MM-SS` by default).

## Locating configs
`hydra_runner._config_path` resolves the config directory in order:
1. Environment variable `LANDSCAPYML_CONFIG_PATH` (may point at a directory or directly at `config.yaml`).
2. Packaged config at `landscapyml/conf`.
3. Repository `conf/` when running from source.
4. `conf/` under the current working directory.

The CLI's `--config-path` flag sets `LANDSCAPYML_CONFIG_PATH` before invoking Hydra.

## Path normalization
`hydra_runner` normalizes `trainer_kwargs.log_dir` and `trainer_kwargs.checkpoint_dir` relative to the config directory so artifact paths remain predictable even when running from other working directories.

## Overriding values
Pass standard Hydra overrides at the CLI or programmatically through `run_with_hydra(overrides=...)`. Examples:
- Change the model and class count: `model=sequence_mlp_classifier model_kwargs.num_classes=3`
- Adjust validation split: `data_kwargs.val_split=0.2 data_kwargs.val_seed=42`
- Disable W&B: `trainer_kwargs.use_wandb=false`

## External models
To load an external LightningModule by class path, set `model=external` and provide a `class_path` plus optional constructor args:
- `model=external model_kwargs.class_path=mypkg.models.MyModule model_kwargs.num_classes=3`
- Or pass nested kwargs: `model_kwargs.class_path=mypkg.models.MyModule model_kwargs.init_kwargs.num_classes=3`
- If the external class is not a LightningModule, provide an adapter wrapper via `model_kwargs.adapter_path`.
