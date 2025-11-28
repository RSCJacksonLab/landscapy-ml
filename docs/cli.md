# Command-line interface

The CLI is exposed via `python -m landscapyml` (see `landscapyml.__main__`). It uses Click plus Hydra for configuration overrides.

## Commands
- `list`: Prints registered model keys and data builder keys from the registries in `trainer.py`.
- `train [--config-path PATH] [HYDRA_OVERRIDES...]`: Runs a Hydra-configured training job. Any remaining args are forwarded to Hydra as overrides (e.g., `trainer_kwargs.max_epochs=5`). `--config-path` can be a directory containing `config.yaml` or a path to a specific `config.yaml`.
- `config-from-csv --csv-path FILE --sequence-column COL --label-column COL [options]`: Reads a CSV, builds a Hydra config via `data_utils.build_config_from_csv`, and writes `config.yaml` under `out_dir` (default `landscapyml_run_conf`).

## Usage examples
- List available registry entries:
  ```bash
  python -m landscapyml list
  ```
- Train with the default `conf/config.yaml` and override the number of classes and max epochs:
  ```bash
  python -m landscapyml train model_kwargs.num_classes=4 trainer_kwargs.max_epochs=10
  ```
- Train using a different config directory:
  ```bash
  python -m landscapyml train --config-path /path/to/conf model=sequence_mlp_classifier
  ```
- Generate a config from a CSV containing sequences and labels:
  ```bash
  python -m landscapyml config-from-csv \
    --csv-path data.csv \
    --sequence-column sequence \
    --label-column family \
    --out-dir landscapyml_run_conf \
    --embedding-mode hard \
    --model-key sequence_gp_classifier
  ```

## Runtime behavior
- Hydra creates a run directory under `outputs/YYYY-MM-DD/HH-MM-SS` by default; TensorBoard logs and checkpoints are placed inside this directory under `logs/` and `checkpoints/` unless overridden.
- `--config-path` sets the environment variable `LANDSCAPYML_CONFIG_PATH` used by `hydra_runner` to locate configs.
- CLI raises `SystemExit` with a non-zero code if Hydra returns an error.
