#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${REPO_ROOT}/.env/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.env/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
MAX_EPOCHS="${MAX_EPOCHS:-50}"
MODEL_KWARGS="${MODEL_KWARGS:-}"
DATA_KWARGS="${DATA_KWARGS:-}"
TRAINER_KWARGS="${TRAINER_KWARGS:-}"
if [[ -z "${MODEL_KWARGS}" ]]; then
  MODEL_KWARGS='{"hidden_channels": 64, "num_layers": 2, "heads": 4, "dropout": 0.1, "learning_rate": 0.001}'
fi
if [[ -z "${DATA_KWARGS}" ]]; then
  DATA_KWARGS='{"normalize_features": true}'
fi
if [[ -z "${TRAINER_KWARGS}" ]]; then
  TRAINER_KWARGS='{"use_wandb": false, "checkpoint_monitor": null}'
fi
USE_DEMO_ROOT=true
USE_DATA_KWARGS=true
for arg in "$@"; do
  if [[ "${arg}" == "--csv-path" || "${arg}" == --csv-path=* ]]; then
    USE_DEMO_ROOT=false
  elif [[ "${arg}" == "--data-kwargs" || "${arg}" == --data-kwargs=* ]]; then
    USE_DATA_KWARGS=false
  fi
done

COMMAND=(
  "${PYTHON_BIN}" -m landscapyml train-landscape
  --model-key graph_attention_regressor
  --data-name landscape_graph_regression
  --max-epochs "${MAX_EPOCHS}"
  --model-kwargs "${MODEL_KWARGS}"
  --trainer-kwargs "${TRAINER_KWARGS}"
)
if [[ "${USE_DEMO_ROOT}" == true ]]; then
  COMMAND+=(--demo-root "${SCRIPT_DIR}")
fi
if [[ "${USE_DATA_KWARGS}" == true ]]; then
  COMMAND+=(--data-kwargs "${DATA_KWARGS}")
fi
COMMAND+=("$@")

"${COMMAND[@]}"
