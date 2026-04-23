#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${REPO_ROOT}/.env/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.env/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
MAX_EPOCHS="${MAX_EPOCHS:-50}"
MODEL_KWARGS="${MODEL_KWARGS:-{\"hidden_channels\": 64, \"num_layers\": 2, \"heads\": 4, \"dropout\": 0.1, \"learning_rate\": 0.001}}"
DATA_KWARGS="${DATA_KWARGS:-{\"normalize_features\": true}}"
TRAINER_KWARGS="${TRAINER_KWARGS:-{\"use_wandb\": false, \"checkpoint_monitor\": null}}"
DEMO_ROOT_ARGS=(--demo-root "${SCRIPT_DIR}")
DATA_KWARGS_ARGS=(--data-kwargs "${DATA_KWARGS}")
for arg in "$@"; do
  if [[ "${arg}" == "--csv-path" || "${arg}" == --csv-path=* ]]; then
    DEMO_ROOT_ARGS=()
  elif [[ "${arg}" == "--data-kwargs" || "${arg}" == --data-kwargs=* ]]; then
    DATA_KWARGS_ARGS=()
  fi
done

"${PYTHON_BIN}" -m landscapyml train-landscape \
  "${DEMO_ROOT_ARGS[@]}" \
  --model-key graph_attention_regressor \
  --data-name landscape_graph_regression \
  --max-epochs "${MAX_EPOCHS}" \
  --model-kwargs "${MODEL_KWARGS}" \
  "${DATA_KWARGS_ARGS[@]}" \
  --trainer-kwargs "${TRAINER_KWARGS}" \
  "$@"
