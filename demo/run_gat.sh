#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
MODEL_KWARGS="${MODEL_KWARGS:-{\"hidden_channels\": 64, \"num_layers\": 2, \"heads\": 4, \"dropout\": 0.1, \"learning_rate\": 0.001}}"
TRAINER_KWARGS="${TRAINER_KWARGS:-{\"use_wandb\": false, \"checkpoint_monitor\": null}}"

python -m landscapyml train-landscape \
  --demo-root "${SCRIPT_DIR}" \
  --model-key graph_attention_regressor \
  --data-name landscape_graph_regression \
  --max-epochs "${MAX_EPOCHS}" \
  --model-kwargs "${MODEL_KWARGS}" \
  --trainer-kwargs "${TRAINER_KWARGS}" \
  "$@"
