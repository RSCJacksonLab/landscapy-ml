#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIT_KWARGS="${FIT_KWARGS:-{\"training_iters\": 100, \"learning_rate\": 0.1}}"

python -m landscapyml train-landscape \
  --demo-root "${SCRIPT_DIR}" \
  --model-key diffusion_prior_gp \
  --fit-kwargs "${FIT_KWARGS}" \
  "$@"
