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
FIT_KWARGS="${FIT_KWARGS:-{\"training_iters\": 100, \"learning_rate\": 0.1, \"normalize_features\": true}}"
DEMO_ROOT_ARGS=(--demo-root "${SCRIPT_DIR}")
FIT_KWARGS_ARGS=(--fit-kwargs "${FIT_KWARGS}")
for arg in "$@"; do
  if [[ "${arg}" == "--csv-path" || "${arg}" == --csv-path=* ]]; then
    DEMO_ROOT_ARGS=()
  elif [[ "${arg}" == "--fit-kwargs" || "${arg}" == --fit-kwargs=* ]]; then
    FIT_KWARGS_ARGS=()
  fi
done

"${PYTHON_BIN}" -m landscapyml train-landscape \
  "${DEMO_ROOT_ARGS[@]}" \
  --model-key diffusion_prior_gp \
  "${FIT_KWARGS_ARGS[@]}" \
  "$@"
