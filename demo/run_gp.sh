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
FIT_KWARGS="${FIT_KWARGS:-}"
if [[ -z "${FIT_KWARGS}" ]]; then
  FIT_KWARGS='{"training_iters": 100, "learning_rate": 0.1, "normalize_features": true}'
fi
USE_DEMO_ROOT=true
USE_FIT_KWARGS=true
for arg in "$@"; do
  if [[ "${arg}" == "--csv-path" || "${arg}" == --csv-path=* ]]; then
    USE_DEMO_ROOT=false
  elif [[ "${arg}" == "--fit-kwargs" || "${arg}" == --fit-kwargs=* ]]; then
    USE_FIT_KWARGS=false
  fi
done

COMMAND=(
  "${PYTHON_BIN}" -m landscapyml train-landscape
  --model-key diffusion_prior_gp
)
if [[ "${USE_DEMO_ROOT}" == true ]]; then
  COMMAND+=(--demo-root "${SCRIPT_DIR}")
fi
if [[ "${USE_FIT_KWARGS}" == true ]]; then
  COMMAND+=(--fit-kwargs "${FIT_KWARGS}")
fi
COMMAND+=("$@")

"${COMMAND[@]}"
