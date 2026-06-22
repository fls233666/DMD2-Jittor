#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${PROJECT_ROOT}"

CONDA_ENV="${CONDA_ENV-dmd2-jittor}"
PYTHON_BIN="${PYTHON_BIN:-python}"
make_conda_run "${CONDA_ENV}" "${PYTHON_BIN}"
setup_jittor_env

DATASET="${DATASET:-cifar10}"
DATA_ROOT="${DATA_ROOT:-data/cifar10}"
METHOD="${METHOD:-auto}"
RETRIES="${RETRIES:-20}"

ARGS=(
  --dataset "${DATASET}"
  --root "${DATA_ROOT}"
  --method "${METHOD}"
  --retries "${RETRIES}"
)

if [[ "${NO_RESUME:-0}" == "1" ]]; then
  ARGS+=(--no-resume)
fi
if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  ARGS+=(--check-only)
fi
if [[ "${REQUIRE_ARCHIVE:-0}" == "1" ]]; then
  ARGS+=(--require-archive)
fi
if [[ "${TRAIN_ONLY:-0}" == "1" ]]; then
  ARGS+=(--train-only)
fi
if [[ "${TEST_ONLY:-0}" == "1" ]]; then
  ARGS+=(--test-only)
fi

"${RUN[@]}" tools/download_datasets.py "${ARGS[@]}" "$@"

echo "dataset download finished."
