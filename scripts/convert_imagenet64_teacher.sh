#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${PROJECT_ROOT}"

JITTOR_CONDA_ENV="${JITTOR_CONDA_ENV-dmd2-jittor}"
PYTORCH_CONDA_ENV="${PYTORCH_CONDA_ENV-dmd2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
make_named_conda_run JITTOR_RUN "${JITTOR_CONDA_ENV}" "${PYTHON_BIN}"
make_named_conda_run PYTORCH_RUN "${PYTORCH_CONDA_ENV}" "${PYTHON_BIN}"
setup_jittor_env

PYTORCH_ROOT="${PYTORCH_ROOT:-../DMD2-pytorch}"
PYTORCH_TEACHER="${PYTORCH_TEACHER:-${PYTORCH_ROOT}/checkpoints/edm-imagenet-64x64-cond-adm.pkl}"
OUTPUT_CKPT="${OUTPUT_CKPT:-checkpoints/imagenet64_teacher/edm_imagenet64_teacher_jittor.pkl}"
SHAPES_PATH="${SHAPES_PATH:-/tmp/imagenet64_teacher_shapes.pkl}"
LOG_PATH="${LOG_PATH:-logs/imagenet64_teacher_conversion.log}"
SOURCE_KEY="${SOURCE_KEY:-ema}"
STRICT="${STRICT:-1}"
USE_CUDA="${USE_CUDA:-0}"

mkdir -p "$(dirname "${OUTPUT_CKPT}")" "$(dirname "${LOG_PATH}")" "$(dirname "${SHAPES_PATH}")"

TARGET_ARGS=(
  --dump-target-shapes "${SHAPES_PATH}"
  --target generator
  --dataset-name imagenet
  --config-name imagenet
  --resolution 64
  --label-dim 1000
)
if [[ "${USE_CUDA}" == "1" ]]; then
  TARGET_ARGS+=(--use-cuda)
fi

"${JITTOR_RUN[@]}" tools/compare_pytorch_jittor.py "${TARGET_ARGS[@]}"

CONVERT_ARGS=(
  "${PYTORCH_TEACHER}"
  "${OUTPUT_CKPT}"
  --input-format pickle
  --source-key "${SOURCE_KEY}"
  --pytorch-root "${PYTORCH_ROOT}"
  --target-state "${SHAPES_PATH}"
  --resolution 64
  --channel-mult 1,2,3,4
  --num-blocks 3
)
if [[ "${STRICT}" == "1" ]]; then
  CONVERT_ARGS+=(--strict)
fi

CONVERT_OUTPUT="$("${PYTORCH_RUN[@]}" tools/convert_pytorch_ckpt.py "${CONVERT_ARGS[@]}" 2>&1)"
{
  echo "# ImageNet-64 EDM teacher conversion"
  echo "# pytorch_teacher=${PYTORCH_TEACHER}"
  echo "# source_key=${SOURCE_KEY}"
  echo "# output_ckpt=${OUTPUT_CKPT}"
  echo "# target_shapes=${SHAPES_PATH}"
  echo "${CONVERT_OUTPUT}"
} | tee "${LOG_PATH}"

echo "saved ImageNet-64 Jittor teacher: ${OUTPUT_CKPT}"
echo "saved conversion log: ${LOG_PATH}"
