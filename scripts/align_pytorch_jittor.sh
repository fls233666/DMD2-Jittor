#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${PROJECT_ROOT}"

TARGET="${TARGET:-generator}"
case "${TARGET}" in
  generator)
    DEFAULT_CKPT="../DMD2-pytorch/checkpoints/imagenet_fid151/pytorch_model.bin"
    DEFAULT_OUTPUT="checkpoints/imagenet64_demo/generator_jittor.pkl"
    ;;
  guidance)
    DEFAULT_CKPT="../DMD2-pytorch/checkpoints/imagenet_fid151/pytorch_model_1.bin"
    DEFAULT_OUTPUT="checkpoints/imagenet64_demo/guidance_jittor.pkl"
    ;;
  *)
    echo "TARGET must be generator or guidance, got: ${TARGET}" >&2
    exit 2
    ;;
esac

PYTORCH_CKPT="${PYTORCH_CKPT:-${DEFAULT_CKPT}}"
PYTORCH_ROOT="${PYTORCH_ROOT:-../DMD2-pytorch}"
SAVE_CONVERTED="${SAVE_CONVERTED:-${DEFAULT_OUTPUT}}"
SHAPES_PATH="${SHAPES_PATH:-/tmp/dmd2_${TARGET}_target_shapes.pkl}"
LOG_PATH="${LOG_PATH:-logs/pytorch_jittor_align_${TARGET}.log}"

JITTOR_CONDA_ENV="${JITTOR_CONDA_ENV-dmd2-jittor}"
PYTORCH_CONDA_ENV="${PYTORCH_CONDA_ENV-dmd2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

make_named_conda_run JITTOR_RUN "${JITTOR_CONDA_ENV}" "${PYTHON_BIN}"
make_named_conda_run PYTORCH_RUN "${PYTORCH_CONDA_ENV}" "${PYTHON_BIN}"
setup_jittor_env

TARGET_ARGS=(--target "${TARGET}")
if [[ "${TARGET}" == "guidance" && "${GAN_CLASSIFIER:-1}" == "1" ]]; then
  TARGET_ARGS+=(--gan-classifier)
fi
if [[ "${USE_CUDA}" == "1" ]]; then
  TARGET_ARGS+=(--use-cuda)
fi

mkdir -p "$(dirname "${SAVE_CONVERTED}")" "$(dirname "${LOG_PATH}")"

"${JITTOR_RUN[@]}" tools/compare_pytorch_jittor.py \
  --dump-target-shapes "${SHAPES_PATH}" \
  "${TARGET_ARGS[@]}" \
  --dataset-name imagenet \
  --config-name imagenet \
  --resolution 64 \
  --label-dim 1000

CONVERT_ARGS=(
  "${PYTORCH_CKPT}"
  "${SAVE_CONVERTED}"
  --pytorch-root "${PYTORCH_ROOT}"
  --target-state "${SHAPES_PATH}"
  --resolution 64
  --channel-mult 1,2,3,4
  --num-blocks 3
  --strict
)

CONVERT_OUTPUT="$("${PYTORCH_RUN[@]}" tools/convert_pytorch_ckpt.py "${CONVERT_ARGS[@]}" 2>&1)"
{
  echo "# PyTorch-Jittor alignment target=${TARGET}"
  echo "# pytorch_ckpt=${PYTORCH_CKPT}"
  echo "# converted=${SAVE_CONVERTED}"
  echo "# target_shapes=${SHAPES_PATH}"
  echo "${CONVERT_OUTPUT}"
} | tee "${LOG_PATH}"

echo "saved converted checkpoint: ${SAVE_CONVERTED}"
echo "saved alignment log: ${LOG_PATH}"
