#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${PROJECT_ROOT}"

CONDA_ENV="${CONDA_ENV-dmd2-jittor}"
PYTHON_BIN="${PYTHON_BIN:-python}"
make_conda_run "${CONDA_ENV}" "${PYTHON_BIN}"
setup_jittor_env

CONVERTED_CKPT="${CONVERTED_CKPT:-checkpoints/imagenet64_demo/generator_jittor.pkl}"
PYTORCH_CKPT="${PYTORCH_CKPT:-../DMD2-pytorch/checkpoints/imagenet_fid151/pytorch_model.bin}"
OUTPUT_GRID="${OUTPUT_GRID:-outputs/grids/imagenet64_demo.svg}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/samples/imagenet64_demo}"
OUTPUT_NPZ="${OUTPUT_NPZ:-outputs/samples/imagenet64_demo.npz}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NROW="${NROW:-4}"
CLASS_IDX="${CLASS_IDX:-}"
LABELS="${LABELS:-}"

if [[ ! -f "${CONVERTED_CKPT}" ]]; then
  if [[ -f "${PYTORCH_CKPT}" ]]; then
    TARGET=generator \
    PYTORCH_CKPT="${PYTORCH_CKPT}" \
    SAVE_CONVERTED="${CONVERTED_CKPT}" \
    "${SCRIPT_DIR}/align_pytorch_jittor.sh"
  else
    echo "Converted checkpoint not found: ${CONVERTED_CKPT}" >&2
    echo "Set CONVERTED_CKPT or PYTORCH_CKPT before running this script." >&2
    exit 2
  fi
fi

ARGS=(
  --checkpoint "${CONVERTED_CKPT}"
  --target generator
  --dataset-name imagenet
  --config-name imagenet
  --resolution 64
  --label-dim 1000
  --batch-size "${BATCH_SIZE}"
  --nrow "${NROW}"
  --output-grid "${OUTPUT_GRID}"
  --output-dir "${OUTPUT_DIR}"
  --output-npz "${OUTPUT_NPZ}"
)

if [[ "${USE_CUDA}" == "1" ]]; then
  ARGS+=(--use-cuda)
fi
if [[ -n "${CLASS_IDX}" ]]; then
  ARGS+=(--class-idx "${CLASS_IDX}")
fi
if [[ -n "${LABELS}" ]]; then
  ARGS+=(--labels "${LABELS}")
fi

"${RUN[@]}" tools/sample_one_step.py "${ARGS[@]}" "$@"

echo "ImageNet-64 demo sampling finished."
echo "grid: ${OUTPUT_GRID}"
echo "images: ${OUTPUT_DIR}"
echo "npz: ${OUTPUT_NPZ}"
