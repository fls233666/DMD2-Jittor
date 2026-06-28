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
PYTORCH_CKPT="${PYTORCH_CKPT:-../DMD2-pytorch/checkpoints/imagenet_fid151/pytorch_model.bin}"
CONVERTED_CKPT="${CONVERTED_CKPT:-checkpoints/imagenet64_demo/generator_jittor.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/alignment/imagenet64_forward}"
LOG_PATH="${LOG_PATH:-logs/imagenet64_forward_alignment.log}"
REPORT_JSON="${REPORT_JSON:-${OUTPUT_DIR}/report.json}"
INPUT_NPZ="${INPUT_NPZ:-${OUTPUT_DIR}/input.npz}"
PYTORCH_OUTPUT_NPZ="${PYTORCH_OUTPUT_NPZ:-${OUTPUT_DIR}/pytorch_output.npz}"
JITTOR_OUTPUT_NPZ="${JITTOR_OUTPUT_NPZ:-${OUTPUT_DIR}/jittor_output.npz}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEED="${SEED:-10}"
LABELS="${LABELS:-}"
RTOL="${RTOL:-1e-4}"
ATOL="${ATOL:-2e-4}"
STRICT="${STRICT:-0}"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_PATH}")"

if [[ ! -f "${CONVERTED_CKPT}" ]]; then
  TARGET=generator \
  PYTORCH_ROOT="${PYTORCH_ROOT}" \
  PYTORCH_CKPT="${PYTORCH_CKPT}" \
  SAVE_CONVERTED="${CONVERTED_CKPT}" \
  "${SCRIPT_DIR}/align_pytorch_jittor.sh"
fi

COMMON_ARGS=(
  --input-npz "${INPUT_NPZ}"
  --batch-size "${BATCH_SIZE}"
  --resolution 64
  --label-dim 1000
  --conditioning-sigma 80.0
  --sigma-data 0.5
  --seed "${SEED}"
)
if [[ -n "${LABELS}" ]]; then
  COMMON_ARGS+=(--labels "${LABELS}")
fi

"${JITTOR_RUN[@]}" tools/align_imagenet64_forward.py \
  --mode make-input \
  "${COMMON_ARGS[@]}"

PYTORCH_FORWARD_ARGS=(
  --mode pytorch-forward
  --pytorch-root "${PYTORCH_ROOT}"
  --pytorch-ckpt "${PYTORCH_CKPT}"
  --output-npz "${PYTORCH_OUTPUT_NPZ}"
  "${COMMON_ARGS[@]}"
)
if [[ "${PYTORCH_USE_CUDA:-${USE_CUDA:-0}}" == "1" ]]; then
  PYTORCH_FORWARD_ARGS+=(--use-cuda)
fi
"${PYTORCH_RUN[@]}" tools/align_imagenet64_forward.py "${PYTORCH_FORWARD_ARGS[@]}"

JITTOR_FORWARD_ARGS=(
  --mode jittor-forward
  --jittor-root "${PROJECT_ROOT}"
  --converted-ckpt "${CONVERTED_CKPT}"
  --output-npz "${JITTOR_OUTPUT_NPZ}"
  "${COMMON_ARGS[@]}"
)
if [[ "${USE_CUDA:-0}" == "1" ]]; then
  JITTOR_FORWARD_ARGS+=(--use-cuda)
fi
"${JITTOR_RUN[@]}" tools/align_imagenet64_forward.py "${JITTOR_FORWARD_ARGS[@]}"

COMPARE_ARGS=(
  --mode compare
  --pytorch-output-npz "${PYTORCH_OUTPUT_NPZ}"
  --jittor-output-npz "${JITTOR_OUTPUT_NPZ}"
  --report-json "${REPORT_JSON}"
  --log-path "${LOG_PATH}"
  --rtol "${RTOL}"
  --atol "${ATOL}"
)
if [[ "${STRICT}" == "1" ]]; then
  COMPARE_ARGS+=(--strict)
fi
"${JITTOR_RUN[@]}" tools/align_imagenet64_forward.py "${COMPARE_ARGS[@]}"

echo "ImageNet-64 forward alignment finished."
echo "input: ${INPUT_NPZ}"
echo "pytorch output: ${PYTORCH_OUTPUT_NPZ}"
echo "jittor output: ${JITTOR_OUTPUT_NPZ}"
echo "report: ${REPORT_JSON}"
echo "log: ${LOG_PATH}"
