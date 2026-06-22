#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${PROJECT_ROOT}"

CONDA_ENV="${CONDA_ENV-dmd2-jittor}"
PYTHON_BIN="${PYTHON_BIN:-python}"
make_conda_run "${CONDA_ENV}" "${PYTHON_BIN}"
setup_jittor_env

RUN_NAME="${RUN_NAME:-cifar10_debug}"
METRICS_LOG="${METRICS_LOG:-logs/${RUN_NAME}/train_metrics.jsonl}"
PERFORMANCE_LOG="${PERFORMANCE_LOG:-logs/${RUN_NAME}/performance.jsonl}"
SAMPLES_INPUT="${SAMPLES_INPUT:-outputs/samples/${RUN_NAME}}"
REF_IMAGES="${REF_IMAGES:-data/ref_images}"

LOSS_CURVE="${LOSS_CURVE:-outputs/curves/${RUN_NAME}_loss.svg}"
PERF_CURVE="${PERF_CURVE:-outputs/curves/${RUN_NAME}_performance.svg}"
SAMPLE_GRID="${SAMPLE_GRID:-outputs/grids/${RUN_NAME}_samples.svg}"
FID_OUTPUT="${FID_OUTPUT:-outputs/fid_results/${RUN_NAME}_fid.json}"
IMAGE_SIZE="${IMAGE_SIZE:-32}"
MAX_IMAGES="${MAX_IMAGES:-64}"
NROW="${NROW:-8}"

if [[ -f "${METRICS_LOG}" ]]; then
  "${RUN[@]}" tools/plot_metrics.py \
    "${METRICS_LOG}" \
    "${LOSS_CURVE}" \
    --keys "${LOSS_KEYS:-loss_generator,loss_guidance,generator/loss_dm,guidance/loss_fake_mean}" \
    --title "${RUN_NAME} loss"
else
  echo "skip loss curve; metric log not found: ${METRICS_LOG}"
fi

if [[ -f "${PERFORMANCE_LOG}" ]]; then
  "${RUN[@]}" tools/plot_metrics.py \
    "${PERFORMANCE_LOG}" \
    "${PERF_CURVE}" \
    --keys "${PERF_KEYS:-samples_per_second,step_time,data_time}" \
    --title "${RUN_NAME} performance"
else
  echo "skip performance curve; performance log not found: ${PERFORMANCE_LOG}"
fi

if [[ -e "${SAMPLES_INPUT}" ]]; then
  "${RUN[@]}" tools/visualize_samples.py \
    "${SAMPLES_INPUT}" \
    "${SAMPLE_GRID}" \
    --nrow "${NROW}" \
    --image-size "${IMAGE_SIZE}" \
    --max-images "${MAX_IMAGES}"
else
  echo "skip sample grid; sample input not found: ${SAMPLES_INPUT}"
fi

if [[ -e "${SAMPLES_INPUT}" && -e "${REF_IMAGES}" ]]; then
  "${RUN[@]}" tools/compute_fid.py \
    --pred "${SAMPLES_INPUT}" \
    --ref "${REF_IMAGES}" \
    --output "${FID_OUTPUT}" \
    --image-size "${IMAGE_SIZE}" \
    --max-images "${MAX_IMAGES}"
else
  echo "skip FID; need generated samples (${SAMPLES_INPUT}) and reference images (${REF_IMAGES})"
fi

echo "evaluation export finished."
