#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${PROJECT_ROOT}"

CONDA_ENV="${CONDA_ENV-dmd2-jittor}"
PYTHON_BIN="${PYTHON_BIN:-python}"
make_conda_run "${CONDA_ENV}" "${PYTHON_BIN}"
setup_jittor_env

RUN_NAME="${RUN_NAME:-cifar10_debug}"
DATA_ROOT="${DATA_ROOT:-data/cifar10}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/${RUN_NAME}}"
SAMPLE_DIR="${SAMPLE_DIR:-outputs/samples/${RUN_NAME}}"
METRICS_LOG="${METRICS_LOG:-logs/${RUN_NAME}/train_metrics.jsonl}"
PERFORMANCE_LOG="${PERFORMANCE_LOG:-logs/${RUN_NAME}/performance.jsonl}"
LOSS_CURVE="${LOSS_CURVE:-outputs/curves/${RUN_NAME}_loss.svg}"
PERF_CURVE="${PERF_CURVE:-outputs/curves/${RUN_NAME}_performance.svg}"
LOSS_SUMMARY="${LOSS_SUMMARY:-outputs/curves/${RUN_NAME}_loss_summary.json}"
PERF_SUMMARY="${PERF_SUMMARY:-outputs/curves/${RUN_NAME}_performance_summary.json}"

MAX_STEPS="${MAX_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-1024}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
NO_AUGMENT="${NO_AUGMENT:-0}"

ARGS=(
  --data-root "${DATA_ROOT}"
  --max-steps "${MAX_STEPS}"
  --batch-size "${BATCH_SIZE}"
  --max-samples "${MAX_SAMPLES}"
  --log-interval "${LOG_INTERVAL}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL}"
  --eval-interval "${EVAL_INTERVAL}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --sample-dir "${SAMPLE_DIR}"
  --metrics-log "${METRICS_LOG}"
  --performance-log "${PERFORMANCE_LOG}"
)

if [[ "${NO_AUGMENT}" == "1" ]]; then
  ARGS+=(--no-augment)
fi
if [[ "${USE_CUDA}" == "1" ]]; then
  ARGS+=(--use-cuda)
fi
if [[ -n "${CLASS_SUBSET:-}" ]]; then
  ARGS+=(--class-subset "${CLASS_SUBSET}")
fi

"${RUN[@]}" tools/train_cifar10_debug.py "${ARGS[@]}" "$@"

if [[ -f "${METRICS_LOG}" ]]; then
  "${RUN[@]}" tools/plot_metrics.py \
    "${METRICS_LOG}" \
    "${LOSS_CURVE}" \
    --keys "${LOSS_KEYS:-loss_generator,loss_guidance,generator/loss_dm,guidance/loss_fake_mean}" \
    --title "${RUN_NAME} loss" \
    --summary-json "${LOSS_SUMMARY}"
fi

if [[ -f "${PERFORMANCE_LOG}" ]]; then
  "${RUN[@]}" tools/plot_metrics.py \
    "${PERFORMANCE_LOG}" \
    "${PERF_CURVE}" \
    --keys "${PERF_KEYS:-samples_per_second,step_time,data_time}" \
    --title "${RUN_NAME} performance" \
    --summary-json "${PERF_SUMMARY}"
fi

echo "CIFAR-10 debug run finished."
echo "metrics: ${METRICS_LOG}"
echo "performance: ${PERFORMANCE_LOG}"
echo "loss curve: ${LOSS_CURVE}"
echo "performance curve: ${PERF_CURVE}"
