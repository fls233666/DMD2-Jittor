#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${PROJECT_ROOT}"

CONDA_ENV="${CONDA_ENV-dmd2-jittor}"
PYTHON_BIN="${PYTHON_BIN:-python}"
make_conda_run "${CONDA_ENV}" "${PYTHON_BIN}"
setup_jittor_env

RUN_NAME="${RUN_NAME:-image_dmd2}"
DATASET_NAME="${DATASET_NAME:-cifar10}"
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
MAX_SAMPLES="${MAX_SAMPLES-1024}"
IMAGE_SIZE="${IMAGE_SIZE:-32}"
LABEL_DIM="${LABEL_DIM:-10}"
SEED="${SEED:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
NROW="${NROW:-4}"
NO_AUGMENT="${NO_AUGMENT:-0}"
USE_FP16="${USE_FP16:-0}"
TEACHER_CONFIG="${TEACHER_CONFIG:-tiny}"
REAL_UNET_CHECKPOINT="${REAL_UNET_CHECKPOINT:-}"
REAL_UNET_STATE_KEY="${REAL_UNET_STATE_KEY:-}"
INIT_FAKE_FROM_REAL="${INIT_FAKE_FROM_REAL:-0}"
INIT_GENERATOR_FROM_REAL="${INIT_GENERATOR_FROM_REAL:-0}"
SKIP_REAL_LINEAR_TRANSPOSE_CACHE="${SKIP_REAL_LINEAR_TRANSPOSE_CACHE:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
RESUME_MODEL_ONLY="${RESUME_MODEL_ONLY:-0}"
DFAKE_GEN_UPDATE_RATIO="${DFAKE_GEN_UPDATE_RATIO:-1}"
GAN_CLASSIFIER="${GAN_CLASSIFIER:-0}"
GEN_CLS_LOSS_WEIGHT="${GEN_CLS_LOSS_WEIGHT:-0.0}"
CLS_LOSS_WEIGHT="${CLS_LOSS_WEIGHT:-1.0}"
DIFFUSION_GAN="${DIFFUSION_GAN:-0}"
DIFFUSION_GAN_MAX_TIMESTEP="${DIFFUSION_GAN_MAX_TIMESTEP:-1}"
LR_GENERATOR="${LR_GENERATOR:-2e-4}"
LR_GUIDANCE="${LR_GUIDANCE:-2e-4}"
ADAM_BETA1="${ADAM_BETA1:-0.0}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
OPTIMIZER="${OPTIMIZER:-adam}"
WARMUP_STEP="${WARMUP_STEP:-0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-}"
EMA_DECAY="${EMA_DECAY:-0.999}"
NO_EMA="${NO_EMA:-0}"
SKIP_FINAL_CHECKPOINT="${SKIP_FINAL_CHECKPOINT:-0}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}"
FINAL_CHECKPOINT="${FINAL_CHECKPOINT:-}"

ARGS=(
  --dataset-name "${DATASET_NAME}"
  --data-root "${DATA_ROOT}"
  --max-steps "${MAX_STEPS}"
  --batch-size "${BATCH_SIZE}"
  --image-size "${IMAGE_SIZE}"
  --label-dim "${LABEL_DIM}"
  --seed "${SEED}"
  --log-interval "${LOG_INTERVAL}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL}"
  --eval-interval "${EVAL_INTERVAL}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --nrow "${NROW}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --sample-dir "${SAMPLE_DIR}"
  --metrics-log "${METRICS_LOG}"
  --performance-log "${PERFORMANCE_LOG}"
  --teacher-config "${TEACHER_CONFIG}"
  --lr-generator "${LR_GENERATOR}"
  --lr-guidance "${LR_GUIDANCE}"
  --adam-beta1 "${ADAM_BETA1}"
  --adam-beta2 "${ADAM_BETA2}"
  --weight-decay "${WEIGHT_DECAY}"
  --optimizer "${OPTIMIZER}"
  --warmup-step "${WARMUP_STEP}"
  --ema-decay "${EMA_DECAY}"
  --dfake-gen-update-ratio "${DFAKE_GEN_UPDATE_RATIO}"
  --gen-cls-loss-weight "${GEN_CLS_LOSS_WEIGHT}"
  --cls-loss-weight "${CLS_LOSS_WEIGHT}"
  --diffusion-gan-max-timestep "${DIFFUSION_GAN_MAX_TIMESTEP}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "${NO_AUGMENT}" == "1" ]]; then
  ARGS+=(--no-augment)
fi
if [[ "${USE_CUDA}" == "1" ]]; then
  ARGS+=(--use-cuda)
fi
if [[ "${USE_FP16}" == "1" ]]; then
  ARGS+=(--use-fp16)
fi
if [[ -n "${CLASS_SUBSET:-}" ]]; then
  ARGS+=(--class-subset "${CLASS_SUBSET}")
fi
if [[ -n "${MAX_GRAD_NORM}" ]]; then
  ARGS+=(--max-grad-norm "${MAX_GRAD_NORM}")
fi
if [[ -n "${REAL_UNET_CHECKPOINT}" ]]; then
  ARGS+=(--real-unet-checkpoint "${REAL_UNET_CHECKPOINT}")
fi
if [[ -n "${REAL_UNET_STATE_KEY}" ]]; then
  ARGS+=(--real-unet-state-key "${REAL_UNET_STATE_KEY}")
fi
if [[ "${INIT_FAKE_FROM_REAL}" == "1" ]]; then
  ARGS+=(--init-fake-from-real)
fi
if [[ "${INIT_GENERATOR_FROM_REAL}" == "1" ]]; then
  ARGS+=(--init-generator-from-real)
fi
if [[ "${SKIP_REAL_LINEAR_TRANSPOSE_CACHE}" == "1" ]]; then
  ARGS+=(--skip-real-linear-transpose-cache)
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  ARGS+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi
if [[ "${RESUME_MODEL_ONLY}" == "1" ]]; then
  ARGS+=(--resume-model-only)
fi
if [[ "${GAN_CLASSIFIER}" == "1" ]]; then
  ARGS+=(--gan-classifier)
fi
if [[ "${NO_EMA}" == "1" ]]; then
  ARGS+=(--no-ema)
fi
if [[ "${DIFFUSION_GAN}" == "1" ]]; then
  ARGS+=(--diffusion-gan)
fi
if [[ -n "${FINAL_CHECKPOINT}" ]]; then
  ARGS+=(--final-checkpoint "${FINAL_CHECKPOINT}")
fi
if [[ "${SKIP_FINAL_CHECKPOINT}" == "1" ]]; then
  ARGS+=(--skip-final-checkpoint)
fi
if [[ "${SKIP_FINAL_EVAL}" == "1" ]]; then
  ARGS+=(--skip-final-eval)
fi

"${RUN[@]}" tools/train_image_dmd2.py "${ARGS[@]}" "$@"

if [[ -f "${METRICS_LOG}" ]]; then
  "${RUN[@]}" tools/plot_metrics.py \
    "${METRICS_LOG}" \
    "${LOSS_CURVE}" \
    --keys "${LOSS_KEYS:-loss_generator,loss_guidance,generator/loss_dm,generator/gen_cls_loss,guidance/loss_fake_mean,guidance/guidance_cls_loss}" \
    --title "${RUN_NAME} loss" \
    --summary-json "${LOSS_SUMMARY}" \
    --split
fi

if [[ -f "${PERFORMANCE_LOG}" ]]; then
  "${RUN[@]}" tools/plot_metrics.py \
    "${PERFORMANCE_LOG}" \
    "${PERF_CURVE}" \
    --keys "${PERF_KEYS:-samples_per_second,step_time,data_time}" \
    --title "${RUN_NAME} performance" \
    --summary-json "${PERF_SUMMARY}"
fi

echo "${DATASET_NAME} image DMD2 run finished."
echo "metrics: ${METRICS_LOG}"
echo "performance: ${PERFORMANCE_LOG}"
echo "loss curves: ${LOSS_CURVE%.*}/"
echo "performance curve: ${PERF_CURVE}"
