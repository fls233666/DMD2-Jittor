#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

make_conda_run() {
  local env_name="$1"
  local python_bin="${2:-python}"

  if [[ -n "${env_name}" ]]; then
    RUN=(conda run --no-capture-output -n "${env_name}" "${python_bin}")
  else
    RUN=("${python_bin}")
  fi
}

make_named_conda_run() {
  local array_name="$1"
  local env_name="$2"
  local python_bin="${3:-python}"
  local -n output_array="${array_name}"

  if [[ -n "${env_name}" ]]; then
    output_array=(conda run --no-capture-output -n "${env_name}" "${python_bin}")
  else
    output_array=("${python_bin}")
  fi
}

setup_jittor_env() {
  local cache_root="${JITTOR_CACHE_ROOT:-/tmp/dmd2_jittor_cpu_home}"
  export HOME="${DMD2_SCRIPT_HOME:-${cache_root}}"
  export JITTOR_HOME="${JITTOR_HOME:-${cache_root}}"

  USE_CUDA="${USE_CUDA:-0}"
  if [[ "${USE_CUDA}" != "1" ]]; then
    export nvcc_path="${nvcc_path:-}"
  fi
}
