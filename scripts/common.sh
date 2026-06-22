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
  else
    if [[ -d /usr/lib/wsl/lib ]]; then
      export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
    fi
    export cuda_arch="${cuda_arch:-89}"
    export conv_opt="${conv_opt:-1}"
    if [[ -z "${cc_path:-}" && -x /usr/bin/g++-10 ]]; then
      export cc_path="/usr/bin/g++-10"
    elif [[ -z "${cc_path:-}" ]]; then
      echo "WARNING: USE_CUDA=1 but /usr/bin/g++-10 was not found." >&2
      echo "Jittor with CUDA 11.5 may fail with the default /usr/bin/g++." >&2
      echo "Install it with: sudo apt install gcc-10 g++-10" >&2
    fi
  fi
}
