#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(realpath "${script_dir}/../..")
recipe="${repo_root}/examples/configs/recipes/llm/grpo-qwen3-30ba3b-thinking-swe1-2n4g-megatron-tp2pp2-rollout-only-specdec.yaml"

: "${HF_HOME:?HF_HOME must point to one stable shared Hugging Face cache}"
: "${NRL_TARGET_MODEL:?NRL_TARGET_MODEL must point to the Thinking target checkpoint}"
: "${NRL_DRAFT_MODEL:?NRL_DRAFT_MODEL must point to the exported ModelOpt drafter}"
: "${NRL_RUNTIME:?NRL_RUNTIME must point to one pre-created shared NeMo-RL environment}"
: "${NRL_OUTPUT_DIR:?NRL_OUTPUT_DIR must be an absolute experiment output path}"
: "${NRL_SPEC_METHOD:?NRL_SPEC_METHOD must be dflash or dspark}"
: "${NRL_NUM_SPECULATIVE_TOKENS:?NRL_NUM_SPECULATIVE_TOKENS must be between 1 and 64}"

for path_name in HF_HOME NRL_TARGET_MODEL NRL_DRAFT_MODEL NRL_RUNTIME NRL_OUTPUT_DIR; do
  path_value=${!path_name}
  if [[ ${path_value} != /* ]]; then
    echo "${path_name} must be an absolute path" >&2
    exit 2
  fi
done

case "${NRL_SPEC_METHOD}" in
  dflash | dspark) ;;
  *)
    echo "unsupported speculative method: ${NRL_SPEC_METHOD}" >&2
    exit 2
    ;;
esac

if [[ ! ${NRL_NUM_SPECULATIVE_TOKENS} =~ ^[0-9]+$ ]] ||
  ((NRL_NUM_SPECULATIVE_TOKENS < 1 || NRL_NUM_SPECULATIVE_TOKENS > 64)); then
  echo "unsupported speculative horizon: ${NRL_NUM_SPECULATIVE_TOKENS}; expected 1..64" >&2
  exit 2
fi

job_id=${SLURM_JOB_ID:-local}
runtime_root="/tmp/nemorl-specdec-${job_id}"
export WANDB_DIR="${WANDB_DIR:-${runtime_root}/wandb}"
export XDG_CACHE_HOME="${runtime_root}/xdg"
export TRITON_CACHE_DIR="${runtime_root}/triton"
export TORCHINDUCTOR_CACHE_DIR="${runtime_root}/torchinductor"
export UV_PROJECT_ENVIRONMENT="${NRL_RUNTIME}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HF_HOME}/uv}"
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

run_name=${WANDB_RUN_NAME:-qwen3-30ba3b-thinking-${NRL_SPEC_METHOD}-k${NRL_NUM_SPECULATIVE_TOKENS}-swe-rollout-only}
command=(
  uv run --frozen --no-sync examples/nemo_gym/run_grpo_nemo_gym.py
  --config "${recipe}"
  "policy.model_name=${NRL_TARGET_MODEL}"
  "policy.tokenizer.name=${NRL_TARGET_MODEL}"
  "logger.log_dir=${NRL_OUTPUT_DIR}/logs"
  "logger.wandb.project=${WANDB_PROJECT:-nemo-rl}"
  "logger.wandb.name=${run_name}"
)

if [[ -n ${DRY_RUN:-} ]]; then
  echo "NRL_NUM_SPECULATIVE_TOKENS=${NRL_NUM_SPECULATIVE_TOKENS}"
  echo "UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
  echo "NEMO_RL_PY_EXECUTABLES_SYSTEM=${NEMO_RL_PY_EXECUTABLES_SYSTEM}"
  echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
  echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
  echo "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

for required_path in \
  "${HF_HOME}" \
  "${NRL_TARGET_MODEL}" \
  "${NRL_DRAFT_MODEL}" \
  "${NRL_RUNTIME}/bin/python"; do
  if [[ ! -e ${required_path} ]]; then
    echo "required path does not exist: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${NRL_OUTPUT_DIR}"
cd "${repo_root}"
exec "${command[@]}"
