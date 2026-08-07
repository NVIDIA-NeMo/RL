#!/usr/bin/env bash
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

# Four-step public Nemotron-3 Nano SWE/W&B acceptance run: four 4-GPU policy
# nodes plus two TP4 managed Dynamo generation nodes.

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to a NeMo-RL image built with BUILD_DYNAMO=1}"
: "${TRAIN_PATH:?Set TRAIN_PATH to the SWE training JSONL}"
: "${VAL_PATH:?Set VAL_PATH to the SWE validation JSONL}"
: "${SIF_FORMATTERS:?Set SIF_FORMATTERS to a Hydra list of SWE image format strings}"
: "${SANDBOX_CONTAINER:?Set SANDBOX_CONTAINER to the NeMo Skills sandbox image}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}"
: "${SLURM_PARTITION:?Set SLURM_PARTITION}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${REPO_ROOT:-$(cd "${script_dir}/../.." && pwd)}
config_path=${CONFIG_PATH:-${repo_root}/examples/configs/recipes/llm/grpo-nanov3-30ba3b-swe-6n4g-megatron-dynamo-wandb.yaml}
entrypoint=${ENTRYPOINT:-${repo_root}/examples/nemo_gym/run_grpo_nemo_gym.py}
ray_sub=${RAY_SUB:-${repo_root}/ray.sub}
model_name=${MODEL_NAME:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}
wandb_project=${WANDB_PROJECT:-nemo-rl-dynamo-swe}
wandb_name=${WANDB_NAME:-nemotron-3-nano-swe-dynamo-${USER}}
results_dir=${RESULTS_DIR:-${repo_root}/results/${wandb_name}}
log_dir=${LOG_DIR:-${results_dir}/logs}
slurm_time_limit=${SLURM_TIME_LIMIT:-04:00:00}
dry_run=${DRY_RUN:-0}

require_path() {
  local path=$1
  local label=$2
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_path "${config_path}" "recipe"
require_path "${entrypoint}" "entrypoint"
require_path "${ray_sub}" "ray.sub"
if [[ "${dry_run}" != "1" ]]; then
  require_path "${CONTAINER}" "Dynamo-enabled container"
  require_path "${TRAIN_PATH}" "training dataset"
  require_path "${VAL_PATH}" "validation dataset"
  require_path "${SANDBOX_CONTAINER}" "sandbox container"
fi

mkdir -p "${results_dir}" "${log_dir}"

export WANDB_API_KEY
export WANDB_MODE=online
export GPUS_PER_NODE=4
export BASE_LOG_DIR="${log_dir}"
export NEMO_SKILLS_SANDBOX_PORT=${NEMO_SKILLS_SANDBOX_PORT:-6000}
export SANDBOX_COMMAND=${SANDBOX_COMMAND:-/start-with-nginx.sh}
export SANDBOX_ENV_VARS="NEMO_SKILLS_SANDBOX_PORT=${NEMO_SKILLS_SANDBOX_PORT}"
export SANDBOX_CONTAINER
export MOUNTS="${repo_root}:${repo_root}${EXTRA_MOUNTS:+,${EXTRA_MOUNTS}}"

command=(
  /opt/nemo_rl_venv/bin/python
  -u
  "${entrypoint}"
  --config
  "${config_path}"
  "policy.model_name=${model_name}"
  "policy.tokenizer.name=${model_name}"
  "data.train.data_path=${TRAIN_PATH}"
  "data.validation.data_path=${VAL_PATH}"
  "env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.container_formatter=${SIF_FORMATTERS}"
  "logger.log_dir=${results_dir}"
  "logger.wandb.project=${wandb_project}"
  "logger.wandb.name=${wandb_name}"
  "checkpointing.checkpoint_dir=${results_dir}/checkpoints"
)
if [[ -n "${EXTRA_HYDRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_hydra_args=(${EXTRA_HYDRA_ARGS})
  command+=("${extra_hydra_args[@]}")
fi
printf -v COMMAND '%q ' "${command[@]}"
export COMMAND

sbatch_args=(
  --nodes=6
  --account="${SLURM_ACCOUNT}"
  --job-name="${wandb_name}"
  --partition="${SLURM_PARTITION}"
  --time="${slurm_time_limit}"
  --gres=gpu:4
  --exclusive
  --output="${log_dir}/slurm-%j.out"
)

if [[ "${dry_run}" == "1" ]]; then
  printf 'COMMAND=%q\n' "${COMMAND}"
  printf 'sbatch'
  printf ' %q' "${sbatch_args[@]}"
  printf ' %q\n' "${ray_sub}"
  exit 0
fi

sbatch "${sbatch_args[@]}" "${ray_sub}"
