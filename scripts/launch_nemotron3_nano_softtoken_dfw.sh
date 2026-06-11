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

BASE=/lustre/fsw/portfolios/llmservice/users/cmunley
JOB_PREFIX="${JOB_PREFIX:-nano3-softtoken-dfw}"
JOB_NAME="${JOB_NAME:-grpo-nemotron3-nano-30ba3b-softtoken-smoke}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)
cd "${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-${BASE}/hf_cache}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results/${JOB_NAME}}"
RUN_DIR="${RESULTS_DIR}/runs/$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${RUN_DIR}/logs"
SLURM_LOG_DIR="${RUN_DIR}/slurm"
CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints"
mkdir -p "${LOG_DIR}" "${SLURM_LOG_DIR}" "${CHECKPOINT_DIR}"
ln -sfn "${RUN_DIR}" "${RESULTS_DIR}/runs/latest"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_agents_dev}"
PARTITION="${PARTITION:-batch}"
WALLTIME="${WALLTIME:-2:00:00}"
NUM_TOTAL_NODES="${NUM_TOTAL_NODES:-2}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-128}"

export CONTAINER="${CONTAINER:-${BASE}/containers/rl.51132285.sqsh}"
[[ -f "${CONTAINER}" ]] || { echo "ERROR: container not found: ${CONTAINER}" >&2; exit 1; }

MOUNTS="/lustre:/lustre"
MOUNTS="${MOUNTS},${PROJECT_ROOT}/nemo_rl:/opt/nemo-rl/nemo_rl"
MOUNTS="${MOUNTS},${PROJECT_ROOT}/examples:/opt/nemo-rl/examples"
MOUNTS="${MOUNTS},${PROJECT_ROOT}/pyproject.toml:/opt/nemo-rl/pyproject.toml"
MOUNTS="${MOUNTS},${HF_HOME}:/root/.cache/huggingface"
[[ -n "${EXTRA_MOUNTS:-}" ]] && MOUNTS="${MOUNTS},${EXTRA_MOUNTS}"
export MOUNTS

PERSISTENT_CACHE="${PERSISTENT_CACHE:-${BASE}/.cache/${JOB_PREFIX}}"
INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
NRL_VLLM_LOCAL_CACHE_DIR="/tmp/nemo_rl_vllm_cache"
mkdir -p "${PERSISTENT_CACHE}"

read -r -d '' SETUP_COMMAND <<SETUPEOF || true
rm -rf "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${NRL_VLLM_LOCAL_CACHE_DIR}"
mkdir -p "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${NRL_VLLM_LOCAL_CACHE_DIR}"
SETUPEOF
export SETUP_COMMAND

CONFIG="examples/configs/recipes/llm/grpo-nemotron3-nano-30ba3b-softtoken-2n8g-fsdp2.yaml"
read -r -d '' TRAIN_CMD <<CMDEOF || true
cd /opt/nemo-rl && date
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_STANDALONE_COMPILE=0
export TORCH_CUDA_ARCH_LIST='9.0'
export RAY_DEDUP_LOGS=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export UV_HTTP_TIMEOUT=300
export UV_CACHE_DIR=${PERSISTENT_CACHE}/uv
export HF_HOME=${HF_HOME}
export HF_TOKEN=${HF_TOKEN:-}
export VLLM_CACHE_ROOT=${NRL_VLLM_LOCAL_CACHE_DIR}
export DG_JIT_CACHE_DIR=${NRL_VLLM_LOCAL_CACHE_DIR}/deep_gemm
export TORCHINDUCTOR_CACHE_DIR=${INDUCTOR_CACHE_DIR}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR}
OMP_NUM_THREADS=16 uv run --frozen ./examples/run_grpo_soft_tokens.py \
  --config ${CONFIG} \
  checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
  logger.log_dir=${LOG_DIR} \
  logger.wandb_enabled=False \
  logger.wandb.name=${JOB_NAME}-$(date +%Y%m%d-%H%M%S)
CMDEOF
export COMMAND="${TRAIN_CMD}"

RAY_SUB="${RAY_SUB:-${PROJECT_ROOT}/ray.sub}"
[[ -f "${RAY_SUB}" ]] || { echo "ERROR: ray.sub not found: ${RAY_SUB}" >&2; exit 1; }

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "${TRAIN_CMD}"
  exit 0
fi

SBATCH_OUTPUT=$(sbatch \
  --nodes="${NUM_TOTAL_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --job-name="${JOB_NAME}" \
  --partition="${PARTITION}" \
  --time="${WALLTIME}" \
  --gres=gpu:${GPUS_PER_NODE} \
  --exclusive --mem=0 \
  --output="${SLURM_LOG_DIR}/%j.out" \
  --error="${SLURM_LOG_DIR}/%j.err" \
  ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
  ${RESERVATION:+--reservation="${RESERVATION}"} \
  ${SBATCH_DEPENDENCY:+--dependency="${SBATCH_DEPENDENCY}"} \
  "${RAY_SUB}")

echo "${SBATCH_OUTPUT}"
JOB_ID=$(echo "${SBATCH_OUTPUT}" | grep -oP '\d+$')
if [[ -n "${JOB_ID}" ]]; then
  echo -e "\n  Slurm logs: ${SLURM_LOG_DIR}/${JOB_ID}.out\n  Monitor:    squeue -u \\$USER -j ${JOB_ID}\n  Run dir:    ${RUN_DIR}"
fi
