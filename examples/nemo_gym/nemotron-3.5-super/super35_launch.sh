#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# Four-GPU GB200 node counts for the reference YAMLs.
STAGE="${1:-}"
case "${STAGE}" in
  student_rlvr)
    NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-32}"
    NUM_GEN_NODES="${NUM_GEN_NODES:-112}"
    ;;
  rlhf_teacher)
    NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-16}"
    NUM_GEN_NODES="${NUM_GEN_NODES:-16}"
    ;;
  reasoning_teacher)
    NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-16}"
    NUM_GEN_NODES="${NUM_GEN_NODES:-44}"
    ;;
  vision_teacher)
    NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-8}"
    NUM_GEN_NODES="${NUM_GEN_NODES:-8}"
    ;;
  mopd)
    NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-64}"
    NUM_GEN_NODES="${NUM_GEN_NODES:-30}"
    # general: 2, RLHF: 4, reasoning: 8, vision: 4; see mopd.yaml.
    NUM_TEACHER_NODES="${NUM_TEACHER_NODES:-18}"
    : "${RLHF_TEACHER_PATH:?Set RLHF_TEACHER_PATH}"
    : "${REASONING_TEACHER_PATH:?Set REASONING_TEACHER_PATH}"
    : "${VISION_TEACHER_PATH:?Set VISION_TEACHER_PATH}"
    ;;
  -h|--help|*)
    echo "Usage: bash $0 {student_rlvr|rlhf_teacher|reasoning_teacher|vision_teacher|mopd} [key=value ...]"
    [[ "${STAGE}" == -h || "${STAGE}" == --help ]] && exit 0
    exit 2
    ;;
esac
shift
NUM_TEACHER_NODES="${NUM_TEACHER_NODES:-0}"
for name in EXP_NAME TRAIN_PATH CONTAINER SHARED_ROOT SLURM_ACCOUNT SLURM_PARTITION; do
  : "${!name:?Set ${name}}"
done
if [[ "${STAGE}" == student_rlvr ]]; then
  : "${MODEL_PATH:?Set MODEL_PATH to the SFT checkpoint}"
else
  : "${RLVR_CHECKPOINT:?Set RLVR_CHECKPOINT to the Student RLVR checkpoint}"
fi
case "${STAGE}" in
  student_rlvr|reasoning_teacher|mopd)
    : "${SANDBOX_CONTAINER:?Set SANDBOX_CONTAINER}" "${DATA_ROOT:?Set DATA_ROOT}"
    ;;
esac

export GPUS_PER_NODE=4
export RESULTS_DIR="${RESULTS_DIR:-${SHARED_ROOT}/results/${EXP_NAME}}"
CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints"
RUN_DIR="${RESULTS_DIR}/runs/$(date +%Y%m%d-%H%M%S-%N)"
NUM_RAY_NODES=$((NUM_TRAIN_NODES + NUM_GEN_NODES + NUM_TEACHER_NODES))
export MOUNTS="${SHARED_ROOT}:${SHARED_ROOT},${PROJECT_ROOT}/examples:/opt/nemo-rl/examples${EXTRA_MOUNTS:+,${EXTRA_MOUNTS}}"
export DEDICATED_RAY_HEAD=0
export BASE_LOG_DIR="${RESULTS_DIR}/ray_logs"
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-60}"
export NRL_WG_USE_RAY_REF="${NRL_WG_USE_RAY_REF:-1}"
export NRL_VLLM_USE_V1="${NRL_VLLM_USE_V1:-1}"
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS="${NRL_VLLM_ASYNC_TIMEOUT_SECONDS:-3600}"
export HF_HOME="${HF_HOME:-${SHARED_ROOT}/hf_cache}"
export HF_MODULES_CACHE="${RUN_DIR}/hf_modules"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT VLLM_ATTENTION_BACKEND
if [[ -n "${SANDBOX_CONTAINER:-}" ]]; then
  export SANDBOX_COMMAND="${SANDBOX_COMMAND:-/start-with-nginx.sh}"
  export SANDBOX_EXTRA_MOUNTS="${SHARED_ROOT}:${SHARED_ROOT}${SANDBOX_EXTRA_MOUNTS:+,${SANDBOX_EXTRA_MOUNTS}}"
fi

export EXTERNAL_VLLM_TOOLS_DIR_HOST="${PROJECT_ROOT}/tools/external_gym_vllm"
export EXTERNAL_VLLM_SHARED_ROOT="${SHARED_ROOT}"
export RAY_SUB="${PROJECT_ROOT}/ray.sub"
source "${EXTERNAL_VLLM_TOOLS_DIR_HOST}/pool_config.sh"
export EXTERNAL_VLLM_POOLS=""
export EXTERNAL_VLLM_NUM_NODES=0
JUDGE_OVERRIDES=()
source "${SCRIPT_DIR}/judge_pools.sh"
NUM_EXTERNAL_SERVICE_NODES="${EXTERNAL_VLLM_NUM_NODES}"
export NUM_EXTERNAL_SERVICE_NODES

# Reference YAML is loaded by training. Caller overrides are last, as in Ultra.
printf -v TRAIN_CMD '%q ' uv run --no-sync examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "examples/nemo_gym/nemotron-3.5-super/${STAGE}.yaml" \
  "cluster.num_nodes=${NUM_RAY_NODES}" \
  "policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES}" \
  "checkpointing.checkpoint_dir=${CHECKPOINT_DIR}" \
  "logger.log_dir=${RUN_DIR}/logs" \
  "${JUDGE_OVERRIDES[@]}" "$@"
export COMMAND="cd /opt/nemo-rl && ${TRAIN_CMD% }"
BATCH_SCRIPT="${RAY_SUB}"
if (( NUM_EXTERNAL_SERVICE_NODES > 0 )); then
  validate_external_vllm_submission "${COMMAND}" "${NUM_EXTERNAL_SERVICE_NODES}"
  BATCH_SCRIPT="${EXTERNAL_VLLM_TOOLS_DIR_HOST}/run_in_allocation.sh"
fi

RESOURCE_ARGS=(--account="${SLURM_ACCOUNT}" --partition="${SLURM_PARTITION}"
  --time="${WALLTIME:-4:00:00}" --gres="gpu:${GPUS_PER_NODE}" --exclusive --mem=0 --export=ALL)
for option in SLURM_QOS:qos SLURM_RESERVATION:reservation EXCLUDE_NODES:exclude SLURM_COMMENT:comment; do
  variable="${option%%:*}"
  [[ -z "${!variable:-}" ]] || RESOURCE_ARGS+=("--${option#*:}=${!variable}")
done
SBATCH_ARGS=(--parsable --nodes="${NUM_RAY_NODES}" "${RESOURCE_ARGS[@]}"
  --job-name="${EXP_NAME}" --dependency=singleton
  --output="${RUN_DIR}/slurm-%j.out" --error="${RUN_DIR}/slurm-%j.err")
[[ -z "${SLURM_SEGMENT_SIZE:-}" ]] || SBATCH_ARGS+=(--segment="${SLURM_SEGMENT_SIZE}")
if (( NUM_EXTERNAL_SERVICE_NODES > 0 )); then
  SBATCH_ARGS+=(: --nodes="${NUM_EXTERNAL_SERVICE_NODES}" "${RESOURCE_ARGS[@]}")
  [[ -z "${EXTERNAL_VLLM_SEGMENT_SIZE:-}" ]] || SBATCH_ARGS+=(--segment="${EXTERNAL_VLLM_SEGMENT_SIZE}")
fi
SBATCH_ARGS+=("${BATCH_SCRIPT}")

echo "${STAGE}: ${NUM_TRAIN_NODES} training + ${NUM_GEN_NODES} generation + ${NUM_TEACHER_NODES} teacher nodes"
echo "External vLLM judge nodes: ${NUM_EXTERNAL_SERVICE_NODES}; allocation total: $((NUM_RAY_NODES + NUM_EXTERNAL_SERVICE_NODES))"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Driver: ${COMMAND}"
printf 'Submit: '; printf '%q ' sbatch "${SBATCH_ARGS[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "DRY_RUN=1: no files written and no job submitted."
  exit 0
fi
mkdir -p -- "${RESULTS_DIR}/runs"
mkdir -- "${RUN_DIR}" "${HF_MODULES_CACHE}"
JOB_ID="$(sbatch "${SBATCH_ARGS[@]}")"
JOB_ID="${JOB_ID%%;*}"
echo "Submitted ${JOB_ID}"
echo "Ray driver log: ${BASE_LOG_DIR}/${JOB_ID}-logs/ray-driver.log"
