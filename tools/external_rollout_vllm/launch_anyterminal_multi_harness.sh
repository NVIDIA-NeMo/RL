#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch the four-harness AnyTerminal recipe with 16 TP1 vLLM servers packed
# four per GPU node. Generation traffic uses consistent-hash routing so an
# agent's successive turns keep hitting the same prefix cache. Control traffic
# fans pause/refit/resume requests out to every backend.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
EXTERNAL_VLLM_TOOLS_DIR_HOST="${PROJECT_ROOT}/tools/external_gym_vllm"

: "${MODEL_PATH:?Set MODEL_PATH to the shared policy checkpoint}"
: "${TRAIN_PATH:?Set TRAIN_PATH to the Terminal-Bench multi-harness JSONL}"
: "${CONTAINER:?Set CONTAINER to the NeMo RL Enroot image}"
: "${ROLLOUT_VLLM_CONTAINER:?Set ROLLOUT_VLLM_CONTAINER to the vLLM Enroot image}"
: "${PREFIX_PLUGIN_WHEEL:?Set PREFIX_PLUGIN_WHEEL to the shared NeMo RL prefix-plugin wheel}"
: "${VLLM_ROUTER_WHEEL:?Set VLLM_ROUTER_WHEEL to the shared vllm-router wheel}"

export EXP_NAME="${EXP_NAME:-anyterminal-multi-harness-external-vllm}"
export EXTERNAL_VLLM_SHARED_ROOT="${EXTERNAL_VLLM_SHARED_ROOT:-/lustre}"
export ACCOUNT="${ACCOUNT:-nemotron_sw_post}"
export PARTITION="${PARTITION:-batch}"
export QOS="${QOS:-normal}"

RUN_ROOT="${RUN_ROOT:-${EXTERNAL_VLLM_SHARED_ROOT}/fsw/portfolios/nemotron/users/${USER:-$(id -un)}/runs}"
export BASE_LOG_DIR="${BASE_LOG_DIR:-${RUN_ROOT}/${EXP_NAME}}"
export HF_HOME="${HF_HOME:-${RUN_ROOT}/cache/huggingface}"
export EXTERNAL_ROLLOUT_HF_EXPORT_DIR="${EXTERNAL_ROLLOUT_HF_EXPORT_DIR:-${BASE_LOG_DIR}/hf_exports}"
export NEMO_GYM_ROOT="${NEMO_GYM_ROOT:-${PROJECT_ROOT}/3rdparty/Gym-workspace/Gym}"
export GYM_VENV_DIR="${GYM_VENV_DIR:-${RUN_ROOT}/cache/gym_venvs}"

export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export TRAIN_NODES="${TRAIN_NODES:-8}"
export RAY_NODES="${TRAIN_NODES}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-2}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"
export ROLLOUT_REPLICAS_PER_NODE="${ROLLOUT_REPLICAS_PER_NODE:-4}"
export ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-1}"
export ROLLOUT_FRONTEND=vllm-router
export VLLM_ROUTER_POLICY="${VLLM_ROUTER_POLICY:-consistent_hash}"
export VLLM_ROUTER_REQUEST_TIMEOUT_S="${VLLM_ROUTER_REQUEST_TIMEOUT_S:-86400}"

CONFIG="${CONFIG:-${PROJECT_ROOT}/examples/nemo_gym/nemotron-3.5-nano/swe_anyterminal_multi_harness_external_vllm.yaml}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
JOB_NAME="${JOB_NAME:-${EXP_NAME}}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"
WANDB_ENTITY="${WANDB_ENTITY:-adlr}"
WANDB_PROJECT="${WANDB_PROJECT:-multi-harness-RL}"
WANDB_NAME="${WANDB_NAME:-${EXP_NAME}}"
ROLLOUT_VLLM_EXECUTABLE="${ROLLOUT_VLLM_EXECUTABLE:-/usr/local/bin/vllm}"
ROLLOUT_REASONING_PARSER_PLUGIN="${PROJECT_ROOT}/nemo_rl/models/generation/vllm/reasoning_parsers/nano_v3_reasoning_parser.py"

for required_file in "${CONFIG}" "${PROJECT_ROOT}/ray.sub" "${PREFIX_PLUGIN_WHEEL}" "${VLLM_ROUTER_WHEEL}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: required file does not exist: ${required_file}" >&2
    exit 1
  fi
done
if [[ ! -d "${NEMO_GYM_ROOT}" ]]; then
  echo "ERROR: NeMo Gym checkout does not exist: ${NEMO_GYM_ROOT}" >&2
  exit 1
fi
for variable_name in GPUS_PER_NODE TRAIN_NODES SEGMENT_SIZE ROLLOUT_REPLICAS ROLLOUT_REPLICAS_PER_NODE ROLLOUT_TENSOR_PARALLEL_SIZE; do
  if [[ ! "${!variable_name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${variable_name} must be a positive integer" >&2
    exit 1
  fi
done
if (( ROLLOUT_REPLICAS % ROLLOUT_REPLICAS_PER_NODE != 0 )); then
  echo "ERROR: ROLLOUT_REPLICAS must be divisible by ROLLOUT_REPLICAS_PER_NODE" >&2
  exit 1
fi
if (( ROLLOUT_TENSOR_PARALLEL_SIZE * ROLLOUT_REPLICAS_PER_NODE > GPUS_PER_NODE )); then
  echo "ERROR: packed rollout replicas require more than GPUS_PER_NODE=${GPUS_PER_NODE}" >&2
  exit 1
fi

mkdir -p "${BASE_LOG_DIR}" "${EXTERNAL_ROLLOUT_HF_EXPORT_DIR}"
MOUNTS="${MOUNTS:-${EXTERNAL_VLLM_SHARED_ROOT}:${EXTERNAL_VLLM_SHARED_ROOT}}"
append_mount() {
  local mount="$1"
  [[ ",${MOUNTS}," == *",${mount},"* ]] || MOUNTS="${MOUNTS:+${MOUNTS},}${mount}"
}
append_mount "${EXTERNAL_VLLM_SHARED_ROOT}:${EXTERNAL_VLLM_SHARED_ROOT}"
append_mount "${NEMO_GYM_ROOT}:${NEMO_GYM_ROOT}"
export MOUNTS

ROLLOUT_BASE_URL=__ROLLOUT_BASE_URL__
ROLLOUT_CONTROL_BASE_URL=__ROLLOUT_CONTROL_BASE_URL__
COMMAND="cd ${PROJECT_ROOT} && OMP_NUM_THREADS=16 NEMO_GYM_VENV_DIR=${GYM_VENV_DIR} HF_HOME=${HF_HOME} RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 UV_HTTP_TIMEOUT=300 NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800 NRL_WG_USE_RAY_REF=1 uv run examples/run_grpo_external_vllm_single_controller.py --config ${CONFIG} policy.model_name=${MODEL_PATH} cluster.num_nodes=${TRAIN_NODES} cluster.gpus_per_node=${GPUS_PER_NODE} cluster.segment_size=${SEGMENT_SIZE} data.train.data_path=${TRAIN_PATH} policy.generation.remote_vllm_cfg.base_url=${ROLLOUT_BASE_URL} ++policy.generation.remote_vllm_cfg.control_base_url=${ROLLOUT_CONTROL_BASE_URL} env.nemo_gym.nemo_gym_log_dir=${BASE_LOG_DIR}/nemo_gym checkpointing.checkpoint_dir=${BASE_LOG_DIR}/checkpoints logger.log_dir=${BASE_LOG_DIR} logger.wandb_enabled=true logger.wandb.entity=${WANDB_ENTITY} logger.wandb.project=${WANDB_PROJECT} logger.wandb.name=${WANDB_NAME}"

source "${EXTERNAL_VLLM_TOOLS_DIR_HOST}/pool_config.sh"
EXTERNAL_VLLM_POOLS=""
register_external_vllm_pool ROLLOUT \
  --display-name "AnyTerminal mutable policy rollout" \
  --model "${MODEL_PATH}" \
  --container "${ROLLOUT_VLLM_CONTAINER}" \
  --launch-mode native \
  --vllm-executable "${ROLLOUT_VLLM_EXECUTABLE}" \
  --replicas "${ROLLOUT_REPLICAS}" \
  --replicas-per-node "${ROLLOUT_REPLICAS_PER_NODE}" \
  --tensor-parallel-size "${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
  --lb-port 9210 \
  --control-lb-port 9211 \
  --vllm-port 8000 \
  --served-model-name policy \
  --url-placeholder "${ROLLOUT_BASE_URL}" \
  --control-url-placeholder "${ROLLOUT_CONTROL_BASE_URL}" \
  --startup-timeout "${STARTUP_TIMEOUT}" \
  --shared-path "${PREFIX_PLUGIN_WHEEL}" \
  --shared-path "${ROLLOUT_REASONING_PARSER_PLUGIN}"

external_vllm_pool_env ROLLOUT \
  "PYTHONPATH=${PREFIX_PLUGIN_WHEEL}" \
  VLLM_PLUGINS=nemo_rl_prefix_api \
  NEMO_RL_VLLM_PREFIX_PLUGIN_REQUIRED=1 \
  VLLM_SERVER_DEV_MODE=1 \
  VLLM_HTTP_TIMEOUT_KEEP_ALIVE=180 \
  VLLM_SSM_CONV_STATE_LAYOUT=DS
external_vllm_pool_args ROLLOUT \
  --trust-remote-code \
  --dtype bfloat16 \
  --return-tokens-as-token-ids \
  --max-model-len 196608 \
  --max-num-seqs 1024 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.85 \
  --async-scheduling \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nano_v3 \
  --reasoning-parser-plugin "${ROLLOUT_REASONING_PARSER_PLUGIN}" \
  --attention-backend FLASHINFER \
  --moe-backend flashinfer_cutlass \
  --enable-expert-parallel \
  --mamba-ssm-cache-dtype float32 \
  --mamba-cache-mode align \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,192,256,384,512],"cudagraph_mode":"PIECEWISE","pass_config":{"fuse_allreduce_rms":false}}' \
  --default-chat-template-kwargs '{"enable_thinking":true}'

NUM_EXTERNAL_SERVICE_NODES="${EXTERNAL_VLLM_NUM_NODES}"
export PROJECT_ROOT EXTERNAL_VLLM_TOOLS_DIR_HOST EXTERNAL_VLLM_SHARED_ROOT
export EXTERNAL_ROLLOUT_HF_EXPORT_DIR BASE_LOG_DIR CONTAINER MOUNTS COMMAND
export VLLM_ROUTER_WHEEL
export EXTERNAL_VLLM_ROUTER_POOL=ROLLOUT
export RAY_SUB="${RAY_SUB:-${PROJECT_ROOT}/ray.sub}"

validate_external_vllm_submission "${COMMAND}" "${NUM_EXTERNAL_SERVICE_NODES}"

echo "Submitting AnyTerminal multi-harness external-vLLM training:"
echo "  revisions:  RL=$(git -C "${PROJECT_ROOT}" rev-parse HEAD) Gym=$(git -C "${NEMO_GYM_ROOT}" rev-parse HEAD)"
echo "  training:   ${TRAIN_NODES} nodes x ${GPUS_PER_NODE} GPUs"
echo "  rollout:    ${ROLLOUT_REPLICAS} TP${ROLLOUT_TENSOR_PARALLEL_SIZE} endpoints on ${NUM_EXTERNAL_SERVICE_NODES} nodes"
echo "  routing:    ${VLLM_ROUTER_POLICY}"
echo "  batch:      32 source tasks x 4 harnesses x 16 generations = 2048"
echo "  context:    196608 tokens"
echo "  W&B:        ${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_NAME}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; submission skipped"
  echo "${COMMAND}"
  exit 0
fi

: "${OPENSANDBOX_DOMAIN:?Set OPENSANDBOX_DOMAIN for Kubernetes sandbox provisioning}"
: "${OPENSANDBOX_API_KEY:?Set OPENSANDBOX_API_KEY for Kubernetes sandbox provisioning}"

BATCH_SCRIPT="${EXTERNAL_VLLM_TOOLS_DIR_HOST}/run_in_allocation_vllm_router.sh"
SLURM_COMMENT="${SLURM_COMMENT:-{\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"120\",\"reason\":\"other\",\"description\":\"External rollout model loading and checkpoint refit\"}}}"
sbatch \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --job-name="${JOB_NAME}" \
  --nodes="${TRAIN_NODES}" \
  --exclusive \
  --mem=0 \
  --gres="gpu:${GPUS_PER_NODE}" \
  --time="${TIME_LIMIT}" \
  --comment="${SLURM_COMMENT}" \
  --export=ALL \
  : \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --job-name="${JOB_NAME}-rollout" \
  --nodes="${NUM_EXTERNAL_SERVICE_NODES}" \
  --exclusive \
  --mem=0 \
  --gres="gpu:${GPUS_PER_NODE}" \
  --time="${TIME_LIMIT}" \
  "${BATCH_SCRIPT}"
