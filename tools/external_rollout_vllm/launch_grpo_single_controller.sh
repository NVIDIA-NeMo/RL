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

# Nemotron 3.5 Nano RLVR SingleController smoke. Policy rollout, GenRM, and
# NL2Bash run as external vLLM services; the training Ray cluster owns only the
# Megatron policy and the Gym-managed safety judge.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
EXTERNAL_VLLM_TOOLS_DIR_HOST="${PROJECT_ROOT}/tools/external_gym_vllm"

export EXP_NAME="${EXP_NAME:-nano35-external-vllm-pd2x2-smoke}"
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/guyueh/container_images/nvidian+nemo-rl+nightly.sqsh}"
export VLLM_CONTAINER="${VLLM_CONTAINER:-${CONTAINER}}"
export ROLLOUT_VLLM_CONTAINER="${ROLLOUT_VLLM_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/guyueh/container_images/vllm+vllm-openai+v0.29.0-aarch64.sqsh}"
export EXTERNAL_VLLM_SHARED_ROOT="${EXTERNAL_VLLM_SHARED_ROOT:-/lustre}"
export ACCOUNT="${ACCOUNT:-nemotron_sw_post}"
export PARTITION="${PARTITION:-batch}"
export QOS="${QOS:-normal}"

MINE="${MINE:-/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/${USER:-$(id -un)}}"
PERSISTENT_CACHE="${PERSISTENT_CACHE:-${MINE}/cache}"
export HF_HOME="${HF_HOME:-${PERSISTENT_CACHE}/hf_home}"
export BASE_LOG_DIR="${BASE_LOG_DIR:-${MINE}/runs/${EXP_NAME}}"
export EXTERNAL_ROLLOUT_HF_EXPORT_DIR="${EXTERNAL_ROLLOUT_HF_EXPORT_DIR:-${BASE_LOG_DIR}/hf_exports}"

# This experiment deliberately uses the checkout-mounted Gym with the existing
# persistent, prebuilt virtual environments rather than the image's Gym copy.
export USE_IMAGE_GYM=0
export GYM_VENV_DIR="${GYM_VENV_DIR:-/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/guyueh/cache/gym_venvs}"

export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/venkats/nemo-evaluator-rundirs/nano_v35_sft/conversions/upsampled-iter6000/hf}"
export TRAIN_PATH="${TRAIN_PATH:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/geshen/rl-data-tools/blends/curriculum_amplified_dolphin_v41_cheery_umbrette.train.jsonl}"
export VAL_PATH="${VAL_PATH:-${TRAIN_PATH}}"
export GENRM_MODEL="${GENRM_MODEL:-/lustre/fsw/portfolios/coreai/users/amahishi/projects/nemo-rl-workspace/ultra-smoke-test/cache/hf_judge_models/hub/models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM/snapshots/116af7cb1a23ce9017b2c412945b0623252655d2}"
export GENRM_REASONING_PARSER="${GENRM_REASONING_PARSER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/guyueh/hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/snapshots/77df655d5e9f8362164ed14dd8b48f8bce657498/ultra_v3_reasoning_parser.py}"
export NL2BASH_JUDGE_MODEL="${NL2BASH_JUDGE_MODEL:-/lustre/fsw/portfolios/coreai/users/amahishi/projects/nemo-rl-workspace/ultra-smoke-test/cache/hf_judge_models/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969}"
export SAFETY_JUDGE_MODEL="${SAFETY_JUDGE_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/super_v3/model_checkpoints/Nemotron-Content-Safety-Reasoning-4B}"

export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/geshen/mopd_nano_fast/images/nemo-skills-sandbox-no-sync.sqsh}"
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-/start-with-nginx.sh}"
export NEMO_SKILLS_SANDBOX_PORT="${NEMO_SKILLS_SANDBOX_PORT:-6000}"
# The sandbox image otherwise starts one stateful worker per CPU. That is far
# beyond this smoke's 32-prompt in-flight limit and creates hundreds of nginx
# upstreams per node. Keep enough capacity for the smoke while making sandbox
# startup and hostname resolution substantially lighter.
export SANDBOX_ENV_VARS="${SANDBOX_ENV_VARS:-NUM_WORKERS=16}"

export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export TRAIN_NODES="${TRAIN_NODES:-4}"
export GYM_NODES="${GYM_NODES:-1}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-2}"
RAY_NODES=$((TRAIN_NODES + GYM_NODES))

ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-4}"
ROLLOUT_DATA_PARALLEL_SIZE="${ROLLOUT_DATA_PARALLEL_SIZE:-1}"
ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-4}"
ROLLOUT_PREFILL_REPLICAS="${ROLLOUT_PREFILL_REPLICAS:-2}"
GENRM_REPLICAS="${GENRM_REPLICAS:-1}"
GENRM_TENSOR_PARALLEL_SIZE="${GENRM_TENSOR_PARALLEL_SIZE:-8}"
NL2BASH_REPLICAS="${NL2BASH_REPLICAS:-2}"
NL2BASH_TENSOR_PARALLEL_SIZE="${NL2BASH_TENSOR_PARALLEL_SIZE:-4}"
VLLM_PYTHON="${VLLM_PYTHON:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python}"
ROLLOUT_VLLM_LAUNCH_MODE="${ROLLOUT_VLLM_LAUNCH_MODE:-native}"
ROLLOUT_VLLM_EXECUTABLE="${ROLLOUT_VLLM_EXECUTABLE:-/usr/local/bin/vllm}"
ROLLOUT_FRONTEND="${ROLLOUT_FRONTEND:-python-proxy}"
case "${ROLLOUT_FRONTEND}" in
  python-proxy)
    BATCH_SCRIPT="${EXTERNAL_VLLM_TOOLS_DIR_HOST}/run_in_allocation.sh"
    ;;
  vllm-router)
    BATCH_SCRIPT="${EXTERNAL_VLLM_TOOLS_DIR_HOST}/run_in_allocation_vllm_router.sh"
    ;;
  *)
    echo "ERROR: ROLLOUT_FRONTEND must be python-proxy or vllm-router" >&2
    exit 1
    ;;
esac

# The Gym-inspired Rust router is packaged separately from both the NeMo-RL
# and vLLM containers. The deployment supplies either its site-packages
# directory or a wheel; the downstream launcher rejects setting both.
if [[ "${ROLLOUT_FRONTEND}" == "vllm-router" ]] && \
  [[ -z "${VLLM_ROUTER_SITE_PACKAGES:-}" && -z "${VLLM_ROUTER_WHEEL:-}" ]]; then
  echo "ERROR: vllm-router requires VLLM_ROUTER_SITE_PACKAGES or VLLM_ROUTER_WHEEL" >&2
  exit 1
fi

CONFIG="${CONFIG:-${PROJECT_ROOT}/examples/nemo_gym/nemotron-3.5-nano/rlvr_sc_smoke_small_external_vllm.yaml}"
TIME_LIMIT="${TIME_LIMIT:-1:30:00}"
JOB_NAME="${JOB_NAME:-${EXP_NAME}}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"
NRL_MAX_STEPS="${NRL_MAX_STEPS:-2}"
ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-32768}"
WANDB_PROJECT="${WANDB_PROJECT:-nano35-rlvr-main-tot-smoke}"
WANDB_NAME="${WANDB_NAME:-${EXP_NAME}}"
if [[ -n "${WANDB_API_KEY:-}" || "${WANDB_MODE:-}" =~ ^(offline|dryrun|disabled)$ ]]; then
  WANDB_ENABLED="${WANDB_ENABLED:-True}"
else
  WANDB_ENABLED=False
fi

for required_file in "${CONFIG}" "${PROJECT_ROOT}/ray.sub"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: required file does not exist: ${required_file}" >&2
    exit 1
  fi
done
for variable_name in GPUS_PER_NODE TRAIN_NODES GYM_NODES SEGMENT_SIZE ROLLOUT_TENSOR_PARALLEL_SIZE ROLLOUT_DATA_PARALLEL_SIZE ROLLOUT_REPLICAS GENRM_REPLICAS GENRM_TENSOR_PARALLEL_SIZE NL2BASH_REPLICAS NL2BASH_TENSOR_PARALLEL_SIZE ROLLOUT_MAX_NEW_TOKENS; do
  if [[ ! "${!variable_name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${variable_name} must be a positive integer" >&2
    exit 1
  fi
done
if [[ ! "${ROLLOUT_PREFILL_REPLICAS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: ROLLOUT_PREFILL_REPLICAS must be a nonnegative integer" >&2
  exit 1
fi
for tensor_parallel_size in "${ROLLOUT_TENSOR_PARALLEL_SIZE}" "${GENRM_TENSOR_PARALLEL_SIZE}" "${NL2BASH_TENSOR_PARALLEL_SIZE}"; do
  if (( tensor_parallel_size % GPUS_PER_NODE != 0 )); then
    echo "ERROR: every external tensor-parallel size must be divisible by GPUS_PER_NODE=${GPUS_PER_NODE}" >&2
    exit 1
  fi
done

mkdir -p "${BASE_LOG_DIR}" "${EXTERNAL_ROLLOUT_HF_EXPORT_DIR}"

# /lustre contains the checkout, Gym source, model inputs, venvs, and outputs.
# Keep any caller-supplied extra mounts while guaranteeing the shared root.
MOUNTS="${MOUNTS:-${EXTERNAL_VLLM_SHARED_ROOT}:${EXTERNAL_VLLM_SHARED_ROOT}}"
append_mount() {
  local mount="$1"
  [[ ",${MOUNTS}," == *",${mount},"* ]] || MOUNTS="${MOUNTS:+${MOUNTS},}${mount}"
}
append_mount "${EXTERNAL_VLLM_SHARED_ROOT}:${EXTERNAL_VLLM_SHARED_ROOT}"
if [[ "${USE_IMAGE_GYM}" == "0" ]]; then
  GYM_ROOT="${PROJECT_ROOT}/3rdparty/Gym-workspace/Gym"
  [[ -d "${GYM_ROOT}" ]] || { echo "ERROR: Gym checkout is missing: ${GYM_ROOT}" >&2; exit 1; }
  append_mount "${GYM_ROOT}:${GYM_ROOT}"
fi

# The amplified-dolphin Gym blend includes ether0, whose rdkit dependency needs
# these host X11 sonames. The training container intentionally remains unchanged.
X11_LIB_DIR="${X11_LIB_DIR:-/usr/lib/aarch64-linux-gnu}"
for library in libX11.so.6 libXau.so.6 libXdmcp.so.6 libXext.so.6 libXrender.so.1 libxcb.so.1; do
  source_path=$(realpath "${X11_LIB_DIR}/${library}" 2>/dev/null || true)
  if [[ -z "${source_path}" || ! -f "${source_path}" ]]; then
    echo "ERROR: ${library} is required by the ether0 Gym server but was not found under ${X11_LIB_DIR}" >&2
    exit 1
  fi
  append_mount "${source_path}:${X11_LIB_DIR}/${library}"
done
export MOUNTS

ROLLOUT_BASE_URL=__ROLLOUT_BASE_URL__
ROLLOUT_CONTROL_BASE_URL=__ROLLOUT_CONTROL_BASE_URL__
GENRM_BASE_URL=__GENRM_BASE_URL__
NL2BASH_BASE_URL=__NL2BASH_BASE_URL__
ROLLOUT_REASONING_PARSER_PLUGIN="${PROJECT_ROOT}/nemo_rl/models/generation/vllm/reasoning_parsers/nano_v3_reasoning_parser.py"

COMMAND="cd ${PROJECT_ROOT} && OMP_NUM_THREADS=16 NEMO_GYM_VENV_DIR=${GYM_VENV_DIR} HF_HOME=${HF_HOME} RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 UV_HTTP_TIMEOUT=300 NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800 NRL_WG_USE_RAY_REF=1 uv run examples/run_grpo_external_vllm_single_controller.py --config ${CONFIG} policy.model_name=${MODEL_PATH} policy.generation.max_new_tokens=${ROLLOUT_MAX_NEW_TOKENS} cluster.num_nodes=${TRAIN_NODES} cluster.gpus_per_node=${GPUS_PER_NODE} cluster.segment_size=${SEGMENT_SIZE} env.nemo_gym.num_gpu_nodes=${GYM_NODES} data.train.data_path=${TRAIN_PATH} data.validation.data_path=${VAL_PATH} policy.generation.remote_vllm_cfg.base_url=${ROLLOUT_BASE_URL} ++env.nemo_gym.genrm_model.responses_api_models.genrm_model.base_url=${GENRM_BASE_URL} ++env.nemo_gym.genrm_model.responses_api_models.genrm_model.model=model ++env.nemo_gym.nl2bash_judge_model.responses_api_models.local_vllm_model.base_url=${NL2BASH_BASE_URL} ++env.nemo_gym.nl2bash_judge_model.responses_api_models.local_vllm_model.model=model env.nemo_gym.safety_judge_model.responses_api_models.local_vllm_model.model=${SAFETY_JUDGE_MODEL} env.nemo_gym.nemo_gym_log_dir=${BASE_LOG_DIR}/nemo_gym checkpointing.checkpoint_dir=${BASE_LOG_DIR} logger.log_dir=${BASE_LOG_DIR} logger.wandb_enabled=${WANDB_ENABLED} logger.wandb.project=${WANDB_PROJECT} logger.wandb.name=${WANDB_NAME} grpo.max_num_steps=${NRL_MAX_STEPS}"
ROLLOUT_CONTROL_ARGS=()
if [[ "${ROLLOUT_FRONTEND}" == "vllm-router" ]]; then
  COMMAND+=" ++policy.generation.remote_vllm_cfg.control_base_url=${ROLLOUT_CONTROL_BASE_URL}"
  ROLLOUT_CONTROL_ARGS=(
    --control-lb-port 9211
    --control-url-placeholder "${ROLLOUT_CONTROL_BASE_URL}"
  )
fi

source "${EXTERNAL_VLLM_TOOLS_DIR_HOST}/pool_config.sh"
EXTERNAL_VLLM_POOLS=""
set +u
register_external_vllm_pool ROLLOUT \
  --display-name "Mutable policy rollout" \
  --model "${MODEL_PATH}" \
  --container "${ROLLOUT_VLLM_CONTAINER}" \
  --python "${VLLM_PYTHON}" \
  --launch-mode "${ROLLOUT_VLLM_LAUNCH_MODE}" \
  --vllm-executable "${ROLLOUT_VLLM_EXECUTABLE}" \
  --replicas "${ROLLOUT_REPLICAS}" \
  --tensor-parallel-size "${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
  --data-parallel-size "${ROLLOUT_DATA_PARALLEL_SIZE}" \
  --prefill-replicas "${ROLLOUT_PREFILL_REPLICAS}" \
  --lb-port 9210 \
  --vllm-port 8000 \
  --served-model-name policy \
  --url-placeholder "${ROLLOUT_BASE_URL}" \
  "${ROLLOUT_CONTROL_ARGS[@]}" \
  --startup-timeout "${STARTUP_TIMEOUT}" \
  --shared-path "${ROLLOUT_REASONING_PARSER_PLUGIN}"
external_vllm_pool_env ROLLOUT \
  VLLM_SERVER_DEV_MODE=1 \
  VLLM_HTTP_TIMEOUT_KEEP_ALIVE=180 \
  VLLM_SSM_CONV_STATE_LAYOUT=DS \
  UCX_MODULE_DIR=/usr/local/lib/python3.12/dist-packages/nixl_cu13.libs/ucx \
  UCX_TLS=all
external_vllm_pool_args ROLLOUT \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 73728 \
  --max-num-seqs 768 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nano_v3 \
  --reasoning-parser-plugin "${ROLLOUT_REASONING_PARSER_PLUGIN}" \
  --attention-backend FLASH_ATTN \
  --moe-backend flashinfer_cutlass \
  --enable-expert-parallel \
  --mamba-ssm-cache-dtype float32 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,192,256,384,512,768],"cudagraph_mode":"PIECEWISE","pass_config":{"fuse_allreduce_rms":false}}'

register_external_vllm_pool GENRM \
  --display-name GenRM \
  --model "${GENRM_MODEL}" \
  --container "${VLLM_CONTAINER}" \
  --python "${VLLM_PYTHON}" \
  --replicas "${GENRM_REPLICAS}" \
  --tensor-parallel-size "${GENRM_TENSOR_PARALLEL_SIZE}" \
  --lb-port 9213 \
  --vllm-port 8000 \
  --served-model-name model \
  --url-placeholder "${GENRM_BASE_URL}" \
  --startup-timeout "${STARTUP_TIMEOUT}" \
  --shared-path "${GENRM_REASONING_PARSER}"
external_vllm_pool_env GENRM \
  FLASHINFER_WORKSPACE_BASE=/tmp \
  VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl \
  VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  NCCL_MNNVL_ENABLE=1
external_vllm_pool_args GENRM \
  --trust-remote-code \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --reasoning-parser ultra_v3 \
  --reasoning-parser-plugin "${GENRM_REASONING_PARSER}" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}' \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":96}' \
  --enable-expert-parallel

register_external_vllm_pool NL2BASH \
  --display-name NL2Bash \
  --model "${NL2BASH_JUDGE_MODEL}" \
  --container "${VLLM_CONTAINER}" \
  --python "${VLLM_PYTHON}" \
  --replicas "${NL2BASH_REPLICAS}" \
  --tensor-parallel-size "${NL2BASH_TENSOR_PARALLEL_SIZE}" \
  --lb-port 9214 \
  --vllm-port 8000 \
  --served-model-name model \
  --url-placeholder "${NL2BASH_BASE_URL}" \
  --startup-timeout "${STARTUP_TIMEOUT}"
external_vllm_pool_env NL2BASH \
  FLASHINFER_WORKSPACE_BASE=/tmp \
  VLLM_USE_FLASHINFER_MOE_FP16=0 \
  VLLM_USE_FLASHINFER_MOE_FP8=0 \
  VLLM_USE_DEEP_GEMM=0 \
  VLLM_MOE_USE_DEEP_GEMM=0 \
  NCCL_MNNVL_ENABLE=1
external_vllm_pool_args NL2BASH \
  --dtype bfloat16 \
  --pipeline-parallel-size 1 \
  --max-model-len 131072 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --attention-backend TRITON_ATTN \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256]}' \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":112}' \
  --enable-expert-parallel
set -u

NUM_EXTERNAL_SERVICE_NODES="${EXTERNAL_VLLM_NUM_NODES}"
export PROJECT_ROOT EXTERNAL_VLLM_TOOLS_DIR_HOST EXTERNAL_VLLM_SHARED_ROOT
export EXTERNAL_ROLLOUT_HF_EXPORT_DIR GPUS_PER_NODE BASE_LOG_DIR CONTAINER MOUNTS COMMAND
export RAY_SUB="${RAY_SUB:-${PROJECT_ROOT}/ray.sub}"

validate_external_vllm_submission "${COMMAND}" "${NUM_EXTERNAL_SERVICE_NODES}"

echo "Submitting Nano 3.5 RLVR external-rollout smoke:"
echo "  Ray cluster: ${RAY_NODES} nodes (${TRAIN_NODES} training + ${GYM_NODES} Gym)"
echo "  Ray segment: ${SEGMENT_SIZE} node(s)"
echo "  rollout:     ${ROLLOUT_PREFILL_REPLICAS} prefill + $((ROLLOUT_REPLICAS - ROLLOUT_PREFILL_REPLICAS)) decode, TP=${ROLLOUT_TENSOR_PARALLEL_SIZE}"
echo "  GenRM:       ${GENRM_REPLICAS} replica(s), TP=${GENRM_TENSOR_PARALLEL_SIZE}"
echo "  NL2Bash:     ${NL2BASH_REPLICAS} replica(s), TP=${NL2BASH_TENSOR_PARALLEL_SIZE}"
echo "  external:    ${NUM_EXTERNAL_SERVICE_NODES} nodes total"
echo "  Gym source:  checkout (USE_IMAGE_GYM=${USE_IMAGE_GYM})"
echo "  Gym venvs:   ${GYM_VENV_DIR}"
echo "  Ray image:   ${CONTAINER}"
echo "  rollout image/mode: ${ROLLOUT_VLLM_CONTAINER} (${ROLLOUT_VLLM_LAUNCH_MODE})"
echo "  rollout frontend:   ${ROLLOUT_FRONTEND}"
echo "  other vLLM image:   ${VLLM_CONTAINER} (nemo-rl-ray)"
echo "  config:      ${CONFIG}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; submission skipped"
  echo "${COMMAND}"
  exit 0
fi

SLURM_COMMENT="${SLURM_COMMENT:-{\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"120\",\"reason\":\"other\",\"description\":\"External rollout model loading and checkpoint refit can leave GPUs idle\"}}}"

sbatch \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --job-name="${JOB_NAME}" \
  --nodes="${RAY_NODES}" \
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
  --job-name="${JOB_NAME}-services" \
  --nodes="${NUM_EXTERNAL_SERVICE_NODES}" \
  --exclusive \
  --mem=0 \
  --gres="gpu:${GPUS_PER_NODE}" \
  --time="${TIME_LIMIT}" \
  "${BATCH_SCRIPT}"
