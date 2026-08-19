#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Public SLURM launcher for the Nemotron 3 Ultra proof-generation and
# proof-verification recipes. It allocates one Ray component for training and
# rollout workers plus optional heterogeneous components for external proof
# judge servers.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=${SCRIPT_DIR}
cd "${PROJECT_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-examples/configs/grpo_proof_rl_64n.yaml}"
NRL_MODEL_PATH="${NRL_MODEL_PATH:-${MODEL_PATH:-}}"
NRL_TRAIN_PATH="${NRL_TRAIN_PATH:-${TRAIN_PATH:-}}"
NRL_VAL_PATH="${NRL_VAL_PATH:-${VAL_PATH:-${NRL_TRAIN_PATH}}}"
EXP_SUFFIX="${EXP_SUFFIX:-nemotron-3-ultra-proof}"
WANDB_NAME="${WANDB_NAME:-${EXP_SUFFIX}}"

: "${NRL_MODEL_PATH:?Set NRL_MODEL_PATH (or MODEL_PATH) to a Hugging Face model ID or mounted checkpoint path}"
: "${NRL_TRAIN_PATH:?Set NRL_TRAIN_PATH (or TRAIN_PATH) to the training JSONL path}"
: "${NRL_VAL_PATH:?Set NRL_VAL_PATH (or VAL_PATH) to the validation JSONL path}"
: "${CONTAINER:?Set CONTAINER to the NeMo RL container image or squashfs path}"
: "${PERSISTENT_CACHE:?Set PERSISTENT_CACHE to a shared cache directory}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT for your cluster}"
SLURM_PARTITION="${SLURM_PARTITION:-${PARTITION:-}}"
: "${SLURM_PARTITION:?Set SLURM_PARTITION for your cluster}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: CONFIG_PATH does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi

# Job shape. NUM_ACTOR_NODES is the Ray allocation and includes both training
# and rollout nodes. The convenience scripts select the published stage shapes.
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_WORKER="${CPUS_PER_WORKER:-144}"
NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-128}"
GENERATION_NUM_NODES="${GENERATION_NUM_NODES:-64}"
NUM_RAY_NODES="${NUM_RAY_NODES:-${NUM_ACTOR_NODES}}"
COLOCATED_INFERENCE="${COLOCATED_INFERENCE:-False}"

if (( NUM_ACTOR_NODES <= 0 || GENERATION_NUM_NODES <= 0 )); then
  echo "ERROR: NUM_ACTOR_NODES and GENERATION_NUM_NODES must be positive." >&2
  exit 1
fi
if (( GENERATION_NUM_NODES >= NUM_ACTOR_NODES )); then
  echo "ERROR: GENERATION_NUM_NODES must be smaller than NUM_ACTOR_NODES." >&2
  exit 1
fi
if (( NUM_RAY_NODES != NUM_ACTOR_NODES )); then
  echo "ERROR: NUM_RAY_NODES must equal NUM_ACTOR_NODES for this recipe." >&2
  exit 1
fi

# Published GB200 shapes allocate complete 16-node topology segments. Set
# SEGMENT_SIZE= explicitly on clusters whose sbatch does not support --segment.
SEGMENT_SIZE="${SEGMENT_SIZE-16}"
if [[ -n "${SEGMENT_SIZE}" ]]; then
  if ! [[ "${SEGMENT_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SEGMENT_SIZE must be a positive integer or empty." >&2
    exit 1
  fi
  if (( NUM_RAY_NODES < SEGMENT_SIZE || NUM_RAY_NODES % SEGMENT_SIZE != 0 )); then
    echo "ERROR: NUM_RAY_NODES=${NUM_RAY_NODES} must be a multiple of SEGMENT_SIZE=${SEGMENT_SIZE}." >&2
    exit 1
  fi
  TRAINING_NUM_NODES=$((NUM_ACTOR_NODES - GENERATION_NUM_NODES))
  if (( TRAINING_NUM_NODES < SEGMENT_SIZE || TRAINING_NUM_NODES % SEGMENT_SIZE != 0 )); then
    echo "ERROR: training nodes=${TRAINING_NUM_NODES} must be a multiple of SEGMENT_SIZE=${SEGMENT_SIZE}." >&2
    exit 1
  fi
fi

# External proof judges. Set USE_HET_SERVERS=0 to disable the additional
# components (the Gym resources then use their configured policy-model judge).
USE_HET_SERVERS="${USE_HET_SERVERS:-1}"
if [[ "${USE_HET_SERVERS}" == "1" ]]; then
  HET_SERVER_COUNT="${HET_SERVER_COUNT:-4}"
else
  HET_SERVER_COUNT=0
fi
HET_SERVER_NODES="${HET_SERVER_NODES:-2}"
HET_SERVER_GPUS_PER_NODE="${HET_SERVER_GPUS_PER_NODE:-${GPUS_PER_NODE}}"
PROOF_JUDGE_MODEL="${PROOF_JUDGE_MODEL:-deepseek-ai/DeepSeek-Math-V2}"
PROOF_JUDGE_PORT="${PROOF_JUDGE_PORT:-5000}"
PROOF_JUDGE_TP_SIZE="${PROOF_JUDGE_TP_SIZE:-$((HET_SERVER_NODES * HET_SERVER_GPUS_PER_NODE))}"

if (( HET_SERVER_COUNT > 0 )); then
  : "${HET_SERVER_CONTAINER:?Set HET_SERVER_CONTAINER to an image containing SGLang}"
  JUDGE_SERVER_ARGS="${JUDGE_SERVER_ARGS:-{\"server_type\":\"sglang\",\"port\":${PROOF_JUDGE_PORT},\"model\":\"${PROOF_JUDGE_MODEL}\",\"n_servers\":${HET_SERVER_COUNT}}}"
  if [[ -z "${HET_SERVER_COMMAND_TEMPLATE:-}" ]]; then
    read -r -d '' HET_SERVER_COMMAND_TEMPLATE <<'SERVER_COMMAND' || true
python3 -m sglang.launch_server \
  --model="${PROOF_JUDGE_MODEL}" \
  --served-model-name="${PROOF_JUDGE_MODEL}" \
  --host=0.0.0.0 \
  --port="${PROOF_JUDGE_PORT}" \
  --tensor-parallel-size="${PROOF_JUDGE_TP_SIZE}" \
  --nnodes="${HET_SERVER_NODES}" \
  --node-rank="${SLURM_PROCID}" \
  --dist-init-addr="${SLURM_MASTER_NODE}:20000" \
  --max-running-requests="${PROOF_JUDGE_MAX_RUNNING_REQUESTS:-256}" \
  --ep-size="${PROOF_JUDGE_TP_SIZE}" \
  --data-parallel-size="${PROOF_JUDGE_TP_SIZE}" \
  --enable-dp-attention \
  --reasoning-parser=deepseek-v3 \
  --mem-fraction-static="${PROOF_JUDGE_MEM_FRACTION:-0.8}"
SERVER_COMMAND
  fi
fi

export GPUS_PER_NODE CPUS_PER_WORKER
export HET_SERVER_COUNT HET_SERVER_NODES HET_SERVER_GPUS_PER_NODE
export HET_SERVER_CONTAINER="${HET_SERVER_CONTAINER:-}"
export HET_SERVER_COMMAND_TEMPLATE="${HET_SERVER_COMMAND_TEMPLATE:-}"
export JUDGE_SERVER_ARGS="${JUDGE_SERVER_ARGS:-}"
export PROOF_JUDGE_MODEL PROOF_JUDGE_PORT PROOF_JUDGE_TP_SIZE
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}"

# Outputs and caches must be on a filesystem visible to every allocated node.
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results/${EXP_SUFFIX}}"
mkdir -p "${RESULTS_DIR}" "${PERSISTENT_CACHE}"
RESULTS_DIR=$(cd "${RESULTS_DIR}" && pwd)
PERSISTENT_CACHE=$(cd "${PERSISTENT_CACHE}" && pwd)
HF_HOME="${HF_HOME:-${PERSISTENT_CACHE}/huggingface}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RESULTS_DIR}/checkpoints}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${RESULTS_DIR}/slurm}"
mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${SLURM_LOG_DIR}" "${HF_HOME}"
export HF_HOME

export BASE_LOG_DIR="${BASE_LOG_DIR:-${RESULTS_DIR}/ray_logs}"
export PROOF_JUDGE_LOG_JSONL_PATH="${PROOF_JUDGE_LOG_JSONL_PATH:-${LOG_DIR}/proof_judge.jsonl}"
export PROOF_VERIFICATION_LOG_JSONL_PATH="${PROOF_VERIFICATION_LOG_JSONL_PATH:-${LOG_DIR}/proof_verification.jsonl}"
export PROOF_GENSELECT_LOG_JSONL_PATH="${PROOF_GENSELECT_LOG_JSONL_PATH:-${LOG_DIR}/proof_genselect.jsonl}"
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS="${NRL_VLLM_ASYNC_TIMEOUT_SECONDS:-7200}"
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-60}"

NRL_VLLM_LOCAL_CACHE_DIR="${NRL_VLLM_LOCAL_CACHE_DIR:-/tmp/nemo_rl_vllm_cache}"
INDUCTOR_CACHE_DIR="${INDUCTOR_CACHE_DIR:-/tmp/nemo_rl_inductor_cache}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/nemo_rl_triton_cache}"
export NRL_VLLM_LOCAL_CACHE_DIR INDUCTOR_CACHE_DIR TRITON_CACHE_DIR

# Optional worktree overlays for development. Release containers built from
# this checkout need no overlays.
MOUNTS="${MOUNTS:-}"
_append_mount() {
  local src="$1" dst="$2"
  [[ -e "${src}" ]] || return 0
  if [[ -n "${MOUNTS}" ]]; then
    MOUNTS+=",${src}:${dst}"
  else
    MOUNTS="${src}:${dst}"
  fi
}

_append_mount "${RESULTS_DIR}" "${RESULTS_DIR}"
_append_mount "${PERSISTENT_CACHE}" "${PERSISTENT_CACHE}"
if [[ "${USE_WORKTREE:-0}" == "1" ]]; then
  _append_mount "${PROJECT_ROOT}/nemo_rl/algorithms/async_utils.py" "/opt/nemo-rl/nemo_rl/algorithms/async_utils.py"
  _append_mount "${PROJECT_ROOT}/nemo_rl/algorithms/grpo.py" "/opt/nemo-rl/nemo_rl/algorithms/grpo.py"
  _append_mount "${PROJECT_ROOT}/nemo_rl/algorithms/loss_functions.py" "/opt/nemo-rl/nemo_rl/algorithms/loss_functions.py"
  _append_mount "${PROJECT_ROOT}/nemo_rl/utils/checkpoint.py" "/opt/nemo-rl/nemo_rl/utils/checkpoint.py"
  _append_mount "${PROJECT_ROOT}/examples/configs/grpo_proof_rl_64n.yaml" "/opt/nemo-rl/examples/configs/grpo_proof_rl_64n.yaml"
  _append_mount "${PROJECT_ROOT}/examples/nemo_gym/run_grpo_nemo_gym.py" "/opt/nemo-rl/examples/nemo_gym/run_grpo_nemo_gym.py"
  for proof_resource in proof_judge proof_genselect proof_verification; do
    _append_mount \
      "${PROJECT_ROOT}/3rdparty/Gym-workspace/Gym/resources_servers/${proof_resource}" \
      "/opt/nemo-rl/3rdparty/Gym-workspace/Gym/resources_servers/${proof_resource}"
  done
fi
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
  if [[ -n "${MOUNTS}" ]]; then
    MOUNTS+=",${EXTRA_MOUNTS}"
  else
    MOUNTS="${EXTRA_MOUNTS}"
  fi
fi
export MOUNTS
# Judges only need the shared Hugging Face cache. Local judge checkpoints can
# be exposed explicitly with HET_SERVER_MOUNTS.
export HET_SERVER_MOUNTS="${HET_SERVER_MOUNTS:-${HF_HOME}:${HF_HOME}}"
export HET_SERVER_WORKDIR="${HET_SERVER_WORKDIR:-/tmp}"

# W&B and Hugging Face credentials are optional. Public models can be loaded
# without HF_TOKEN; logging is disabled when WANDB_API_KEY is absent.
WANDB_PROJ="${WANDB_PROJ:-nemotron-3-ultra-imo}"
WANDB_ENABLED=False
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
  WANDB_ENABLED=True
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  export WANDB_ENTITY
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

# Construct a shell-safe driver command. Every extra positional argument is a
# Hydra override and remains one argument after SLURM/container forwarding.
TRAIN_ARGS=(
  uv run ./examples/nemo_gym/run_grpo_nemo_gym.py
  --config "${CONFIG_PATH}"
  "policy.model_name=${NRL_MODEL_PATH}"
  "cluster.gpus_per_node=${GPUS_PER_NODE}"
  "cluster.num_nodes=${NUM_ACTOR_NODES}"
  "cluster.segment_size=${SEGMENT_SIZE:-null}"
  "policy.generation.colocated.enabled=${COLOCATED_INFERENCE}"
  "policy.generation.colocated.resources.num_nodes=${GENERATION_NUM_NODES}"
  "policy.generation.colocated.resources.gpus_per_node=${GPUS_PER_NODE}"
  "env.nemo_gym.nemo_gym_log_dir=${LOG_DIR}/nemo_gym"
  "data.train.data_path=${NRL_TRAIN_PATH}"
  "data.validation.data_path=${NRL_VAL_PATH}"
  "checkpointing.checkpoint_dir=${CHECKPOINT_DIR}"
  "logger.log_dir=${LOG_DIR}"
  "logger.wandb_enabled=${WANDB_ENABLED}"
  "logger.wandb.name=${WANDB_NAME}"
  "logger.wandb.project=${WANDB_PROJ}"
)

if [[ -n "${NRL_MAX_STEPS:-}" ]]; then
  TRAIN_ARGS+=("grpo.max_num_steps=${NRL_MAX_STEPS}")
fi
if [[ "${ENABLE_MTP_INFERENCE:-0}" == "1" ]]; then
  TRAIN_ARGS+=(
    "++policy.generation.vllm_cfg.enable_prefix_caching=true"
    "++policy.generation.vllm_kwargs.enable_chunked_prefill=true"
    "++policy.generation.vllm_kwargs.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8480}"
    "++policy.generation.vllm_kwargs.mamba_cache_mode=align"
    "~policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes"
    "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-5}"
    "++policy.generation.vllm_kwargs.speculative_config.method=mtp"
  )
fi
TRAIN_ARGS+=("$@")

printf -v TRAIN_ARGS_QUOTED '%q ' "${TRAIN_ARGS[@]}"
printf -v CODE_ROOT_QUOTED '%q' "${CODE_ROOT:-/opt/nemo-rl}"
printf -v CACHE_ROOT_QUOTED '%q' "${PERSISTENT_CACHE}"
TRAIN_CMD="cd ${CODE_ROOT_QUOTED} && date && source /opt/nemo-rl/3rdparty/vllm/nemo-rl.env && \
OMP_NUM_THREADS=16 RAY_DEDUP_LOGS=1 NRL_VLLM_USE_V1=1 \
VLLM_CACHE_ROOT=${NRL_VLLM_LOCAL_CACHE_DIR} \
DG_JIT_CACHE_DIR=${NRL_VLLM_LOCAL_CACHE_DIR}/deep_gemm \
TORCHINDUCTOR_CACHE_DIR=${INDUCTOR_CACHE_DIR} \
TRITON_CACHE_DIR=${TRITON_CACHE_DIR} \
UV_CACHE_DIR=${CACHE_ROOT_QUOTED}/uv \
NRL_USE_FASTOKENS=${NRL_USE_FASTOKENS:-1} \
${TRAIN_ARGS_QUOTED}"

export COMMAND="${TRAIN_CMD}"

RAY_SUB="${RAY_SUB:-${PROJECT_ROOT}/ray_het.sub}"
if [[ ! -f "${RAY_SUB}" ]]; then
  echo "ERROR: Heterogeneous Ray submission script not found: ${RAY_SUB}" >&2
  exit 1
fi

WALLTIME="${WALLTIME:-4:00:00}"
DEPENDENCY="singleton"
SBATCH_ARGS=(
  --nodes="${NUM_RAY_NODES}"
  --account="${SLURM_ACCOUNT}"
  --job-name="${WANDB_NAME}"
  --partition="${SLURM_PARTITION}"
  --time="${WALLTIME}"
  --gres="gpu:${GPUS_PER_NODE}"
  --output="${SLURM_LOG_DIR}/%x-%j.out"
  --error="${SLURM_LOG_DIR}/%x-%j.err"
  --exclusive
  --mem=0
  --dependency="${DEPENDENCY}"
)
[[ -n "${SLURM_QOS:-}" ]] && SBATCH_ARGS+=(--qos="${SLURM_QOS}")
[[ -n "${SLURM_RESERVATION:-}" ]] && SBATCH_ARGS+=(--reservation="${SLURM_RESERVATION}")
[[ -n "${EXCLUDE_NODES:-}" ]] && SBATCH_ARGS+=(--exclude="${EXCLUDE_NODES}")
[[ -n "${SEGMENT_SIZE:-}" ]] && SBATCH_ARGS+=(--segment="${SEGMENT_SIZE}")

if (( HET_SERVER_COUNT > 0 )); then
  for ((het_group = 1; het_group <= HET_SERVER_COUNT; het_group++)); do
    nodes_var="HET_GROUP_${het_group}_NODES"
    group_nodes="${!nodes_var:-${HET_SERVER_NODES}}"
    SBATCH_ARGS+=(
      :
      --nodes="${group_nodes}"
      --account="${SLURM_ACCOUNT}"
      --job-name="${WANDB_NAME}-judge-${het_group}"
      --partition="${SLURM_PARTITION}"
      --time="${WALLTIME}"
      --gres="gpu:${HET_SERVER_GPUS_PER_NODE}"
      --exclusive
      --mem=0
    )
    [[ -n "${SLURM_QOS:-}" ]] && SBATCH_ARGS+=(--qos="${SLURM_QOS}")
    [[ -n "${SLURM_RESERVATION:-}" ]] && SBATCH_ARGS+=(--reservation="${SLURM_RESERVATION}")
  done
fi

echo "Ray nodes: ${NUM_RAY_NODES} (training=$((NUM_ACTOR_NODES - GENERATION_NUM_NODES)), rollout=${GENERATION_NUM_NODES})"
echo "Judge components: ${HET_SERVER_COUNT} x ${HET_SERVER_NODES} node(s)"
if (( HET_SERVER_COUNT > 0 )); then
  echo "Judge model: ${PROOF_JUDGE_MODEL} (port=${PROOF_JUDGE_PORT}, tp=${PROOF_JUDGE_TP_SIZE})"
fi
echo "Results: ${RESULTS_DIR}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Training command:"
  printf '%s\n' "${TRAIN_CMD}"
  echo "Submission command:"
  printf 'sbatch '
  printf '%q ' "${SBATCH_ARGS[@]}" "${RAY_SUB}"
  printf '\n'
  exit 0
fi

sbatch "${SBATCH_ARGS[@]}" "${RAY_SUB}"
