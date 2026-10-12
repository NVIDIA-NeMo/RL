#!/bin/bash
set -euo pipefail

# =============================================================================
# scripts/repro_super_prod_stage1.1.sh
#
# GRPO Super V3 pipe-cleaning on GB200 NVL72 with NeMo Gym on the ultra-v3-posttraining branch
#
# By default, this runs from what's built into the container without overlay mounts applied. 
# Set USE_WORKTREE=1 to overlay your local worktree submodules for development.
# Set INTERACTIVE=1 to get a persistent allocation in slurm for iterative debugging.
#
# Usage:
#   ./repro_super_prod_stage1.1.sh                                   # batch, bare container (10 steps)
#   NRL_MAX_STEPS=4 ./repro_super_prod_stage1.1.sh                   # CI: fewer steps
#   USE_WORKTREE=1 ./repro_super_prod_stage1.1.sh                    # batch, overlay local code
#   WALLTIME=4:00:00 ./repro_super_prod_stage1.1.sh
#
# Extra positional arguments are forwarded as Hydra overrides:
#   ./repro_super_prod_stage1.1.sh grpo.max_num_steps=2 policy.precision=float32
#
# Interactive debugging (reuse allocation across runs):
#   INTERACTIVE=1 ./repro_super_prod_stage1.1.sh                     # submits, auto-runs, waits
#   INTERACTIVE=1 INTERACTIVE_WAIT=0 ./repro_super_prod_stage1.1.sh  # submit only (no foreground wait)
#   INTERACTIVE=1 INTERACTIVE_WALLTIME=2:0:0 SLURM_QOS=short ./repro_super_prod_stage1.1.sh  # submit and wait in foreground
#
#   A background watcher auto-runs the training command as soon as Ray is ready,
#   so GPUs are never idle waiting for you to type. After training finishes the
#   allocation stays alive — re-attach and iterate without requeueing.
#
#   Once Ray is up, you can:
#     # Run non-interactively from login node
#     COMMAND="$(cat <jobid>-run-cmd.sh)" bash <jobid>-attach.sh
#
#     # Or attach interactively, then run inside the container
#     bash <jobid>-attach.sh
#     source <jobid>-run-cmd.sh
#
#     # Edit and re-run without requeueing
#     vim <jobid>-run-cmd.sh
#     COMMAND="$(cat <jobid>-run-cmd.sh)" bash <jobid>-attach.sh
# =============================================================================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=${SCRIPT_DIR}
cd ${PROJECT_ROOT}

USE_WORKTREE="${USE_WORKTREE:-0}"
INTERACTIVE="${INTERACTIVE:-0}"
INTERACTIVE_WAIT="${INTERACTIVE_WAIT:-1}"



# ---------- Precision configuration ------
get_precision_config() {
  local PRECISION_RECIPE="$1"
  local DISABLE_FP8_LINEAR="$2"
  local DISABLE_FP8_MOE="$3"
  local ENABLE_FP8_PARAM_IN_TRAIN="$4"
  local PRECISION_EXTRA_ARGS=""

  MXFP8_GEN_EXTRA_ARGS="policy.generation.vllm_cfg.precision=fp8 \
++policy.generation.vllm_cfg.fp8_cfg.is_mx=true \
policy.generation.vllm_cfg.gpu_memory_utilization=0.8 \
policy.generation.vllm_cfg.tensor_parallel_size=4 \
policy.generation.vllm_cfg.expert_parallel_size=4"

  IGNORED_LAYER_KWS="\"conv1d\",\"mtp\""
  if [ "$DISABLE_FP8_MOE" == "1" ]; then
  IGNORED_LAYER_KWS="$IGNORED_LAYER_KWS,\".experts.\""
  fi
  if [ "$DISABLE_FP8_LINEAR" == "1" ]; then
  IGNORED_LAYER_KWS="$IGNORED_LAYER_KWS,\"in_proj\",\"out_proj\",\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\"fc1_latent_proj\",\"fc2_latent_proj\",\"shared_experts\""
  fi
  MXFP8_GEN_EXTRA_ARGS="$MXFP8_GEN_EXTRA_ARGS +policy.generation.vllm_cfg.quantization_ignored_layer_kws=[$IGNORED_LAYER_KWS]"

  MXFP8_TRAIN_EXTRA_ARGS="policy.megatron_cfg.fp8_cfg.enabled=true \
policy.megatron_cfg.fp8_cfg.fp8="e4m3" \
policy.megatron_cfg.fp8_cfg.fp8_recipe="mxfp8" \
++policy.megatron_cfg.fp8_cfg.fp8_param=false \
policy.megatron_cfg.moe_router_dtype=fp32 \
policy.megatron_cfg.expert_model_parallel_size=64 \
"

  MXFP8_PARAM_EXTRA_ARGS="++policy.megatron_cfg.fp8_cfg.fp8_param=true \
+policy.megatron_cfg.optimizer.reuse_grad_buf_for_mxfp8_param_ag=true \
+policy.megatron_cfg.optimizer.fp8_recipe=mxfp8 \
+policy.megatron_cfg.optimizer.overlap_param_gather=true \
++policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=true \
++policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=true \
"

  if [ "$ENABLE_FP8_PARAM_IN_TRAIN" == "1" ]; then
  MXFP8_TRAIN_EXTRA_ARGS="$MXFP8_TRAIN_EXTRA_ARGS $MXFP8_PARAM_EXTRA_ARGS"
  fi

  if [ "$PRECISION_RECIPE" == "mxfp8-rollout" ]; then
  PRECISION_EXTRA_ARGS="$MXFP8_GEN_EXTRA_ARGS"
  elif [ "$PRECISION_RECIPE" == "mxfp8-train" ]; then
  PRECISION_EXTRA_ARGS="$MXFP8_TRAIN_EXTRA_ARGS"
  elif [ "$PRECISION_RECIPE" == "mxfp8-e2e" ]; then
  PRECISION_EXTRA_ARGS="$MXFP8_GEN_EXTRA_ARGS $MXFP8_TRAIN_EXTRA_ARGS"
  else
  PRECISION_EXTRA_ARGS=""
  fi

  echo "${PRECISION_EXTRA_ARGS}"
}

PRECISION_RECIPE="${PRECISION_RECIPE:-bf16}"
DISABLE_FP8_LINEAR="${DISABLE_FP8_LINEAR:-0}"
DISABLE_FP8_MOE="${DISABLE_FP8_MOE:-0}"
ENABLE_FP8_PARAM_IN_TRAIN="${ENABLE_FP8_PARAM_IN_TRAIN:-0}"
PRECISION_EXTRA_ARGS=$(get_precision_config "${PRECISION_RECIPE}" "${DISABLE_FP8_LINEAR}" "${DISABLE_FP8_MOE}" "${ENABLE_FP8_PARAM_IN_TRAIN}")

echo "PRECISION_RECIPE: ${PRECISION_RECIPE}"
echo "PRECISION_EXTRA_ARGS: ${PRECISION_EXTRA_ARGS}"

# ---------- SLURM configuration ----------
SLURM_ACCOUNT="${SLURM_ACCOUNT:-llmservice_nemotron_ultra}"
PARTITION="${PARTITION:-batch}"
SLURM_QOS="${SLURM_QOS:-normal}"
WALLTIME="${WALLTIME:-4:00:00}"

# ---------- Container & mounts ----------
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/nemo_rl/images/pipe.45898348.sqsh}"
MOUNTS="/lustre:/lustre"

# GB200 NVL72: fixed at 4 GPUs/node. Must match --gres=gpu:4 passed to sbatch.
export GPUS_PER_NODE=4
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-144}"

# ---------- Persistent cache directories ----------
PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fsw/portfolios/llmservice/users/${USER}/.cache/nemotron_ultra}"

# ---------- HuggingFace Configuration ----------
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/llmservice/users/${USER}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/hub}"

# ---------- W&B Configuration ----------
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# ---------- Model Configuration ----------
TP="${TP:-4}"
CP="${CP:-4}"
EP="${EP:-8}"
PP="${PP:-1}"
ETP="${ETP:-1}"
VLLM_TP="${VLLM_TP:-4}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.8}"
MAX_LENGTH="${MAX_LENGTH:-49152}"

# ---------- Training ----------
NRL_MAX_STEPS="${NRL_MAX_STEPS:-1000000}"
VAL_PERIOD="${VAL_PERIOD:-10000}"
SAVE_PERIOD="${SAVE_PERIOD:-10}"
LR="${LR:-3e-6}"
MIN_LR="${MIN_LR:-3e-6}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-10}"
KL="${KL:-0}"

# ---------- GRPO ----------
PPS="${PPS:-256}"
GPP="${GPP:-16}"
GBS="${GBS:-4096}"
FORCE_ON_POLICY_RATIO="${FORCE_ON_POLICY_RATIO:-True}"
TIS_THRESHOLD="${TIS_THRESHOLD:-5}"
SEQ_LOGPROB_ERROR_THRESHOLD="${SEQ_LOGPROB_ERROR_THRESHOLD:-2}"
ADVANTAGE_CLIP_LOW="${ADVANTAGE_CLIP_LOW:--100}"
ADVANTAGE_CLIP_HIGH="${ADVANTAGE_CLIP_HIGH:-100}"
OVERLONG_FILTERING="${OVERLONG_FILTERING:-False}"

# ---------- Async GRPO ----------
ASYNC_GRPO=True
MAX_TRAJECTORY_AGE_STEPS=1
IN_FLIGHT_WEIGHT_UPDATES=True
RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES=False
COLOCATED_INFERENCE=False

# ---------- Job Shape ----------
GENERATION_NUM_NODES="${GENERATION_NUM_NODES:-59}"
NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-123}"
COLOCATED_INFERENCE="${COLOCATED_INFERENCE:-False}"

NUM_GENRM_NODES="${NUM_GENRM_NODES:-2}"
NUM_LLMJUDGE_NODES="${NUM_LLMJUDGE_NODES:-2}"
NUM_SAFETY_NODES="${NUM_SAFETY_NODES:-1}"
NUM_GYM_EXTRA_NODES="${NUM_GYM_EXTRA_NODES:-0}"
NUM_JUDGE_NODES=$((NUM_GENRM_NODES + NUM_LLMJUDGE_NODES + NUM_SAFETY_NODES + NUM_GYM_EXTRA_NODES))
NUM_TOTAL_NODES=$((NUM_ACTOR_NODES + NUM_JUDGE_NODES))

# ---------- DO NOT CHANGE ----------
# ---------- Logging, W&B, and Job Prefix ----------
export BASE_LOG_DIR="/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/nemo_rl/logs"
WANDB_PROJ="ultra-v3-pipeclean"
JOB_PREFIX="${JOB_PREFIX:-pipeclean-ultra-rl}"
# ---------- DO NOT CHANGE ----------

# ---------- Ray log sync (copy actor logs from /tmp/ray to $LOG_DIR/ray/) ----------
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-60}"

EXP_SUFFIX="${JOB_PREFIX}-super-v3-grpo_sft-quantum-apex_warping-muscox_tp${TP}_cp${CP}_ep${EP}_pp${PP}_gpp${GPP}_pps${PPS}_gbs${GBS}"
export BASE_LOG_DIR="${BASE_LOG_DIR}/${EXP_SUFFIX}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/${EXP_SUFFIX}}"
mkdir -p "${CHECKPOINT_DIR}"
CHECKPOINT_DIR="$(cd "${CHECKPOINT_DIR}" && pwd)"


# GB200 NVL72: each rack has 18 nodes sharing an NVLink domain.
# --segment tells SLURM to allocate nodes in groups of this size from
# the same topology block, guaranteeing complete rack-aligned segments for training EP.
# Inference and judges inherit the constraint but don't require it.
# Must stay in sync with cluster.segment_size in the YAML config.
#
# When SEGMENT_SIZE is unset, default to 16 if NUM_TOTAL_NODES is divisible by 16.
# Slurm requires the node count to be evenly divisible by the segment size.
SEGMENT_SIZE="${SEGMENT_SIZE:-}"
if [ -z "${SEGMENT_SIZE}" ] && [ "$((NUM_TOTAL_NODES % 16))" -eq 0 ] && [ "${NUM_TOTAL_NODES}" -ge 16 ]; then
  SEGMENT_SIZE=16
fi
if [ -n "${SEGMENT_SIZE}" ] && [ "${SEGMENT_SIZE}" -gt 1 ]; then
  if [ "${NUM_TOTAL_NODES}" -lt "${SEGMENT_SIZE}" ]; then
    echo "ERROR: NUM_TOTAL_NODES=${NUM_TOTAL_NODES} < SEGMENT_SIZE=${SEGMENT_SIZE}" >&2
    exit 1
  fi
  if [ "$((NUM_TOTAL_NODES % SEGMENT_SIZE))" -ne 0 ]; then
    echo "ERROR: NUM_TOTAL_NODES=${NUM_TOTAL_NODES} is not evenly divisible by SEGMENT_SIZE=${SEGMENT_SIZE}" >&2
    echo "  Slurm requires --nodes to be a multiple of --segment." >&2
    echo "  Adjust NUM_TOTAL_NODES (currently: actor=${NUM_ACTOR_NODES} + judge=${NUM_JUDGE_NODES})" >&2
    echo "  or set SEGMENT_SIZE to a divisor of ${NUM_TOTAL_NODES}." >&2
    exit 1
  fi
fi

# ---------- Model and data paths ----------
NRL_TRAIN_PATH="${NRL_TRAIN_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/data/gym/rl-data-tools/blends/curriculum_v29_warping-muskox.train.no_swerl.jsonl}"
NRL_VAL_PATH="${NRL_VAL_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/data/gym/rl-data-tools/blends/curriculum_v29_warping-muskox.val.no_swerl.jsonl}"
#NRL_MODEL_PATH="${NRL_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/dmosallanezh/models/nemotronsuper_vQuantumApex}"
NRL_MODEL_PATH="${NRL_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/soumyes/share/ckpts/super-v3/super-v3-sft-quantum-apex}"
NRL_GENRM_MODEL_PATH="${NRL_GENRM_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/qwen235b_principle_comparison_genrm_step1230}"
NRL_NL2BASH_JUDGE_MODEL_PATH="${NRL_NL2BASH_JUDGE_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
NRL_SAFETY_MODEL_PATH="${NRL_SAFETY_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/super_v3/model_checkpoints/Nemotron-Content-Safety-Reasoning-4B}"

# ---------- Lean4 sandbox (for math_formal_lean) ----------
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-sandbox-latest.sqsh}"
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-/start-with-nginx.sh}"
export NEMO_SKILLS_SANDBOX_PORT="${NEMO_SKILLS_SANDBOX_PORT:-6000}"

# ---------- W&B Name ----------
WANDB_NAME="${EXP_SUFFIX}"

# ---------- Code snapshot ----------
# Batch mode: snapshot by default so code is frozen at submission time.
# Interactive mode: live directory by default for fast iteration.
# Override with USE_SNAPSHOT=0 or USE_SNAPSHOT=1 to force either behavior.
if [[ "${INTERACTIVE}" == "1" ]]; then
  USE_SNAPSHOT="${USE_SNAPSHOT:-0}"
else
  USE_SNAPSHOT="${USE_SNAPSHOT:-1}"
fi

if [[ "${USE_SNAPSHOT}" == "1" ]]; then
  SNAPSHOT_DIR=$(bash "${PROJECT_ROOT}/tools/code_snapshot.sh" "${EXP_SUFFIX}")

  # Symlink 3rdparty/vllm if present (large, not git-tracked in all setups)
  if [[ -d "${PROJECT_ROOT}/3rdparty/vllm" ]] && [[ ! -e "${SNAPSHOT_DIR}/3rdparty/vllm" ]]; then
    mkdir -p "${SNAPSHOT_DIR}/3rdparty"
    ln -s "${PROJECT_ROOT}/3rdparty/vllm" "${SNAPSHOT_DIR}/3rdparty/vllm"
  fi

  echo "Code snapshot: ${SNAPSHOT_DIR}"
  OVERLAY_SOURCE="${SNAPSHOT_DIR}"
else
  OVERLAY_SOURCE="${PROJECT_ROOT}"
fi

# ---------- Persistent cache directories ----------
# Per-user cache for compiled artifacts (vLLM, FlashInfer cubins, Deep Gemm
# JIT, Triton, Inductor, uv).  Each user gets their own directory to avoid
# shared-permission issues on Lustre.
#
# Default path: /lustre/fsw/portfolios/{access_group}/users/$USER/.cache
# where {access_group} is the first segment of SLURM_ACCOUNT
# (e.g. llmservice_nemotron_ultra → llmservice).
#
# Override with PERSISTENT_CACHE=/path/to/your/cache if needed.
if [[ -z "${PERSISTENT_CACHE:-}" ]]; then
  _access_group="${SLURM_ACCOUNT%%_*}"
  PERSISTENT_CACHE="/lustre/fsw/portfolios/llmservice/users/${USER}/.cache/nemotron_ultra"
fi
VLLM_CACHE_DIR="${PERSISTENT_CACHE}/vllm_compile_cache"
FLASHINFER_CUBIN_CACHE="${PERSISTENT_CACHE}/flashinfer_cubins"
FLASHINFER_WS_BASE="${PERSISTENT_CACHE}/flashinfer_workspace"
#FLASHINFER_CUBIN_CACHE="/lustre/fsw/portfolios/llmservice/users/ansubramania/.cache/nemotron_ultra/flashinfer_cubins/"
#FLASHINFER_WS_BASE="/lustre/fsw/portfolios/llmservice/users/ansubramania/.cache/nemotron_ultra/flashinfer_workspace/"
LUSTRE_INDUCTOR_CACHE="${PERSISTENT_CACHE}/inductor_cache"
LUSTRE_TRITON_CACHE="${PERSISTENT_CACHE}/triton_cache"
INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"

mkdir -p "${VLLM_CACHE_DIR}" "${FLASHINFER_CUBIN_CACHE}" "${FLASHINFER_WS_BASE}" \
  "${LUSTRE_INDUCTOR_CACHE}" "${LUSTRE_TRITON_CACHE}"

VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://github.com/vllm-project/vllm/releases/download/v0.17.0/vllm-0.17.0-cp38-abi3-manylinux_2_31_aarch64.whl}"

# =============================================================================
# Validation
# =============================================================================

# Walltime cap: fail early if walltime exceeds partition limit.
_walltime_secs() {
  local t="$1" h m s
  IFS=: read -r h m s <<< "${t}"
  echo $(( 10#${h} * 3600 + 10#${m} * 60 + 10#${s} ))
}

case "${SLURM_QOS}" in
  batch_large_long) _max_walltime=$(( 7 * 24 * 3600 )); _max_label="7-day" ;;
  *)                _max_walltime=$(( 4 * 3600 ));       _max_label="4-hour" ;;
esac

if (( $(_walltime_secs "${WALLTIME}") > _max_walltime )); then
  echo "ERROR: WALLTIME=${WALLTIME} exceeds the ${_max_label} maximum for QOS ${SLURM_QOS:-default}."
  exit 1
fi

# QOS=interactive caps walltime at 2 hours.
if [[ "${SLURM_QOS}" == "interactive" ]]; then
  if (( $(_walltime_secs "${INTERACTIVE_WALLTIME:-${WALLTIME}}") > 2 * 3600 )); then
    echo "ERROR: SLURM_QOS=interactive requires walltime <= 2 hours."
    echo "  Set INTERACTIVE_WALLTIME=2:0:0 or use a different QOS (e.g. SLURM_QOS=short)."
    exit 1
  fi
fi

# W&B: warn (but don't fail) if WANDB_API_KEY is unset — runs will log locally only.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARNING: WANDB_API_KEY is not set. W&B logging will fail or fall back to offline mode."
  echo "  export WANDB_API_KEY=<your-key> to enable cloud logging."
fi

# HF_TOKEN: required when loading models from the HuggingFace Hub (not local paths).
# Hub IDs look like "org/model-name" (no leading slash). Local paths start with "/".
if [[ "${NRL_MODEL_PATH}" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_./-]+$ ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: NRL_MODEL_PATH (${NRL_MODEL_PATH}) looks like a HuggingFace Hub model ID"
    echo "  but HF_TOKEN is not set. Export HF_TOKEN to authenticate with the Hub."
    exit 1
  fi
fi

# =============================================================================
# Worktree setup (only when USE_WORKTREE=1)
# =============================================================================
if [[ "${USE_WORKTREE}" == "1" ]]; then
  WORKTREE_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"
  MAIN_REPO_ROOT="${MAIN_REPO_ROOT:-$(git -C "${WORKTREE_ROOT}" worktree list --porcelain | awk '/^worktree /{print $2}' | grep -v '/.worktrees/' | head -n1)}"

  if [[ -z "${MAIN_REPO_ROOT}" || ! -d "${MAIN_REPO_ROOT}" ]]; then
    echo "Could not resolve MAIN_REPO_ROOT; set MAIN_REPO_ROOT explicitly."
    exit 1
  fi

  if [[ ! -f "${MAIN_REPO_ROOT}/3rdparty/vllm/nemo-rl.env" ]]; then
    echo "Missing main vLLM env file: ${MAIN_REPO_ROOT}/3rdparty/vllm/nemo-rl.env"
    exit 1
  fi

  MISSING=0
  for p in \
    "${WORKTREE_ROOT}/3rdparty/Gym-workspace/Gym/nemo_gym/cli.py" \
    "${WORKTREE_ROOT}/3rdparty/Megatron-LM-workspace/Megatron-LM" \
    "${WORKTREE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" \
    "${WORKTREE_ROOT}/3rdparty/Automodel-workspace/Automodel"
  do
    if [[ ! -e "${p}" ]]; then
      echo "Missing required worktree path: ${p}"
      MISSING=1
    fi
  done
  if [[ "${MISSING}" -ne 0 ]]; then
    echo "Initialize submodules on login node first:"
    echo "  git -C ${WORKTREE_ROOT} submodule update --init --recursive"
    exit 1
  fi
  echo "Worktree mode: overlaying ${WORKTREE_ROOT}"
  echo "Main repo vLLM: ${MAIN_REPO_ROOT}/3rdparty/vllm"
fi


# =============================================================================
# Code root — container path or worktree
# =============================================================================
# NOTE: In bare container mode we assume /opt/nemo-rl/3rdparty/vllm/nemo-rl.env
# exists inside the container. 
# This can't be verified from the login node.
if [[ "${USE_WORKTREE}" == "1" ]]; then
  CODE_ROOT="${WORKTREE_ROOT}"
  VLLM_ENV_SOURCE="source ${MAIN_REPO_ROOT}/3rdparty/vllm/nemo-rl.env && "
else
  CODE_ROOT="/opt/nemo-rl"
  VLLM_ENV_SOURCE="source /opt/nemo-rl/3rdparty/vllm/nemo-rl.env && "
fi

echo "Nodes: ${NUM_TOTAL_NODES} (actor=${NUM_ACTOR_NODES} [train=$((NUM_ACTOR_NODES - GENERATION_NUM_NODES)), gen=${GENERATION_NUM_NODES}], judge=${NUM_JUDGE_NODES})"
echo "Code root: ${CODE_ROOT}"
echo "Persistent cache root: ${PERSISTENT_CACHE}"

# =============================================================================
# Build the training command
# =============================================================================
# All env vars that need to reach compute nodes are set INSIDE the command
# string. sbatch does not propagate the login node's exports — ray.sub starts
# a fresh shell and executes $COMMAND via enroot exec inside the container.
#
# All static config (parallelism, vLLM kwargs, judge server_args, sequence
# packing, etc.) lives in grpo_ultra_v3.yaml. Only per-run variables are
# overridden here.
TRAIN_CMD="cd ${CODE_ROOT} && date ; \
${VLLM_ENV_SOURCE}\
OMP_NUM_THREADS=16 \
RAY_DEDUP_LOGS=1 \
NRL_VLLM_USE_V1=1 \
VLLM_CACHE_ROOT=${VLLM_CACHE_DIR} \
DG_JIT_CACHE_DIR=${VLLM_CACHE_DIR}/deep_gemm \
TORCHINDUCTOR_CACHE_DIR=${INDUCTOR_CACHE_DIR} \
TRITON_CACHE_DIR=${TRITON_CACHE_DIR} \
UV_CACHE_DIR=${PERSISTENT_CACHE}/uv \
NEMO_GYM_SKIP_VENV_IF_PRESENT=1 \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
UV_HTTP_TIMEOUT=10 \
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_PRECOMPILED_WHEEL_LOCATION} \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
VLLM_FLASHINFER_MOE_BACKEND=latency \
FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_CACHE} \
FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WS_BASE} \
NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800 \
HF_HOME=${HF_HOME} \
HF_TOKEN=${HF_TOKEN:-} \
uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
--config examples/configs/grpo_repro_super_stage1.1.yaml \
policy.model_name=${NRL_MODEL_PATH} \
cluster.gpus_per_node=4 \
cluster.num_nodes=${NUM_TOTAL_NODES} \
grpo.val_period=${VAL_PERIOD} \
grpo.num_prompts_per_step=${PPS} \
grpo.num_generations_per_prompt=${GPP} \
grpo.advantage_clip_low=${ADVANTAGE_CLIP_LOW} \
grpo.advantage_clip_high=${ADVANTAGE_CLIP_HIGH} \
grpo.seq_logprob_error_threshold=${SEQ_LOGPROB_ERROR_THRESHOLD} \
grpo.async_grpo.enabled=${ASYNC_GRPO} \
grpo.async_grpo.max_trajectory_age_steps=${MAX_TRAJECTORY_AGE_STEPS} \
grpo.async_grpo.in_flight_weight_updates=${IN_FLIGHT_WEIGHT_UPDATES} \
grpo.async_grpo.recompute_kv_cache_after_weight_updates=${RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES} \
policy.train_global_batch_size=${GBS} \
policy.max_total_sequence_length=${MAX_LENGTH} \
policy.megatron_cfg.tensor_model_parallel_size=${TP} \
policy.megatron_cfg.context_parallel_size=${CP} \
policy.megatron_cfg.expert_model_parallel_size=${EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${PP} \
policy.megatron_cfg.expert_tensor_parallel_size=${ETP} \
policy.megatron_cfg.scheduler.lr_warmup_iters=${LR_WARMUP_ITERS} \
policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP} \
policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_GPU_UTIL} \
policy.generation.vllm_cfg.max_model_len=${MAX_LENGTH} \
policy.generation.colocated.enabled=${COLOCATED_INFERENCE} \
policy.generation.colocated.resources.num_nodes=${GENERATION_NUM_NODES} \
policy.generation.colocated.resources.gpus_per_node=4 \
env.nemo_gym.num_gpu_nodes=${NUM_JUDGE_NODES} \
env.nemo_gym.genrm_model.responses_api_models.genrm_model.model=${NRL_GENRM_MODEL_PATH} \
env.nemo_gym.nl2bash_judge_model.responses_api_models.local_vllm_model.model=${NRL_NL2BASH_JUDGE_MODEL_PATH} \
env.nemo_gym.safety_judge_model.responses_api_models.local_vllm_model.model=${NRL_SAFETY_MODEL_PATH} \
loss_fn.force_on_policy_ratio=${FORCE_ON_POLICY_RATIO} \
loss_fn.truncated_importance_sampling_ratio=${TIS_THRESHOLD} \
loss_fn.reference_policy_kl_penalty=${KL} \
data.train.data_path=${NRL_TRAIN_PATH} \
data.validation.data_path=${NRL_VAL_PATH} \
checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
checkpointing.save_period=${SAVE_PERIOD} \
logger.log_dir=${CHECKPOINT_DIR}/logs \
logger.wandb_enabled=True \
logger.wandb.name=${WANDB_NAME} \
logger.wandb.project=${WANDB_PROJ} \
${NRL_MAX_STEPS:+grpo.max_num_steps=${NRL_MAX_STEPS}} \
${PRECISION_EXTRA_ARGS} \
${*}"


# =============================================================================
# Overlay mounts
# =============================================================================
# Local source directories are bind-mounted into the container so edits on
# Lustre take effect without rebuilding the container.
#
# MOUNTED BY DEFAULT:
#   NRL_NEMO_RL_DIR      → /opt/nemo-rl/nemo_rl          (Python package)
#   NRL_CONFIGS_DIR      → /opt/nemo-rl/examples/configs  (YAML configs)
#
# OPT-IN ONLY (set the env var to enable — overlaying these shadows prebuilt
# venvs and compiled artifacts inside the container, forcing expensive
# re-creation at startup):
#   NRL_MEGATRON_LM_DIR     → /opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM
#   NRL_MEGATRON_BRIDGE_DIR → /opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
#   NRL_GYM_DIR             → /opt/nemo-rl/3rdparty/Gym-workspace/Gym
#   NRL_VLLM_DIR            → /opt/nemo-rl/3rdparty/vllm
#
# To enable a 3rdparty mount, set the var to the host path. Example:
#   NRL_GYM_DIR=/path/to/Gym ./launch_ultra_pipeclean.sh
#
# Paths that don't exist on disk are silently skipped (container built-ins
# are used instead). Set any var to "" to explicitly skip a default mount.
# =============================================================================
NRL_NEMO_RL_DIR="${NRL_NEMO_RL_DIR:-${OVERLAY_SOURCE}/nemo_rl}"
NRL_CONFIGS_DIR="${NRL_CONFIGS_DIR:-${OVERLAY_SOURCE}/examples/configs}"
NRL_MEGATRON_LM_DIR="${NRL_MEGATRON_LM_DIR:-}"
NRL_MEGATRON_BRIDGE_DIR="${NRL_MEGATRON_BRIDGE_DIR:-}"
NRL_GYM_DIR="${NRL_GYM_DIR:-}"
NRL_VLLM_DIR="${NRL_VLLM_DIR:-}"

_maybe_mount() {
  local src="$1" dst="$2" label="$3"
  if [[ -z "${src}" ]]; then
    return
  fi
  if [[ -d "${src}" ]]; then
    MOUNTS="${MOUNTS},${src}:${dst}"
    echo "  Mount: ${label} → ${dst}"
  else
    echo "  Skip:  ${label} (${src} not found on disk, using container built-in)"
  fi
}

echo ""
echo "Overlay mounts:"
_maybe_mount "${NRL_NEMO_RL_DIR}" "/opt/nemo-rl/nemo_rl" "nemo_rl"
_maybe_mount "${NRL_CONFIGS_DIR}" "/opt/nemo-rl/examples/configs" "configs"
_maybe_mount "${NRL_MEGATRON_LM_DIR}" "/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM" "Megatron-LM"
_maybe_mount "${NRL_MEGATRON_BRIDGE_DIR}" "/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" "Megatron-Bridge"
_maybe_mount "${NRL_GYM_DIR}" "/opt/nemo-rl/3rdparty/Gym-workspace/Gym" "NeMo-Gym"
_maybe_mount "${NRL_VLLM_DIR}" "/opt/nemo-rl/3rdparty/vllm" "vLLM"

if [[ "${USE_WORKTREE}" == "1" ]]; then
  MOUNTS="${MOUNTS},${WORKTREE_ROOT}:${WORKTREE_ROOT}"
fi

if [[ "${USE_SNAPSHOT}" == "1" ]]; then
  MOUNTS="${MOUNTS},${SNAPSHOT_DIR}:${SNAPSHOT_DIR}"
fi

if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
  MOUNTS="${MOUNTS},${EXTRA_MOUNTS}"
fi

export MOUNTS

# Resolve ray.sub
if [[ "${USE_WORKTREE}" == "1" ]]; then
  RAY_SUB="${WORKTREE_ROOT}/ray.sub"
else
  RAY_SUB="${RAY_SUB:-${PROJECT_ROOT}/ray.sub}"
fi

if [[ ! -f "${RAY_SUB}" ]]; then
  echo "ERROR: ray.sub not found at ${RAY_SUB}"
  echo "Set RAY_SUB=/path/to/ray.sub or use USE_WORKTREE=1"
  exit 1
fi

# =================================================================================================================
# Per-node cache seeding
# =================================================================================================================
# Triton and Inductor compile to node-local /tmp to avoid Lustre race conditions during concurrent JIT compilation.
# To avoid cold-start penalties, we seed /tmp from a warm Lustre cache before Ray starts (SETUP_COMMAND)
# and sync new artifacts back afterwards (TEARDOWN_COMMAND).
# Both commands run on every node via ray.sub.
# =================================================================================================================
read -r -d '' SETUP_COMMAND <<SETUPEOF || true
echo "[CACHE SEED] Seeding Triton/Inductor caches from Lustre..."
LOCAL_IND="${INDUCTOR_CACHE_DIR}"
LOCAL_TRI="${TRITON_CACHE_DIR}"
LUSTRE_IND="${LUSTRE_INDUCTOR_CACHE}"
LUSTRE_TRI="${LUSTRE_TRITON_CACHE}"
mkdir -p "\$LOCAL_IND" "\$LOCAL_TRI"
if [ -d "\$LUSTRE_IND" ] && [ "\$(ls -A "\$LUSTRE_IND" 2>/dev/null)" ]; then
  cp -a "\$LUSTRE_IND/." "\$LOCAL_IND/" && echo "[CACHE SEED] Inductor: seeded from Lustre" \
    || echo "[CACHE SEED] Inductor: seed failed (non-fatal)"
else
  echo "[CACHE SEED] Inductor: no warm cache on Lustre yet"
fi
if [ -d "\$LUSTRE_TRI" ] && [ "\$(ls -A "\$LUSTRE_TRI" 2>/dev/null)" ]; then
  cp -a "\$LUSTRE_TRI/." "\$LOCAL_TRI/" && echo "[CACHE SEED] Triton: seeded from Lustre" \
    || echo "[CACHE SEED] Triton: seed failed (non-fatal)"
else
  echo "[CACHE SEED] Triton: no warm cache on Lustre yet"
fi
echo "[CACHE SEED] Done."
SETUPEOF
export SETUP_COMMAND


# =============================================================================
# Interactive mode
# =============================================================================
# When COMMAND is empty/unset, ray.sub starts the Ray cluster then idles.
# It creates $SLURM_SUBMIT_DIR/<jobid>-attach.sh which supports:
#   bash <jobid>-attach.sh              # interactive shell on head node
#   bash <jobid>-attach.sh 1            # interactive shell on worker 1
#   COMMAND='...' bash <jobid>-attach.sh # run command non-interactively
#
# We save the training command to <jobid>-run-cmd.sh so the user can:
#   1. Attach interactively and source/paste it
#   2. Run non-interactively: COMMAND="$(cat <jobid>-run-cmd.sh)" bash <jobid>-attach.sh
#   3. Edit and re-run without requeueing
#
# A background watcher auto-runs the training command as soon as Ray is ready,
# so the scheduler never preempts the job for idle GPUs. After training finishes
# the allocation stays alive — re-attach and iterate without requeueing.
# =============================================================================
if [[ "${INTERACTIVE}" == "1" ]]; then
  # Ensure COMMAND is not in the environment. ray.sub does COMMAND=${COMMAND:-}
  # so unset → empty string → idle mode (creates attach script, sleeps forever).
  unset COMMAND 2>/dev/null || true

  # Interactive allocations default to 1h; INTERACTIVE_WALLTIME overrides.
  WALLTIME="${INTERACTIVE_WALLTIME:-4:0:0}"

  echo ""
  echo "================================================================"
  echo "  INTERACTIVE MODE"
  echo "================================================================"
  echo "  Submitting ${NUM_TOTAL_NODES}-node allocation (walltime: ${WALLTIME})"
  echo "  Ray cluster will start; training auto-runs when ready."
  echo ""

  submission_output=$(sbatch \
    --nodes="${NUM_TOTAL_NODES}" \
    --account="${SLURM_ACCOUNT}" \
    --job-name="interactive-${WANDB_NAME}" \
    --partition=batch \
    --time="${WALLTIME}" \
    --gres=gpu:4 \
    --exclusive \
    --mem=0 \
    ${SEGMENT_SIZE:+--segment="${SEGMENT_SIZE}"} \
    ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
    "${RAY_SUB}")

  echo "${submission_output}"

  if [[ "${submission_output}" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    JOB_ID="${BASH_REMATCH[1]}"
  else
    echo "ERROR: Could not parse job ID from sbatch output."
    exit 1
  fi

  # ray.sub writes the attach script to $SLURM_SUBMIT_DIR/<jobid>-attach.sh.
  # SLURM_SUBMIT_DIR is the cwd when sbatch was invoked, which is our $(pwd).
  LAUNCH_DIR="$(pwd)"
  ATTACH_SCRIPT="${LAUNCH_DIR}/${JOB_ID}-attach.sh"
  CMD_FILE="${LAUNCH_DIR}/${JOB_ID}-run-cmd.sh"

  # Save the training command. This file is intended to be:
  #   - Sourced from inside an interactive attach session, OR
  #   - Passed via: COMMAND="$(cat <file>)" bash <jobid>-attach.sh
  cat > "${CMD_FILE}" <<CMDEOF
${TRAIN_CMD}
CMDEOF
  chmod +x "${CMD_FILE}"

  # -----------------------------------------------------------------
  # Background watcher — auto-runs training so GPUs are never idle
  # waiting for a human to type the first command.
  # Polls for the attach script, then fires the training command.
  # After training finishes the allocation stays alive (ray.sub idles)
  # so the user can re-attach and iterate.
  # -----------------------------------------------------------------
  WATCHER_LOG="${LAUNCH_DIR}/${JOB_ID}-watcher.log"

  nohup bash -c '
    set -euo pipefail
    ATTACH_SCRIPT="'"${ATTACH_SCRIPT}"'"
    CMD_FILE="'"${CMD_FILE}"'"
    JOB_ID="'"${JOB_ID}"'"

    echo "[$(date)] Watcher started for job ${JOB_ID}"
    echo "[$(date)] Polling for attach script: ${ATTACH_SCRIPT}"

    while [[ ! -f "${ATTACH_SCRIPT}" ]]; do
      state=$(squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null || true)
      if [[ -z "${state}" ]]; then
        echo "[$(date)] Job ${JOB_ID} is no longer in the queue. Exiting watcher."
        exit 1
      fi
      echo "[$(date)] Job state: ${state}"
      sleep 15
    done

    echo "[$(date)] Ray cluster ready. Auto-running training command..."
    COMMAND="$(cat "${CMD_FILE}")" bash "${ATTACH_SCRIPT}"
    rc=$?
    echo "[$(date)] Training command finished (exit code: ${rc})."
    echo "[$(date)] Allocation is still alive — re-attach with:"
    echo "  bash ${ATTACH_SCRIPT}"
  ' > "${WATCHER_LOG}" 2>&1 &

  WATCHER_PID=$!
  disown "${WATCHER_PID}"

  echo ""
  echo "  Saved training command to:"
  echo "    ${CMD_FILE}"
  echo ""
  echo "  Background watcher running (PID: ${WATCHER_PID})"
  echo "    Log: ${WATCHER_LOG}"
  echo "    tail -f ${WATCHER_LOG}"
  echo ""
  echo "  Training will auto-start when Ray is ready, even if you're away."
  echo ""
  echo "  After training finishes, the allocation stays alive. Re-attach with:"
  echo "    bash ${ATTACH_SCRIPT}"
  echo ""
  echo "  Between runs (clean up GPUs, clear caches, re-run):"
  echo "    python ${PROJECT_ROOT}/reset_ray_cluster.py"
  echo "    source ${CMD_FILE}"
  echo ""
  echo "  Edit the command and re-run without requeueing:"
  echo "    vim ${CMD_FILE}"
  echo "    source ${CMD_FILE}"
  echo ""
  echo "  Cancel: scancel ${JOB_ID}"
  echo "  Kill watcher: kill ${WATCHER_PID}"

  if [[ "${INTERACTIVE_WAIT}" == "1" ]]; then
    echo ""
    echo "  Also waiting in foreground (Ctrl+C is safe — watcher continues)..."
    echo ""

    # Foreground poll — purely for UX. The watcher handles the real work.
    prev_state=""
    while [[ ! -f "${ATTACH_SCRIPT}" ]]; do
      state=$(squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null || true)
      if [[ -z "${state}" ]]; then
        echo "  Job ${JOB_ID} is no longer in the queue. Check: sacct -j ${JOB_ID}"
        echo "  (Watcher may have already handled this — check ${WATCHER_LOG})"
        exit 1
      fi
      if [[ "${state}" != "${prev_state}" ]]; then
        echo "  [$(date +%H:%M:%S)] Job state: ${state}"
        prev_state="${state}"
      fi
      sleep 15
    done

    echo ""
    echo "  Ray cluster is ready! Watcher is auto-running the training command."
    echo "  You can attach to monitor:"
    echo "    bash ${ATTACH_SCRIPT}"
    echo "    tail -f ${WATCHER_LOG}"
    echo ""
  fi

  exit 0
fi

# =============================================================================
# Batch mode — set COMMAND and submit
# =============================================================================
export COMMAND="${TRAIN_CMD}"

sbatch \
  --nodes="${NUM_TOTAL_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --job-name="${WANDB_NAME}" \
  --partition="${PARTITION}" \
  --time="${WALLTIME}" \
  --gres=gpu:4 \
  --exclusive \
  --mem=0 \
  --dependency=singleton \
  ${SEGMENT_SIZE:+--segment="${SEGMENT_SIZE}"} \
  ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
  "${RAY_SUB}"
