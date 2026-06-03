#!/bin/bash
set -euo pipefail

# =============================================================================
# ultra_launch.sh
#
# Public launcher for Nemotron 3 Ultra post-training stages on a SLURM cluster.
#
# Each training stage (Student RLVR, teacher RLVR/RLHF stages, MOPD) has a
# matching YAML config under examples/configs/ultra/. The stage-specific
# hyperparameters (batch size, advantage clip, MoE parallelism, etc.) live
# in the YAML; this launcher only handles orchestration: SLURM submission,
# code snapshotting, persistent cache management, container mounts, and the
# Hydra overrides that vary per run (data paths, model checkpoint, judge
# endpoints, log directories).
#
# Usage:
#
#   EXP_NAME=ultra-student-rlvr-001 \
#   CONFIG_PATH=examples/configs/ultra/student_rlvr.yaml \
#   MODEL_PATH=/path/to/sft_checkpoint \
#   TRAIN_PATH=/path/to/train.jsonl \
#   VAL_PATH=/path/to/val.jsonl \
#   CONTAINER=nvcr.io/nvidia/nemo-rl:<tag> \
#   SANDBOX_CONTAINER=/path/to/nemo-skills-sandbox.sqsh \
#   PERSISTENT_CACHE=/path/to/persistent/cache \
#   SLURM_PARTITION=batch \
#   SLURM_ACCOUNT=your_account \
#   GENRM_MODEL=<HF repo id or local path>      # Or set GENRM_BASE_URL to use a remote service
#   NL2BASH_JUDGE_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
#   SAFETY_JUDGE_MODEL=/path/to/safety_checkpoint \
#   bash ultra_launch.sh
#
# Optional knobs:
#   WALLTIME=4:00:00                       Slurm --time
#   SLURM_QOS=normal                       Slurm --qos
#   SLURM_RESERVATION=                     Slurm --reservation
#   SLURM_DEPENDENCY=                      Extra Slurm dependency, merged with
#                                          singleton (e.g. afterany:<jobid>)
#   EXCLUDE_NODES=                         Slurm --exclude
#   NUM_TRAIN_NODES=64                     Training (Megatron) nodes
#   NUM_GEN_NODES=172                      vLLM generation nodes
#   NUM_GYM_NODES=20                       NeMo Gym (judge) nodes
#   ENABLE_MTP_INFERENCE=0                 1 to enable MTP speculative decoding
#   NUM_SPECULATIVE_TOKENS=5               MTP speculative tokens
#   MAX_NUM_BATCHED_TOKENS=8480            vLLM max batched tokens (MTP)
#   NRL_MAX_STEPS=                         Override grpo.max_num_steps
#   EXTRA_MOUNTS=                          Comma-separated host:container pairs
#   USE_SNAPSHOT=1                         Snapshot source tree at submission
#   DRY_RUN=0                              1 to print TRAIN_CMD and exit
#   INTERACTIVE=0                          1 to bring up Ray and idle for attach
#                                          (no training driver) for debugging
#   INTERACTIVE_WAIT=1                     0 to submit and return immediately
#   INTERACTIVE_WALLTIME=                  override WALLTIME for the interactive alloc
#   HF_HOME=                               HuggingFace cache root (recommended)
#   HF_TOKEN=                              HuggingFace API token
#   WANDB_API_KEY=                         Weights & Biases API key
#   WANDB_PROJ=nemotron-3-ultra            W&B project
#   WANDB_ENTITY=                          W&B entity
#
# Hydra overrides are forwarded verbatim as positional arguments:
#   bash ultra_launch.sh policy.megatron_cfg.optimizer.lr=1e-6 grpo.val_period=50
#
# GB200 NVL72 nodes have 4 GPUs each. SLURM total = NUM_TRAIN + NUM_GEN + NUM_GYM
# and must be a multiple of SEGMENT_SIZE (default 16, one NVLink domain group).
# =============================================================================

# =============================================================================
# Required environment
# =============================================================================
: "${EXP_NAME:?EXP_NAME is required (used for job name, W&B run, checkpoint/log dirs)}"
: "${CONFIG_PATH:?CONFIG_PATH is required (e.g. examples/configs/ultra/student_rlvr.yaml)}"
: "${MODEL_PATH:?MODEL_PATH is required (initial policy checkpoint, HF repo id or local path)}"
: "${TRAIN_PATH:?TRAIN_PATH is required (training data jsonl path)}"
: "${VAL_PATH:?VAL_PATH is required (validation data jsonl path)}"
: "${CONTAINER:?CONTAINER is required (NGC image URI or .sqsh path)}"
: "${SANDBOX_CONTAINER:?SANDBOX_CONTAINER is required (nemo-skills sandbox image)}"
: "${PERSISTENT_CACHE:?PERSISTENT_CACHE is required (Lustre dir for vLLM/Triton/Inductor caches)}"
: "${SLURM_PARTITION:?SLURM_PARTITION is required}"
: "${SLURM_ACCOUNT:?SLURM_ACCOUNT is required}"
# Judge models are recipe-specific. Most teachers (student RLVR, IFBench, RLHF,
# Reasoning) need all three (GenRM, NL2Bash, Safety). The SWE teacher uses
# code-execution rewards and needs none of them. Set per recipe; unset vars
# skip the corresponding override.
NL2BASH_JUDGE_MODEL="${NL2BASH_JUDGE_MODEL:-}"
SAFETY_JUDGE_MODEL="${SAFETY_JUDGE_MODEL:-}"
GENRM_BASE_URL="${GENRM_BASE_URL:-}"
GENRM_MODEL="${GENRM_MODEL:-}"
GENRM_OVERRIDE=""
if [[ -n "${GENRM_BASE_URL}" ]]; then
  GENRM_OVERRIDE="env.nemo_gym.genrm_model.responses_api_models.genrm_model.base_url=${GENRM_BASE_URL}"
elif [[ -n "${GENRM_MODEL}" ]]; then
  GENRM_OVERRIDE="env.nemo_gym.genrm_model.responses_api_models.genrm_model.model=${GENRM_MODEL}"
fi

# SIF_DIR: for the SWE teacher recipe — directory containing apptainer .sif
# images for SWE-Bench / SWE-Gym / R2E-Gym instances. The yaml's
# container_formatter uses `${sif_dir}/...` paths. Unset for non-SWE recipes.
SIF_DIR="${SIF_DIR:-}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: CONFIG_PATH does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi

# =============================================================================
# Project root and code root
# =============================================================================
PROJECT_ROOT=$(realpath "$PWD")
cd "${PROJECT_ROOT}"

# =============================================================================
# Job identity — fixed name for singleton.
# Slurm --dependency=singleton serialises queued submissions with the same name
# so a resubmission after preemption resumes from the latest checkpoint instead
# of running in parallel.
# =============================================================================
JOB_NAME="${EXP_NAME}"

# =============================================================================
# Output directories
# =============================================================================
RESULTS_DIR="${RESULTS_DIR:-results/${EXP_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RESULTS_DIR}/checkpoints}"

# Per-submission dirs for logs and Slurm output (timestamped for history).
RUN_DIR="${RESULTS_DIR}/runs/$(date +%Y%m%d-%H%M)"
LOG_DIR="${RUN_DIR}/logs"
SLURM_LOG_DIR="${RUN_DIR}/slurm"
mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${SLURM_LOG_DIR}"
ln -sfn "$(realpath "${RUN_DIR}")" "${RESULTS_DIR}/runs/latest"

# ray.sub reads BASE_LOG_DIR and creates $BASE_LOG_DIR/$SLURM_JOB_ID-logs/ for
# ray infrastructure logs (ray-head.log, ray-driver.log, ray-worker-*.log,
# topology probes, attach scripts, etc.).
export BASE_LOG_DIR="${BASE_LOG_DIR:-${RESULTS_DIR}/ray_logs}"

# =============================================================================
# SLURM configuration
# =============================================================================
WALLTIME="${WALLTIME:-4:00:00}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_RESERVATION="${SLURM_RESERVATION:-}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
# INTERACTIVE=1 brings up the Ray cluster and idles for attachment (no training
# driver), so you can run/debug the recipe by hand. INTERACTIVE_WAIT=1 (default)
# blocks until Ray is ready; INTERACTIVE_WALLTIME overrides WALLTIME for the alloc.
INTERACTIVE="${INTERACTIVE:-0}"
INTERACTIVE_WAIT="${INTERACTIVE_WAIT:-1}"
# If set (format DD:HH:MM:SS), training stops early to reserve time for a final
# checkpoint save before walltime. Unset to use the YAML's default and let
# slurm walltime end the job naturally — fine when each step checkpoints.
CHECKPOINTING_SAVE_BY="${CHECKPOINTING_SAVE_BY:-}"

# =============================================================================
# Container & mounts
# =============================================================================
export CONTAINER
MOUNTS="${MOUNTS:-}"

# GB200 NVL72: fixed at 4 GPUs/node.
export GPUS_PER_NODE=4
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-144}"

# =============================================================================
# HuggingFace configuration
# =============================================================================
if [[ -n "${HF_HOME:-}" ]]; then
  export HF_HOME
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/hub}"
else
  echo "[WARN] HF_HOME is not set — HuggingFace will use the default cache (~/.cache/huggingface) per-node." >&2
fi

# =============================================================================
# W&B configuration
# =============================================================================
WANDB_PROJ="${WANDB_PROJ:-nemotron-3-ultra}"
WANDB_NAME="${EXP_NAME}"
WANDB_ENABLED=False
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
  WANDB_ENABLED=True
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    export WANDB_ENTITY
  fi
else
  echo "[WARN] WANDB_API_KEY is not set — W&B logging will be disabled." >&2
fi

# =============================================================================
# Training overrides
# =============================================================================
NRL_MAX_STEPS="${NRL_MAX_STEPS:-}"

# =============================================================================
# MTP speculative decoding (optional)
# =============================================================================
ENABLE_MTP_INFERENCE="${ENABLE_MTP_INFERENCE:-0}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8480}"
MTP_EXTRA_ARGS=""
if [[ "${ENABLE_MTP_INFERENCE}" == "1" ]]; then
  MTP_EXTRA_ARGS="\
++policy.generation.vllm_cfg.enable_prefix_caching=true \
++policy.generation.vllm_kwargs.enable_chunked_prefill=true \
++policy.generation.vllm_kwargs.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
++policy.generation.vllm_kwargs.mamba_cache_mode=align \
~policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes \
++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${NUM_SPECULATIVE_TOKENS} \
++policy.generation.vllm_kwargs.speculative_config.method=mtp"
  echo "MTP speculative decoding ENABLED (num_speculative_tokens=${NUM_SPECULATIVE_TOKENS})"
fi

# =============================================================================
# Job shape — defaults match the 256-node student_rlvr.yaml
#
#   Training:  64 nodes ( 256 GPUs) — Megatron training backend
#   vLLM:     172 nodes ( 688 GPUs) — async generation, EP=8 instances at TP=8
#   Gym:       20 nodes (  80 GPUs) — judges (GenRM, NL2Bash, Safety)
#
# Override via NUM_TRAIN_NODES / NUM_GEN_NODES / NUM_GYM_NODES.
#
# For STAGE_TYPE=mopd, additional teacher nodes are allocated for the
# non-colocated teacher panel: NUM_UNIQUE_TEACHERS × NUM_NODES_PER_TEACHER.
# =============================================================================
NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-64}"
NUM_GEN_NODES="${NUM_GEN_NODES:-172}"
NUM_GYM_NODES="${NUM_GYM_NODES:-20}"

STAGE_TYPE="${STAGE_TYPE:-grpo}"
NUM_TEACHER_NODES=0
MOPD_OVERRIDES=""
if [[ "${STAGE_TYPE}" == "mopd" ]]; then
  : "${NRL_GENERAL_TEACHER_PATH:?NRL_GENERAL_TEACHER_PATH is required for STAGE_TYPE=mopd (path to the Student RLVR output checkpoint)}"
  NUM_UNIQUE_TEACHERS="${NUM_UNIQUE_TEACHERS:-5}"
  NUM_NODES_PER_TEACHER="${NUM_NODES_PER_TEACHER:-4}"
  NUM_TEACHER_NODES=$((NUM_UNIQUE_TEACHERS * NUM_NODES_PER_TEACHER))
  TEACHER_TP="${TEACHER_TP:-8}"
  TEACHER_CP="${TEACHER_CP:-2}"
  TEACHER_PP="${TEACHER_PP:-1}"
  TEACHER_EP="${TEACHER_EP:-16}"

  # _teachers.general is required; other slots fall back via the YAML's
  # interpolation. Pass only the slots the user explicitly set.
  MOPD_OVERRIDES="_teachers.general=${NRL_GENERAL_TEACHER_PATH}"
  for _slot in RLHF IFBENCH REASONING SWE; do
    _var="NRL_${_slot}_TEACHER_PATH"
    _val="${!_var:-}"
    if [[ -n "${_val}" ]]; then
      MOPD_OVERRIDES="${MOPD_OVERRIDES} _teachers.$(echo ${_slot} | tr A-Z a-z)=${_val}"
    fi
  done

  # Teacher parallelism + per-teacher node count
  MOPD_OVERRIDES="${MOPD_OVERRIDES} \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.tensor_model_parallel_size=${TEACHER_TP} \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.context_parallel_size=${TEACHER_CP} \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.pipeline_model_parallel_size=${TEACHER_PP} \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.expert_model_parallel_size=${TEACHER_EP} \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.num_nodes=${NUM_NODES_PER_TEACHER}"

  echo "MOPD: ${NUM_UNIQUE_TEACHERS} teacher pools × ${NUM_NODES_PER_TEACHER} nodes = ${NUM_TEACHER_NODES} teacher nodes"
fi

NUM_ACTOR_NODES=$((NUM_TRAIN_NODES + NUM_GEN_NODES + NUM_TEACHER_NODES))
NUM_TOTAL_NODES=$((NUM_ACTOR_NODES + NUM_GYM_NODES))

if (( NUM_TRAIN_NODES <= 0 )); then
  echo "ERROR: NUM_TRAIN_NODES must be > 0 (got ${NUM_TRAIN_NODES})" >&2; exit 1
fi
if (( NUM_GEN_NODES <= 0 )); then
  echo "ERROR: NUM_GEN_NODES must be > 0 (got ${NUM_GEN_NODES})" >&2; exit 1
fi
if (( NUM_GYM_NODES < 0 )); then
  echo "ERROR: NUM_GYM_NODES must be >= 0 (got ${NUM_GYM_NODES})" >&2; exit 1
fi

# GB200 NVL72 topology: 18 nodes per NVLink domain, allocate in groups of 16.
SEGMENT_SIZE="${SEGMENT_SIZE:-16}"
if (( NUM_TOTAL_NODES < SEGMENT_SIZE )); then
  echo "ERROR: NUM_TOTAL_NODES=${NUM_TOTAL_NODES} < SEGMENT_SIZE=${SEGMENT_SIZE}" >&2
  exit 1
fi
if (( NUM_TOTAL_NODES % SEGMENT_SIZE != 0 )); then
  echo "ERROR: NUM_TOTAL_NODES=${NUM_TOTAL_NODES} is not divisible by SEGMENT_SIZE=${SEGMENT_SIZE}." >&2
  echo "  Training=${NUM_TRAIN_NODES} + Generation=${NUM_GEN_NODES} + Gym=${NUM_GYM_NODES} + Teachers=${NUM_TEACHER_NODES} = ${NUM_TOTAL_NODES}" >&2
  echo "  Adjust node counts so the total is a multiple of ${SEGMENT_SIZE}." >&2
  exit 1
fi

# =============================================================================
# NeMo Skills sandbox (for math_formal_lean, ns_tools, etc.)
# =============================================================================
export SANDBOX_CONTAINER
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-/start-with-nginx.sh}"
export NEMO_SKILLS_SANDBOX_PORT="${NEMO_SKILLS_SANDBOX_PORT:-6000}"

# =============================================================================
# Ray log sync
# =============================================================================
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-60}"

CODE_ROOT="/opt/nemo-rl"
VLLM_ENV_SOURCE="source /opt/nemo-rl/3rdparty/vllm/nemo-rl.env && "

# =============================================================================
# Persistent cache directories
# =============================================================================
# Lustre holds the warm persistent cache. At job start, SETUP_COMMAND clears
# stale /tmp caches then seeds node-local /tmp from Lustre. JIT writes go to
# /tmp to avoid Lustre metadata contention from parallel compilation.
_vllm_cache_precision="bf16"
CACHE_READ_DIR="${PERSISTENT_CACHE}/cache_read"
CACHE_WRITE_DIR="${PERSISTENT_CACHE}/cache_write"
LUSTRE_VLLM_CACHE="${CACHE_WRITE_DIR}/vllm_compile_cache_${_vllm_cache_precision}"
LUSTRE_FLASHINFER_CUBIN_CACHE="${PERSISTENT_CACHE}/flashinfer_cubins"
FLASHINFER_CUBIN_CACHE="/tmp/nemo_rl_flashinfer_cubins"
FLASHINFER_WS_BASE="${PERSISTENT_CACHE}/flashinfer_workspace"
LUSTRE_INDUCTOR_CACHE="${PERSISTENT_CACHE}/inductor_cache"
LUSTRE_TRITON_CACHE="${PERSISTENT_CACHE}/triton_cache"
NRL_VLLM_LOCAL_CACHE_DIR="/tmp/nemo_rl_vllm_cache"
NRL_VLLM_CACHE_SEED_DIR="/tmp/nemo_rl_vllm_cache_warm"
INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
CACHE_SYNC_FREQUENCY="${CACHE_SYNC_FREQUENCY:-0}"

export LUSTRE_VLLM_CACHE
export LUSTRE_INDUCTOR_CACHE
export LUSTRE_TRITON_CACHE
export CACHE_READ_DIR
export CACHE_WRITE_DIR
export NRL_VLLM_LOCAL_CACHE_DIR
export INDUCTOR_CACHE_DIR
export TRITON_CACHE_DIR
export CACHE_SYNC_FREQUENCY

mkdir -p "${LUSTRE_FLASHINFER_CUBIN_CACHE}" "${FLASHINFER_WS_BASE}" \
  "${LUSTRE_INDUCTOR_CACHE}" "${LUSTRE_TRITON_CACHE}" \
  "${CACHE_READ_DIR}" "${CACHE_WRITE_DIR}"

# Read path  : cache_read/*.tar.zst   — compute nodes extract tarballs (hundreds of concurrent reads)
# Write path : cache_write/*/        — sidecar rsyncs individual files (one sequential writer)
# Splitting reads (tarball) from writes (directory) avoids Lustre MDT invalidation storms
# and lets rsync accumulate the union of all roles' kernels across jobs.
for _name in inductor_cache triton_cache; do
  _write_dir="${CACHE_WRITE_DIR}/${_name}"
  _old_dir="${PERSISTENT_CACHE}/${_name}"

  # One-time migration: move legacy dir → cache_write/ (instant rename, same FS)
  if ([ ! -d "$_write_dir" ] || [ -z "$(ls -A "$_write_dir" 2>/dev/null)" ]) \
     && [ -d "$_old_dir" ] && [ -n "$(ls -A "$_old_dir" 2>/dev/null)" ]; then
    [ -d "$_write_dir" ] && rmdir "$_write_dir" 2>/dev/null
    mv "$_old_dir" "$_write_dir" 2>/dev/null \
      && echo "[CACHE] Moved legacy ${_name}/ → cache_write/${_name}/" \
      || echo "[CACHE] Failed to move legacy ${_name}/"
  fi
done

# vLLM: migrate the most recent legacy seed dir → cache_write/ (one-time, instant rename)
_vllm_write="${CACHE_WRITE_DIR}/vllm_compile_cache_${_vllm_cache_precision}"
_vllm_read_tar="${CACHE_READ_DIR}/vllm_compile_cache_${_vllm_cache_precision}.tar.zst"

if [ ! -d "$_vllm_write" ] || [ -z "$(ls -A "$_vllm_write" 2>/dev/null)" ]; then
  _best="$(ls -1dt \
      "${PERSISTENT_CACHE}/vllm_compile_cache_${_vllm_cache_precision}" \
      "${PERSISTENT_CACHE}/vllm_compile_cache_${_vllm_cache_precision}_"* \
    2>/dev/null \
    | while IFS= read -r d; do
        [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ] && echo "$d" && break
      done
  )" || true
  if [ -n "$_best" ]; then
    [ -d "$_vllm_write" ] && rmdir "$_vllm_write" 2>/dev/null || true
    mv "$_best" "$_vllm_write" 2>/dev/null \
      && echo "[CACHE] Moved $(basename "$_best") → cache_write/vllm_compile_cache_${_vllm_cache_precision}/" \
      || echo "[CACHE] Failed to move vLLM cache"
  fi
fi

# Purge redundant legacy vLLM cache directories.
# The old sidecar wrote every vLLM seed as a separate directory on Lustre
# (e.g. vllm_compile_cache_bf16_2058, _3072, ...). With cache_write/ + tarball,
# only cache_write/vllm_compile_cache_{precision}/ matters. All seed copies are
# content-addressed duplicates — safe to remove after migration.
_purge_count=0
for _d in "${PERSISTENT_CACHE}/vllm_compile_cache_${_vllm_cache_precision}" \
          "${PERSISTENT_CACHE}/vllm_compile_cache_${_vllm_cache_precision}_"*; do
  [ -d "$_d" ] || continue
  rm -rf "$_d" 2>/dev/null && (( _purge_count++ )) || true
done
for _d in "${PERSISTENT_CACHE}"/vllm_compile_cache_[0-9]*/; do
  [ -d "$_d" ] || continue
  rm -rf "$_d" 2>/dev/null && (( _purge_count++ )) || true
done
for _d in "${PERSISTENT_CACHE}/vllm_compile_cache" \
          "${PERSISTENT_CACHE}/vllm_compile_cache_warm"; do
  [ -d "$_d" ] || continue
  rm -rf "$_d" 2>/dev/null && (( _purge_count++ )) || true
done
if (( _purge_count > 0 )); then
  echo "[CACHE] Purged ${_purge_count} redundant legacy vLLM cache directories from ${PERSISTENT_CACHE}/"
fi

# =============================================================================
# Code snapshot
# =============================================================================
# Snapshot the git-tracked source tree so the code is frozen at submission time.
# This guarantees we know exactly which code was used for a given experiment.
# Set USE_SNAPSHOT=0 to skip (runs from container built-in or live checkout).
# Interactive mode defaults to the live checkout for fast iteration; batch snapshots.
if [[ "${INTERACTIVE}" == "1" ]]; then
  USE_SNAPSHOT="${USE_SNAPSHOT:-0}"
else
  USE_SNAPSHOT="${USE_SNAPSHOT:-1}"
fi

if [[ "${USE_SNAPSHOT}" == "1" ]]; then
  if [[ ! -f "${PROJECT_ROOT}/tools/code_snapshot.sh" ]]; then
    echo "ERROR: tools/code_snapshot.sh not found at ${PROJECT_ROOT}/tools/code_snapshot.sh" >&2
    echo "  Set USE_SNAPSHOT=0 to run from the live checkout instead." >&2
    exit 1
  fi
  SNAPSHOT_DIR=$(bash "${PROJECT_ROOT}/tools/code_snapshot.sh" "${JOB_NAME}")

  if [[ -d "${PROJECT_ROOT}/3rdparty/vllm" ]] && [[ ! -e "${SNAPSHOT_DIR}/3rdparty/vllm" ]]; then
    mkdir -p "${SNAPSHOT_DIR}/3rdparty"
    ln -s "${PROJECT_ROOT}/3rdparty/vllm" "${SNAPSHOT_DIR}/3rdparty/vllm"
  fi

  echo "Code snapshot: ${SNAPSHOT_DIR}"
  OVERLAY_SOURCE="${SNAPSHOT_DIR}"
else
  OVERLAY_SOURCE="${PROJECT_ROOT}"
fi

# =============================================================================
# Container mounts
# =============================================================================
# By default, nemo_rl (Python package) and examples/configs (YAML configs) from
# the code snapshot are overlaid into the container. Everything else uses the
# container's built-in code at /opt/nemo-rl.
#
# To overlay additional components (e.g. a local Megatron-LM checkout), pass
# EXTRA_MOUNTS as a comma-separated list of host:container pairs:
#
#   EXTRA_MOUNTS="/path/to/Megatron-LM:/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM" bash ultra_launch.sh
#
# Container paths for reference:
#   /opt/nemo-rl/nemo_rl                                              — Python package
#   /opt/nemo-rl/examples/configs                                     — YAML configs
#   /opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM           — Megatron-LM
#   /opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge   — Megatron-Bridge
#   /opt/nemo-rl/3rdparty/Gym-workspace/Gym                           — NeMo-Gym
#   /opt/nemo-rl/3rdparty/vllm                                        — vLLM
# =============================================================================
_append_mount() {
  if [[ -z "${MOUNTS}" ]]; then
    MOUNTS="$1"
  else
    MOUNTS="${MOUNTS},$1"
  fi
}

if [[ -d "${OVERLAY_SOURCE}/nemo_rl" ]]; then
  _append_mount "${OVERLAY_SOURCE}/nemo_rl:/opt/nemo-rl/nemo_rl"
  echo "  Mount: nemo_rl → /opt/nemo-rl/nemo_rl"
fi
if [[ -d "${OVERLAY_SOURCE}/examples/configs" ]]; then
  _append_mount "${OVERLAY_SOURCE}/examples/configs:/opt/nemo-rl/examples/configs"
  echo "  Mount: configs → /opt/nemo-rl/examples/configs"
fi
if [[ -d "${OVERLAY_SOURCE}/3rdparty/Gym-workspace/Gym" ]]; then
  _append_mount "${OVERLAY_SOURCE}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
  echo "  Mount: Gym → /opt/nemo-rl/3rdparty/Gym-workspace/Gym"
fi

if [[ "${USE_SNAPSHOT}" == "1" ]]; then
  _append_mount "${SNAPSHOT_DIR}:${SNAPSHOT_DIR}"
fi

if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
  _append_mount "${EXTRA_MOUNTS}"
  echo "  Extra mounts: ${EXTRA_MOUNTS}"
fi

export MOUNTS

# =============================================================================
# Resolve ray.sub
# =============================================================================
RAY_SUB="${RAY_SUB:-${PROJECT_ROOT}/ray.sub}"
if [[ ! -f "${RAY_SUB}" ]]; then
  echo "ERROR: ray.sub not found at ${RAY_SUB}" >&2
  exit 1
fi

# =============================================================================
# Per-node cache seeding (SETUP_COMMAND)
# =============================================================================
# Triton, Inductor, and FlashInfer cubins compile/download to node-local /tmp to
# avoid Lustre race conditions and file lock contention during concurrent JIT
# compilation. To avoid cold-start penalties, we seed /tmp from a warm Lustre
# cache before Ray starts.
#
# IMPORTANT: Stale /tmp caches from previous jobs can cause hangs (e.g. the
# Triton bundler skipping non-empty temp dirs). We rm -rf /tmp caches first,
# then seed fresh from Lustre.
# =============================================================================
read -r -d '' SETUP_COMMAND <<SETUPEOF || true
command -v zstd >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq zstd; } 2>/dev/null || true
echo "[CACHE SEED] Clearing stale /tmp caches and seeding from Lustre..."
WARM_SEED="${NRL_VLLM_CACHE_SEED_DIR}"
LOCAL_IND="${INDUCTOR_CACHE_DIR}"
LOCAL_TRI="${TRITON_CACHE_DIR}"
CACHE_READ="${CACHE_READ_DIR}"

# vLLM caches are per-instance (VLLM_CACHE_ROOT_{seed}). Clear ALL from prior jobs.
rm -rf /tmp/nemo_rl_vllm_cache /tmp/nemo_rl_vllm_cache_*
rm -rf "\$LOCAL_IND" "\$LOCAL_TRI"
mkdir -p "\$LOCAL_IND" "\$LOCAL_TRI"

_seed_cache() {
  local tarball="\$1" local_dir="\$2" name="\$3"
  if [ -f "\$tarball" ]; then
    tar --zstd -xf "\$tarball" -C "\$local_dir" \
      && echo "[CACHE SEED] \$name: seeded from tarball (\$(du -sh "\$local_dir" 2>/dev/null | cut -f1))" \
      || echo "[CACHE SEED] \$name: tarball extract failed (non-fatal)"
  else
    echo "[CACHE SEED] \$name: no warm cache on Lustre yet"
  fi
}

# Seed vLLM compile cache from cache_read/ tarball (one per precision).
rm -rf "\$WARM_SEED"
_vllm_tar="\$CACHE_READ/vllm_compile_cache_${_vllm_cache_precision}.tar.zst"
if [ -f "\$_vllm_tar" ]; then
  mkdir -p "\$WARM_SEED"
  tar --zstd -xf "\$_vllm_tar" -C "\$WARM_SEED" \
    && echo "[CACHE SEED] vLLM (${_vllm_cache_precision}): seeded from tarball (\$(du -sh "\$WARM_SEED" 2>/dev/null | cut -f1))" \
    || echo "[CACHE SEED] vLLM: tarball extract failed (non-fatal)"
else
  echo "[CACHE SEED] vLLM: no warm cache on Lustre yet"
fi

_seed_cache "\$CACHE_READ/inductor_cache.tar.zst" "\$LOCAL_IND" "Inductor"
_seed_cache "\$CACHE_READ/triton_cache.tar.zst" "\$LOCAL_TRI" "Triton"

echo "[CACHE SEED] Done."
SETUPEOF
export SETUP_COMMAND

# =============================================================================
# Build the training command
# =============================================================================
# Stage-specific hyperparameters (batch sizes, advantage clip, MoE parallelism,
# learning rate, etc.) live in CONFIG_PATH. The launcher only passes the
# per-run overrides: cluster shape, paths, judge endpoints, logging.
# =============================================================================
TRAIN_CMD="cd ${CODE_ROOT} && date ; \
${VLLM_ENV_SOURCE}\
OMP_NUM_THREADS=16 \
RAY_DEDUP_LOGS=1 \
WANDB_INIT_TIMEOUT=300 \
VLLM_CACHE_ROOT=${NRL_VLLM_LOCAL_CACHE_DIR} \
NRL_VLLM_CACHE_SEED_DIR=${NRL_VLLM_CACHE_SEED_DIR} \
DG_JIT_CACHE_DIR=${NRL_VLLM_LOCAL_CACHE_DIR}/deep_gemm \
TORCHINDUCTOR_CACHE_DIR=${INDUCTOR_CACHE_DIR} \
TRITON_CACHE_DIR=${TRITON_CACHE_DIR} \
UV_CACHE_DIR=${PERSISTENT_CACHE}/uv \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
UV_HTTP_TIMEOUT=10 \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
VLLM_FLASHINFER_MOE_BACKEND=latency \
NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800 \
NRL_WG_USE_RAY_REF=1 \
HF_HOME=${HF_HOME:-} \
HF_TOKEN=${HF_TOKEN:-} \
NRL_USE_FASTOKENS=${NRL_USE_FASTOKENS:-1} \
uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
--config ${CONFIG_PATH} \
policy.model_name=${MODEL_PATH} \
cluster.num_nodes=${NUM_ACTOR_NODES} \
policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
${CHECKPOINTING_SAVE_BY:+checkpointing.checkpoint_must_save_by=${CHECKPOINTING_SAVE_BY}} \
data.train.data_path=${TRAIN_PATH} \
data.validation.data_path=${VAL_PATH} \
${GENRM_OVERRIDE:+${GENRM_OVERRIDE}} \
${NL2BASH_JUDGE_MODEL:+env.nemo_gym.nl2bash_judge_model.responses_api_models.local_vllm_model.model=${NL2BASH_JUDGE_MODEL}} \
${SAFETY_JUDGE_MODEL:+env.nemo_gym.safety_judge_model.responses_api_models.local_vllm_model.model=${SAFETY_JUDGE_MODEL}} \
${SIF_DIR:+sif_dir=${SIF_DIR}} \
env.nemo_gym.nemo_gym_log_dir=${LOG_DIR}/nemo_gym \
logger.log_dir=${LOG_DIR} \
logger.wandb_enabled=${WANDB_ENABLED} \
logger.wandb.name=${WANDB_NAME} \
logger.wandb.project=${WANDB_PROJ} \
${NRL_MAX_STEPS:+grpo.max_num_steps=${NRL_MAX_STEPS}} \
${MTP_EXTRA_ARGS} \
${MOPD_OVERRIDES} \
${*}"

export COMMAND="${TRAIN_CMD}"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "================================================================"
echo "  Nemotron 3 Ultra — ${EXP_NAME} (${NUM_TOTAL_NODES}-node)"
echo "================================================================"
echo "  Job name:    ${JOB_NAME}  (singleton — only one runs at a time)"
echo "  Config:      ${CONFIG_PATH}"
echo "  Nodes:       ${NUM_TOTAL_NODES} total  (segment=${SEGMENT_SIZE})"
echo "    Training:  ${NUM_TRAIN_NODES}  ($((NUM_TRAIN_NODES * GPUS_PER_NODE)) GPUs)"
echo "    vLLM gen:  ${NUM_GEN_NODES}  ($((NUM_GEN_NODES * GPUS_PER_NODE)) GPUs)"
echo "    Gym:       ${NUM_GYM_NODES}  ($((NUM_GYM_NODES * GPUS_PER_NODE)) GPUs)"
if (( NUM_TEACHER_NODES > 0 )); then
echo "    Teachers:  ${NUM_TEACHER_NODES}  ($((NUM_TEACHER_NODES * GPUS_PER_NODE)) GPUs)"
fi
echo "  Walltime:    ${WALLTIME}"
echo ""
echo "  Checkpoints: ${CHECKPOINT_DIR}  (stable — auto-resumes across jobs)"
echo "  Run dir:     ${RUN_DIR}"
echo "  Logs:        ${LOG_DIR}"
echo "  Slurm logs:  ${SLURM_LOG_DIR}"
echo "  W&B:         ${WANDB_PROJ} / ${WANDB_NAME} (enabled=${WANDB_ENABLED})"
echo ""
echo "  Model:       ${MODEL_PATH}"
echo "  Train data:  ${TRAIN_PATH}"
echo "  Val data:    ${VAL_PATH}"
echo "  Container:   ${CONTAINER}"
echo "  Sandbox:     ${SANDBOX_CONTAINER}"
if [[ "${USE_SNAPSHOT}" == "1" ]]; then
echo "  Snapshot:    ${SNAPSHOT_DIR}"
fi
echo ""
echo "  Monitor:  squeue -u \$USER -n ${JOB_NAME}"
echo "  Logs:     tail -f ${SLURM_LOG_DIR}/*.out"
echo "  Latest:   ls -la ${RESULTS_DIR}/runs/latest"
echo ""
echo "================================================================"
echo ""

# =============================================================================
# Record code provenance in the run directory
# =============================================================================
{
  echo "timestamp: $(date -Iseconds)"
  echo "branch: $(git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "commit: $(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "dirty: $(git -C "${PROJECT_ROOT}" status --porcelain 2>/dev/null | head -20)"
  echo "snapshot: ${USE_SNAPSHOT}"
  if [[ "${USE_SNAPSHOT}" == "1" ]]; then
    echo "snapshot_dir: ${SNAPSHOT_DIR}"
  fi
  echo "container: ${CONTAINER}"
  echo "config: ${CONFIG_PATH}"
  echo "command: ${TRAIN_CMD}"
} > "${RUN_DIR}/provenance.txt"

# =============================================================================
# Dry-run mode: print everything, don't submit
# =============================================================================
DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 — printing TRAIN_CMD and exiting without submission."
  echo ""
  echo "--- TRAIN_CMD ---"
  echo "${TRAIN_CMD}"
  echo "--- end ---"
  exit 0
fi

# =============================================================================
# Interactive mode: bring up Ray and idle for attachment (no training driver)
# =============================================================================
# With COMMAND empty, ray.sub starts the Ray cluster, writes <jobid>-attach.sh,
# then idles. We save the driver command to <jobid>-run-cmd.sh so you can attach
# and run it by hand, edit it, and re-run without requeueing.
if [[ "${INTERACTIVE}" == "1" ]]; then
  unset COMMAND 2>/dev/null || true   # empty COMMAND -> ray.sub idle/interactive mode
  WALLTIME="${INTERACTIVE_WALLTIME:-${WALLTIME}}"

  echo ""
  echo "================================================================"
  echo "  INTERACTIVE MODE — ${NUM_TOTAL_NODES}-node allocation (walltime ${WALLTIME})"
  echo "  Ray will start and idle until you attach."
  echo "================================================================"

  SBATCH_OUTPUT=$(sbatch \
    --nodes="${NUM_TOTAL_NODES}" \
    --account="${SLURM_ACCOUNT}" \
    --job-name="interactive-${JOB_NAME}" \
    --partition="${SLURM_PARTITION}" \
    --time="${WALLTIME}" \
    --gres=gpu:${GPUS_PER_NODE} \
    --exclusive \
    --mem=0 \
    --segment="${SEGMENT_SIZE}" \
    --output="${SLURM_LOG_DIR}/%j.out" \
    --error="${SLURM_LOG_DIR}/%j.err" \
    ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
    ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
    ${SLURM_RESERVATION:+--reservation="${SLURM_RESERVATION}"} \
    "${RAY_SUB}")
  echo "${SBATCH_OUTPUT}"
  JOB_ID=$(echo "${SBATCH_OUTPUT}" | grep -oP '\d+$')
  [[ -z "${JOB_ID}" ]] && { echo "ERROR: could not parse job ID from sbatch output." >&2; exit 1; }

  LAUNCH_DIR="$(pwd)"
  ATTACH_SCRIPT="${LAUNCH_DIR}/${JOB_ID}-attach.sh"
  CMD_FILE="${LAUNCH_DIR}/${JOB_ID}-run-cmd.sh"
  cat > "${CMD_FILE}" <<CMDEOF
${TRAIN_CMD}
CMDEOF
  chmod +x "${CMD_FILE}"

  echo ""
  echo "  Driver command saved to:  ${CMD_FILE}"
  echo "  When Ray is up:"
  echo "    bash ${ATTACH_SCRIPT}                          # shell on the head node (Ray already up)"
  echo "    source ${CMD_FILE}                             # run the recipe inside that shell"
  echo "    # or non-interactively: COMMAND=\"\$(cat ${CMD_FILE})\" bash ${ATTACH_SCRIPT}"
  echo "  Edit ${CMD_FILE} and re-source to iterate without requeueing.  Cancel: scancel ${JOB_ID}"

  if [[ "${INTERACTIVE_WAIT}" == "1" ]]; then
    echo ""
    echo "  Waiting for Ray (Ctrl+C to stop waiting; the job keeps running)..."
    prev_state=""
    while [[ ! -f "${ATTACH_SCRIPT}" ]]; do
      state=$(squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null || true)
      [[ -z "${state}" ]] && { echo "  Job ${JOB_ID} left the queue. Check: sacct -j ${JOB_ID}"; exit 1; }
      [[ "${state}" != "${prev_state}" ]] && { echo "  [$(date +%H:%M:%S)] state: ${state}"; prev_state="${state}"; }
      sleep 15
    done
    echo ""
    echo "  Ray is ready — attach: bash ${ATTACH_SCRIPT}"
  fi
  exit 0
fi

# =============================================================================
# Submit
# =============================================================================
# Always serialise same-name submissions via singleton; optionally chain after
# another job with SLURM_DEPENDENCY (e.g. "afterany:3044848" or "afterok:JOBID").
SLURM_DEPENDENCY="${SLURM_DEPENDENCY:-}"
DEPENDENCY="singleton"
[[ -n "${SLURM_DEPENDENCY}" ]] && DEPENDENCY="singleton,${SLURM_DEPENDENCY}"

SBATCH_OUTPUT=$(sbatch \
  --nodes="${NUM_TOTAL_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --job-name="${JOB_NAME}" \
  --partition="${SLURM_PARTITION}" \
  --time="${WALLTIME}" \
  --gres=gpu:${GPUS_PER_NODE} \
  --exclusive \
  --mem=0 \
  --dependency="${DEPENDENCY}" \
  --segment="${SEGMENT_SIZE}" \
  --output="${SLURM_LOG_DIR}/%j.out" \
  --error="${SLURM_LOG_DIR}/%j.err" \
  ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
  ${SLURM_RESERVATION:+--reservation="${SLURM_RESERVATION}"} \
  "${RAY_SUB}")

echo "${SBATCH_OUTPUT}"
JOB_ID=$(echo "${SBATCH_OUTPUT}" | grep -oP '\d+$')

if [[ -n "${JOB_ID}" ]]; then
  echo ""
  echo "  Ray logs:    ${BASE_LOG_DIR}/${JOB_ID}-logs/"
  echo ""
fi
