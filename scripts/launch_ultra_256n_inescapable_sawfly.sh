#!/bin/bash
set -euo pipefail

# =============================================================================
# launch_ultra_256n_inescapable_sawfly.sh
#
# GRPO Ultra V3 — 256-node GB200 NVL72 inescapable-sawfly experiment with NeMo Gym
#
# Replicates jiaqiz's inescapable-sawfly 256-node setup.
#
# By default, this runs from what's built into the container without overlay mounts applied.
# Set USE_WORKTREE=1 to overlay your local worktree submodules for development.
# Set INTERACTIVE=1 to get a persistent allocation in slurm for iterative debugging.
#
# Usage:
#   ./launch_ultra_256n_inescapable_sawfly.sh
#   NRL_MAX_STEPS=4 ./launch_ultra_256n_inescapable_sawfly.sh
#   USE_WORKTREE=1 ./launch_ultra_256n_inescapable_sawfly.sh
#   WALLTIME=4:00:00 ./launch_ultra_256n_inescapable_sawfly.sh
#   DRY_RUN=1 ./launch_ultra_256n_inescapable_sawfly.sh
#
# Adjust node allocation:
#   NUM_TRAIN_NODES=32 NUM_GEN_NODES=80 NUM_GYM_NODES=16 ./launch_ultra_256n_inescapable_sawfly.sh
#
# Extra positional arguments are forwarded as Hydra overrides:
#   ./launch_ultra_256n_inescapable_sawfly.sh grpo.max_num_steps=2 policy.precision=float32
#
# Interactive debugging (reuse allocation across runs):
#   INTERACTIVE=1 ./launch_ultra_256n_inescapable_sawfly.sh
#   INTERACTIVE=1 INTERACTIVE_WAIT=0 ./launch_ultra_256n_inescapable_sawfly.sh
#   INTERACTIVE=1 INTERACTIVE_WALLTIME=2:0:0 SLURM_QOS=short ./launch_ultra_256n_inescapable_sawfly.sh
# =============================================================================
#
# Example(yifu):
# USE_SNAPSHOT=0 \
# SLURM_ACCOUNT=llmservice_nemotron_ultra \
# EXP_SUFFIX=ultra-v3-grpo-mtp5-prefcache-inescapable_sawfly \
# NRL_MEGATRON_LM_DIR=/lustre/fsw/portfolios/coreai/users/yifuw/code/ultra3/nemo-rl-internal/3rdparty/Megatron-LM-workspace/Megatron-LM/ \
# NRL_MEGATRON_BRIDGE_DIR=/lustre/fsw/portfolios/coreai/users/yifuw/code/ultra3/nemo-rl-internal/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/ \
# NRL_VLLM_DIR=/lustre/fsw/portfolios/coreai/users/yifuw/code/ultra3/nemo-rl-internal/3rdparty/vllm \
# PERSISTENT_CACHE=/lustre/fsw/portfolios/llmservice/users/yifuw/ultra_cache_mtp5_inescapable_sawfly \
# WALLTIME=24:00:00 \
# SLURM_QOS=normal \
# ./launch_ultra_256n_inescapable_sawfly.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)
cd "${PROJECT_ROOT}"

USE_WORKTREE="${USE_WORKTREE:-0}"
INTERACTIVE="${INTERACTIVE:-0}"
INTERACTIVE_WAIT="${INTERACTIVE_WAIT:-1}"

# ---------- Precision configuration ----------
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
policy.megatron_cfg.fp8_cfg.fp8=\"e4m3\" \
policy.megatron_cfg.fp8_cfg.fp8_recipe=\"mxfp8\" \
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
WALLTIME="${WALLTIME:-24:00:00}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

# ---------- Container & mounts ----------
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/nemo_rl/images/high_stripe/rl.nightly.sqsh}"
MOUNTS="/lustre:/lustre"

# GB200 NVL72: fixed at 4 GPUs/node. Must match --gres=gpu:4 passed to sbatch.
export GPUS_PER_NODE=4
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-144}"

# ---------- HuggingFace Configuration ----------
export HF_HOME="${HF_HOME:-}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-}"

# ---------- W&B Configuration ----------
WANDB_PROJ="${WANDB_PROJ:-ultra-v3-pipeclean}"
WANDB_ENTITY="${WANDB_ENTITY:-nvidia}"
WANDB_NAME="${WANDB_NAME:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY

# ---------- Training ----------
NRL_MAX_STEPS="${NRL_MAX_STEPS:-}"

# ---------- Model configuration ----------
TP="${TP:-8}"
CP="${CP:-8}"
EP="${EP:-64}"
PP="${PP:-1}"
GPP="${GPP:-16}"
PPS="${PPS:-256}"
GBS="${GBS:-4096}"
VAL_PERIOD="${VAL_PERIOD:-10000}"
SAVE_PERIOD="${SAVE_PERIOD:-6}"
ADVANTAGE_CLIP_LOW="${ADVANTAGE_CLIP_LOW:--20}"
ADVANTAGE_CLIP_HIGH="${ADVANTAGE_CLIP_HIGH:-20}"

# =============================================================================
# Job shape — 3-way split: Training, Generation (vLLM), Gym (judges)
#
#   Training:  128 nodes (512 GPUs)  — Megatron training backend
#   vLLM:      118 nodes (472 GPUs)  — 59 instances at TP=8
#   Gym:        10 nodes ( 40 GPUs)  — judges (GenRM, NL2Bash, Safety)
#
# =============================================================================
NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-128}"
NUM_GEN_NODES="${NUM_GEN_NODES:-118}"
NUM_GYM_NODES="${NUM_GYM_NODES:-10}"

NUM_TOTAL_NODES=$((NUM_TRAIN_NODES + NUM_GEN_NODES + NUM_GYM_NODES))

if (( NUM_TRAIN_NODES <= 0 )); then
  echo "ERROR: NUM_TRAIN_NODES must be > 0 (got ${NUM_TRAIN_NODES})" >&2; exit 1
fi
if (( NUM_GEN_NODES <= 0 )); then
  echo "ERROR: NUM_GEN_NODES must be > 0 (got ${NUM_GEN_NODES})" >&2; exit 1
fi
if (( NUM_GYM_NODES < 0 )); then
  echo "ERROR: NUM_GYM_NODES must be >= 0 (got ${NUM_GYM_NODES})" >&2; exit 1
fi

# GB200 NVL72: 18 nodes per NVLink domain, allocate in groups of 16.
SEGMENT_SIZE="${SEGMENT_SIZE:-}"
if [ -z "${SEGMENT_SIZE}" ] && [ "${NUM_TOTAL_NODES}" -ge 16 ]; then
  SEGMENT_SIZE=16
fi
if [ -n "${SEGMENT_SIZE}" ] && [ "${NUM_TOTAL_NODES}" -lt "${SEGMENT_SIZE}" ]; then
  echo "ERROR: NUM_TOTAL_NODES=${NUM_TOTAL_NODES} < SEGMENT_SIZE=${SEGMENT_SIZE}" >&2
  exit 1
fi
if [ -n "${SEGMENT_SIZE}" ] && (( NUM_TOTAL_NODES % SEGMENT_SIZE != 0 )); then
  echo "ERROR: NUM_TOTAL_NODES=${NUM_TOTAL_NODES} is not divisible by SEGMENT_SIZE=${SEGMENT_SIZE}." >&2
  echo "  Training=${NUM_TRAIN_NODES} + Generation=${NUM_GEN_NODES} + Gym=${NUM_GYM_NODES} = ${NUM_TOTAL_NODES}" >&2
  echo "  Adjust node counts so the total is a multiple of ${SEGMENT_SIZE}." >&2
  exit 1
fi

# ---------- Model and data paths ----------
NRL_TRAIN_PATH="${NRL_TRAIN_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/data/gym/rl-data-tools/blends/curriculum_v35_inescapable-sawfly.train.efforts0p15_qamathcode.jsonl}"
NRL_VAL_PATH="${NRL_VAL_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/data/gym/rl-data-tools/blends/curriculum_v35_inescapable-sawfly.train.efforts0p15_qamathcode.jsonl}"
NRL_MODEL_PATH="${NRL_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/wdai/megatron-lm-ultra/checkpoints/ultra-v3-sft-bf16-hybridep-ep64-cp32-bindpcie-recompute-offload-288k-nano-loss-032026/iter_0003200/hf}"
NRL_GENRM_MODEL_PATH="${NRL_GENRM_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/qwen235b_principle_comparison_genrm_step1230}"
NRL_NL2BASH_JUDGE_MODEL_PATH="${NRL_NL2BASH_JUDGE_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
NRL_SAFETY_MODEL_PATH="${NRL_SAFETY_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/super_v3/model_checkpoints/Nemotron-Content-Safety-Reasoning-4B}"

# ---------- Lean4 sandbox (for math_formal_lean) ----------
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-sandbox-latest.sqsh}"
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-/start-with-nginx.sh}"
export NEMO_SKILLS_SANDBOX_PORT="${NEMO_SKILLS_SANDBOX_PORT:-6000}"

# ---------- Ray log sync ----------
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-60}"

# ---------- Job identity ----------
JOB_PREFIX="${JOB_PREFIX:-pipeclean-ultra-rl}"
EXP_SUFFIX="${EXP_SUFFIX:-${JOB_PREFIX}-inescapable-sawfly_tp${TP}_cp${CP}_ep${EP}_pp${PP}_gpp${GPP}_pps${PPS}_gbs${GBS}}"
JOB_NAME="${EXP_SUFFIX}"

# ---------- Output directories ----------
export BASE_LOG_DIR="${BASE_LOG_DIR:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/nemo_rl/logs}"
export BASE_LOG_DIR="${BASE_LOG_DIR}/${EXP_SUFFIX}"

RESULTS_DIR="${RESULTS_DIR:-results/${EXP_SUFFIX}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RESULTS_DIR}}"
mkdir -p "${CHECKPOINT_DIR}"
CHECKPOINT_DIR="$(cd "${CHECKPOINT_DIR}" && pwd)"
RESULTS_DIR="$(cd "${RESULTS_DIR}" && pwd)"

# Per-submission dirs for logs and slurm output (timestamped for history).
RUN_DIR="${RESULTS_DIR}/runs/$(date +%Y%m%d-%H%M)"
LOG_DIR="${RUN_DIR}/logs"
SLURM_LOG_DIR="${RUN_DIR}/slurm"
mkdir -p "${LOG_DIR}" "${SLURM_LOG_DIR}"
ln -sfn "${RUN_DIR}" "${RESULTS_DIR}/runs/latest"

# W&B name defaults to EXP_SUFFIX if not set
WANDB_NAME="${WANDB_NAME:-${EXP_SUFFIX}}"

# ---------- Code snapshot ----------
if [[ "${INTERACTIVE}" == "1" ]]; then
  USE_SNAPSHOT="${USE_SNAPSHOT:-0}"
else
  USE_SNAPSHOT="${USE_SNAPSHOT:-1}"
fi

if [[ "${USE_SNAPSHOT}" == "1" ]]; then
  SNAPSHOT_DIR=$(bash "${PROJECT_ROOT}/tools/code_snapshot.sh" "${EXP_SUFFIX}")

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
if [[ -z "${PERSISTENT_CACHE:-}" ]]; then
  _access_group="${SLURM_ACCOUNT%%_*}"
  PERSISTENT_CACHE="/lustre/fsw/portfolios/${_access_group}/users/${USER}/.cache/nemotron_ultra"
fi
# bf16 and mxfp8 share torch_compile hash dirs but compile different subgraphs,
# so they MUST use separate Lustre trees to avoid seeding partial caches.
case "${PRECISION_RECIPE}" in
  mxfp8-rollout|mxfp8-e2e) _vllm_cache_precision="mxfp8" ;;
  *)                        _vllm_cache_precision="bf16"  ;;
esac
CACHE_READ_DIR="${PERSISTENT_CACHE}/cache_read"
CACHE_WRITE_DIR="${PERSISTENT_CACHE}/cache_write"
LUSTRE_VLLM_CACHE="${CACHE_WRITE_DIR}/vllm_compile_cache_${_vllm_cache_precision}"
LUSTRE_FLASHINFER_CUBIN_CACHE="${PERSISTENT_CACHE}/flashinfer_cubins"
FLASHINFER_CUBIN_CACHE="/tmp/nemo_rl_flashinfer_cubins"
FLASHINFER_WS_BASE="${PERSISTENT_CACHE}/flashinfer_workspace"
LUSTRE_INDUCTOR_CACHE="${PERSISTENT_CACHE}/inductor_cache"
LUSTRE_TRITON_CACHE="${PERSISTENT_CACHE}/triton_cache"
# Node-local (fast) — each vLLM instance writes to ${NRL_VLLM_LOCAL_CACHE_DIR}_{seed}.
# Set as VLLM_CACHE_ROOT so torch.compile hits local disk, not Lustre.
NRL_VLLM_LOCAL_CACHE_DIR="/tmp/nemo_rl_vllm_cache"
# Read-only warm seed — SETUP_COMMAND extracts the precision-scoped tarball here per-node.
# The framework rsyncs this into each instance's cache before compilation.
NRL_VLLM_CACHE_SEED_DIR="/tmp/nemo_rl_vllm_cache_warm"
INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
CACHE_SYNC_FREQUENCY="${CACHE_SYNC_FREQUENCY:-120}"

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
# Write path : cache_write/*/     — sidecar rsyncs individual files (one sequential writer)
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
# Current precision: base + all seed-suffixed (the best was already migrated above)
for _d in "${PERSISTENT_CACHE}/vllm_compile_cache_${_vllm_cache_precision}" \
          "${PERSISTENT_CACHE}/vllm_compile_cache_${_vllm_cache_precision}_"*; do
  [ -d "$_d" ] || continue
  rm -rf "$_d" 2>/dev/null && (( _purge_count++ )) || true
done
# Old-format dirs (pre-precision-fix): vllm_compile_cache_<number> with no bf16/mxfp8
for _d in "${PERSISTENT_CACHE}"/vllm_compile_cache_[0-9]*/; do
  [ -d "$_d" ] || continue
  rm -rf "$_d" 2>/dev/null && (( _purge_count++ )) || true
done
# Stale base (no precision suffix) and old warm seed dir
for _d in "${PERSISTENT_CACHE}/vllm_compile_cache" \
          "${PERSISTENT_CACHE}/vllm_compile_cache_warm"; do
  [ -d "$_d" ] || continue
  rm -rf "$_d" 2>/dev/null && (( _purge_count++ )) || true
done
if (( _purge_count > 0 )); then
  echo "[CACHE] Purged ${_purge_count} redundant legacy vLLM cache directories from ${PERSISTENT_CACHE}/"
fi

# Generate/refresh cache_read/ tarballs via srun (avoids slow tar/find on login node).
# Triggered when at least one tarball is missing. The srun script also refreshes
# stale tarballs while the compute node is allocated.
_missing_tarballs=()
for _tar_name in inductor_cache triton_cache "vllm_compile_cache_${_vllm_cache_precision}"; do
  _tar="${CACHE_READ_DIR}/${_tar_name}.tar.zst"
  _wd="${CACHE_WRITE_DIR}/${_tar_name}"
  if [ -d "$_wd" ] && [ -n "$(ls -A "$_wd" 2>/dev/null)" ] && [ ! -f "$_tar" ]; then
    _missing_tarballs+=("$_tar_name")
  fi
done

if (( ${#_missing_tarballs[@]} > 0 )); then
  echo "[CACHE] Missing tarballs: ${_missing_tarballs[*]}"
  echo "[CACHE] Generating via srun on a compute node..."
  _promo_script="${CACHE_WRITE_DIR}/.promote_tarballs_$$.sh"
  cat > "$_promo_script" <<'PROMOSCRIPT'
#!/bin/bash
set -euo pipefail
CACHE_READ_DIR="$1"; CACHE_WRITE_DIR="$2"; shift 2
for _tar_name in "$@"; do
  _read_tar="${CACHE_READ_DIR}/${_tar_name}.tar.zst"
  _write_dir="${CACHE_WRITE_DIR}/${_tar_name}"
  [ -d "$_write_dir" ] && [ -n "$(ls -A "$_write_dir" 2>/dev/null)" ] || continue
  _needs=0
  if [ ! -f "$_read_tar" ]; then
    _needs=1
  elif find "$_write_dir" -type f -newer "$_read_tar" -print -quit 2>/dev/null | grep -q .; then
    _needs=1
  fi
  if (( _needs )); then
    echo "Creating/refreshing ${_tar_name}.tar.zst..."
    tar --zstd -cf "${_read_tar}.tmp.$$" --blocking-factor=8192 -C "$_write_dir" --exclude='tmp*' --exclude='.tmp_*' --exclude='.*' . \
      && mv "${_read_tar}.tmp.$$" "$_read_tar" \
      && echo "Done: $(du -sh "$_read_tar" | cut -f1)" \
      || { rm -f "${_read_tar}.tmp.$$"; echo "Failed: ${_tar_name}"; }
  else
    echo "${_tar_name}: tarball up to date"
  fi
done
PROMOSCRIPT
  chmod +x "$_promo_script"
  srun -N1 -n1 -t 00:30:00 -A "${SLURM_ACCOUNT}" -p cpu \
    -q cpu-normal \
    bash "$_promo_script" "${CACHE_READ_DIR}" "${CACHE_WRITE_DIR}" \
      inductor_cache triton_cache "vllm_compile_cache_${_vllm_cache_precision}" \
    && echo "[CACHE] srun tarball generation complete" \
    || echo "[CACHE] srun tarball generation failed (non-fatal, first job will compile from scratch)"
  rm -f "$_promo_script"
fi

VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://github.com/vllm-project/vllm/releases/download/v0.17.0/vllm-0.17.0-cp38-abi3-manylinux_2_31_aarch64.whl}"

# =============================================================================
# Validation
# =============================================================================
_walltime_secs() {
  local t="$1" h m s
  IFS=: read -r h m s <<< "${t}"
  echo $(( 10#${h} * 3600 + 10#${m} * 60 + 10#${s} ))
}

if (( $(_walltime_secs "${WALLTIME}") > 24 * 3600 )); then
  echo "ERROR: WALLTIME=${WALLTIME} exceeds the 24-hour maximum."
  exit 1
fi

if [[ "${SLURM_QOS}" == "interactive" ]]; then
  if (( $(_walltime_secs "${INTERACTIVE_WALLTIME:-${WALLTIME}}") > 2 * 3600 )); then
    echo "ERROR: SLURM_QOS=interactive requires walltime <= 2 hours."
    echo "  Set INTERACTIVE_WALLTIME=2:0:0 or use a different QOS (e.g. SLURM_QOS=short)."
    exit 1
  fi
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARNING: WANDB_API_KEY is not set. W&B logging will fail or fall back to offline mode."
  echo "  export WANDB_API_KEY=<your-key> to enable cloud logging."
fi

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
if [[ "${USE_WORKTREE}" == "1" ]]; then
  CODE_ROOT="${WORKTREE_ROOT}"
  VLLM_ENV_SOURCE="source ${MAIN_REPO_ROOT}/3rdparty/vllm/nemo-rl.env && "
else
  CODE_ROOT="/opt/nemo-rl"
  VLLM_ENV_SOURCE="source /opt/nemo-rl/3rdparty/vllm/nemo-rl.env && "
fi

echo ""
echo "================================================================"
echo "  GRPO Ultra V3 — ${NUM_TOTAL_NODES}-node inescapable-sawfly"
echo "================================================================"
echo "  Job name:    ${JOB_NAME}  (singleton — only one runs at a time)"
echo "  Nodes:       ${NUM_TOTAL_NODES} total  (segment=${SEGMENT_SIZE:-none})"
echo "    Training:  ${NUM_TRAIN_NODES}  ($((NUM_TRAIN_NODES * GPUS_PER_NODE)) GPUs)"
echo "    vLLM gen:  ${NUM_GEN_NODES}  ($((NUM_GEN_NODES * GPUS_PER_NODE)) GPUs)"
echo "    Gym:       ${NUM_GYM_NODES}  ($((NUM_GYM_NODES * GPUS_PER_NODE)) GPUs)"
echo "  Walltime:    ${WALLTIME}"
echo ""
echo "  Checkpoints: ${CHECKPOINT_DIR}"
echo "  Run dir:     ${RUN_DIR}"
echo "  Logs:        ${LOG_DIR}"
echo "  Slurm logs:  ${SLURM_LOG_DIR}"
echo "  W&B:         ${WANDB_ENTITY}/${WANDB_PROJ} / ${WANDB_NAME}"
echo ""
echo "  Model:       ${NRL_MODEL_PATH}"
echo "  Container:   ${CONTAINER}"
if [[ "${USE_SNAPSHOT}" == "1" ]]; then
echo "  Snapshot:    ${SNAPSHOT_DIR}"
fi
echo ""
echo "  Code root:   ${CODE_ROOT}"
echo "  Cache root:  ${PERSISTENT_CACHE}"
echo ""
echo "  Monitor:  squeue -u \$USER -n ${JOB_NAME}"
echo "  Logs:     tail -f ${SLURM_LOG_DIR}/*.out"
echo "  Latest:   ls -la ${RESULTS_DIR}/runs/latest"
echo ""
echo "================================================================"
echo ""

# =============================================================================
# Build the training command
# =============================================================================
TRAIN_CMD="cd ${CODE_ROOT} && date ; \
${VLLM_ENV_SOURCE}\
OMP_NUM_THREADS=16 \
RAY_DEDUP_LOGS=1 \
NRL_VLLM_USE_V1=1 \
VLLM_CACHE_ROOT=${NRL_VLLM_LOCAL_CACHE_DIR} \
NRL_VLLM_CACHE_SEED_DIR=${NRL_VLLM_CACHE_SEED_DIR} \
DG_JIT_CACHE_DIR=${NRL_VLLM_LOCAL_CACHE_DIR}/deep_gemm \
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
NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800 \
NRL_WG_USE_RAY_REF=1 \
HF_HOME=${HF_HOME} \
HF_TOKEN=${HF_TOKEN:-} \
uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
--config examples/configs/grpo_ultra_256n4g_inescapable_sawfly.yaml \
policy.model_name=${NRL_MODEL_PATH} \
cluster.gpus_per_node=4 \
cluster.num_nodes=${NUM_TOTAL_NODES} \
grpo.num_prompts_per_step=${PPS} \
grpo.num_generations_per_prompt=${GPP} \
grpo.val_period=${VAL_PERIOD} \
grpo.advantage_clip_low=${ADVANTAGE_CLIP_LOW} \
grpo.advantage_clip_high=${ADVANTAGE_CLIP_HIGH} \
grpo.penalize_invalid_tool_call=false \
grpo.penalize_malformed_thinking=false \
policy.train_global_batch_size=${GBS} \
policy.megatron_cfg.tensor_model_parallel_size=${TP} \
policy.megatron_cfg.context_parallel_size=${CP} \
policy.megatron_cfg.expert_model_parallel_size=${EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${PP} \
policy.megatron_cfg.empty_unused_memory_level=1 \
policy.generation.vllm_cfg.expert_parallel_size=1 \
policy.generation.colocated.enabled=False \
policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
policy.generation.colocated.resources.gpus_per_node=4 \
checkpointing.save_period=${SAVE_PERIOD} \
env.nemo_gym.num_gpu_nodes=${NUM_GYM_NODES} \
env.nemo_gym.safety_judge_model.responses_api_models.vllm_model.server_args.max_num_seqs=256 \
env.nemo_gym.nl2bash_judge_model.responses_api_models.vllm_model.server_args.max_num_seqs=256 \
env.nemo_gym.genrm_model.responses_api_models.vllm_model.server_args.max_num_seqs=256 \
env.nemo_gym.genrm_model.responses_api_models.vllm_model.model=${NRL_GENRM_MODEL_PATH} \
env.nemo_gym.nl2bash_judge_model.responses_api_models.vllm_model.model=${NRL_NL2BASH_JUDGE_MODEL_PATH} \
env.nemo_gym.safety_judge_model.responses_api_models.vllm_model.model=${NRL_SAFETY_MODEL_PATH} \
data.train.data_path=${NRL_TRAIN_PATH} \
data.validation.data_path=${NRL_VAL_PATH} \
checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
logger.log_dir=${LOG_DIR} \
logger.wandb_enabled=True \
logger.wandb.name=${WANDB_NAME} \
logger.wandb.project=${WANDB_PROJ} \
${NRL_MAX_STEPS:+grpo.max_num_steps=${NRL_MAX_STEPS}} \
${PRECISION_EXTRA_ARGS} \
${*}"


# =============================================================================
# Overlay mounts
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

# =============================================================================
# Per-node cache seeding
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
# Record code provenance
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
  echo "command: ${TRAIN_CMD}"
} > "${RUN_DIR}/provenance.txt"

# =============================================================================
# Dry-run mode
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
# Interactive mode
# =============================================================================
if [[ "${INTERACTIVE}" == "1" ]]; then
  unset COMMAND 2>/dev/null || true

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
    --job-name="interactive-${EXP_SUFFIX}" \
    --partition=batch \
    --time="${WALLTIME}" \
    --gres=gpu:4 \
    --exclusive \
    --mem=0 \
    ${SEGMENT_SIZE:+--segment="${SEGMENT_SIZE}"} \
    ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
    --output="${SLURM_LOG_DIR}/%j.out" \
    --error="${SLURM_LOG_DIR}/%j.err" \
    "${RAY_SUB}")

  echo "${submission_output}"

  if [[ "${submission_output}" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    JOB_ID="${BASH_REMATCH[1]}"
  else
    echo "ERROR: Could not parse job ID from sbatch output."
    exit 1
  fi

  LAUNCH_DIR="$(pwd)"
  ATTACH_SCRIPT="${LAUNCH_DIR}/${JOB_ID}-attach.sh"
  CMD_FILE="${LAUNCH_DIR}/${JOB_ID}-run-cmd.sh"

  cat > "${CMD_FILE}" <<CMDEOF
${TRAIN_CMD}
CMDEOF
  chmod +x "${CMD_FILE}"

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

SBATCH_OUTPUT=$(sbatch \
  --nodes="${NUM_TOTAL_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --job-name="${JOB_NAME}" \
  --partition="${PARTITION}" \
  --time="${WALLTIME}" \
  --gres=gpu:4 \
  --exclusive \
  --mem=0 \
  --dependency="${SBATCH_DEPENDENCY:-singleton}" \
  ${SEGMENT_SIZE:+--segment="${SEGMENT_SIZE}"} \
  ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
  --output="${SLURM_LOG_DIR}/%j.out" \
  --error="${SLURM_LOG_DIR}/%j.err" \
  "${RAY_SUB}")

echo "${SBATCH_OUTPUT}"
JOB_ID=$(echo "${SBATCH_OUTPUT}" | grep -oP '\d+$')

if [[ -n "${JOB_ID}" ]]; then
  echo ""
  echo "  Ray logs:    ${BASE_LOG_DIR}/${JOB_ID}-logs/"
  echo ""
fi
