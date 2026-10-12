#!/bin/bash
set -euo pipefail

# =============================================================================
# launch_ultra_768n.sh
#
# GRPO Ultra V3 — 768-node GB200 NVL72 scale run with NeMo Gym
#
# Batch-only launch script. All logs, checkpoints, and slurm output are written under a shared Lustre directory.
# Uses --dependency=singleton with a fixed job name to prevent concurrent runs.
#
# Usage:
#   ./launch_ultra_768n.sh
#   WALLTIME=8:00:00 ./launch_ultra_768n.sh
#   NRL_MAX_STEPS=10 ./launch_ultra_768n.sh
#   DRY_RUN=1 ./launch_ultra_768n.sh           # print resolved config, don't submit
#
# Adjust node allocation:
#   NUM_TRAIN_NODES=256 NUM_GEN_NODES=480 NUM_GYM_NODES=32 ./launch_ultra_768n.sh
#
# Mount local code into the container (default: container built-in, no overlays):
#   EXTRA_MOUNTS="/path/to/nemo_rl:/opt/nemo-rl/nemo_rl" ./launch_ultra_768n.sh
#
# Extra positional arguments are forwarded as Hydra overrides:
#   ./launch_ultra_768n.sh grpo.max_num_steps=2 policy.precision=float32
#
# Enable MTP (multi-token prediction) speculative decoding for vLLM:
#   ENABLE_MTP_INFERENCE=1 ./launch_ultra_768n.sh
#   ENABLE_MTP_INFERENCE=1 NUM_SPECULATIVE_TOKENS=3 ./launch_ultra_768n.sh
#
# =============================================================================

# =============================================================================
# Required environment — fail fast with clear messages
# =============================================================================
if [[ -z "${HF_HOME:-}" ]]; then
  echo "ERROR: HF_HOME is not set. Export it to a shared HuggingFace cache directory." >&2
  echo "  Example: export HF_HOME=/lustre/.../hf_home" >&2
  exit 1
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not set. W&B logging requires an API key." >&2
  echo "  Get yours at https://wandb.ai/authorize and export WANDB_API_KEY=<key>" >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)
cd "${PROJECT_ROOT}"
PRECISION_RECIPE=${PRECISION_RECIPE:-bf16}

# =============================================================================
# Job identity — fixed name for singleton.
# Must be deterministic so that queued submissions with
# --dependency=singleton correctly serialise instead of running in parallel.
# =============================================================================
JOB_PREFIX="${JOB_PREFIX:-ultra-rl}"
JOB_NAME="${JOB_PREFIX}-768n-${PRECISION_RECIPE}"

# =============================================================================
# Output directories
# =============================================================================
export BASE_LOG_DIR="${BASE_LOG_DIR:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/nemo_rl/logs}"

RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results/${JOB_NAME}}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RESULTS_DIR}/checkpoints}"

RUN_DIR="${RESULTS_DIR}/runs/$(date +%Y%m%d-%H%M)"
LOG_DIR="${RUN_DIR}/logs"
SLURM_LOG_DIR="${RUN_DIR}/slurm"
mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${SLURM_LOG_DIR}"
ln -sfn "${RUN_DIR}" "${RESULTS_DIR}/runs/latest"

# =============================================================================
# SLURM configuration
# =============================================================================
SLURM_ACCOUNT="${SLURM_ACCOUNT:-llmservice_nemotron_ultra}"
PARTITION="${PARTITION:-batch}"
SLURM_QOS="${SLURM_QOS:-normal}"
WALLTIME="${WALLTIME:-4:00:00}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
RESERVATION="${RESERVATION:-sla_res_nemotron}"

# =============================================================================
# Container & mounts
# =============================================================================
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/nemo_rl/images/high_stripe/rl.nightly.sqsh}"
MOUNTS="/lustre:/lustre"

# GB200 NVL72: fixed at 4 GPUs/node.
export GPUS_PER_NODE=4
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-144}"

# =============================================================================
# HuggingFace configuration
# =============================================================================
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

# =============================================================================
# W&B configuration
# =============================================================================
WANDB_PROJ="${WANDB_PROJ:-ultra-v3-pipeclean}"
WANDB_ENTITY="${WANDB_ENTITY:-nvidia}"
WANDB_NAME="${WANDB_NAME:-${JOB_NAME}-$(date +%Y%m%d-%H%M%S)}"
export WANDB_API_KEY
export WANDB_ENTITY

# =============================================================================
# Training
# =============================================================================
NRL_MAX_STEPS="${NRL_MAX_STEPS:-}"

# =============================================================================
# MTP speculative decoding (optional)
# =============================================================================
# Set ENABLE_MTP_INFERENCE=1 to turn on MTP speculative decoding for vLLM inference.
# Tune via NUM_SPECULATIVE_TOKENS / MAX_NUM_BATCHED_TOKENS if needed.
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
# Job shape — specify the 3 intuitive node counts; everything else is derived.
#
#   Training:  256 nodes (1024 GPUs) — Megatron training backend
#   vLLM:      480 nodes (1920 GPUs) — 240 instances at TP=8 EP=8
#   Gym:        32 nodes ( 128 GPUs) — judges (GenRM, NL2Bash, Safety)
#
# =============================================================================
NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-256}"
NUM_GEN_NODES="${NUM_GEN_NODES:-480}"
NUM_GYM_NODES="${NUM_GYM_NODES:-32}"

NUM_ACTOR_NODES=$((NUM_TRAIN_NODES + NUM_GEN_NODES))
NUM_TOTAL_NODES=$((NUM_ACTOR_NODES + NUM_GYM_NODES))

# Sanity checks — catch typos before wasting a Slurm allocation.
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
  echo "  Training=${NUM_TRAIN_NODES} + Generation=${NUM_GEN_NODES} + Gym=${NUM_GYM_NODES} = ${NUM_TOTAL_NODES}" >&2
  echo "  Adjust node counts so the total is a multiple of ${SEGMENT_SIZE}." >&2
  exit 1
fi

# =============================================================================
# Model and data paths
# =============================================================================
NRL_TRAIN_PATH="${NRL_TRAIN_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/data/gym/rl-data-tools/blends/curriculum_v29_warping-muskox.no-swerl.train.jsonl}"
NRL_VAL_PATH="${NRL_VAL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/data/gym/rl-data-tools/blends/curriculum_v29_warping-muskox.no-swerl.val.jsonl}"
NRL_MODEL_PATH="${NRL_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/soumyes/sft-runs/eval_and_sleep/ultra-v3-sft-bf16-hybridep-ep64-cp32-bindpcie-recompute-offload-mar13-blend-512k-filt-1e-5/iter_0001900/hf}"
NRL_GENRM_MODEL_PATH="${NRL_GENRM_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/qwen235b_principle_comparison_genrm_step1230}"
NRL_NL2BASH_JUDGE_MODEL_PATH="${NRL_NL2BASH_JUDGE_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
NRL_SAFETY_MODEL_PATH="${NRL_SAFETY_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/super_v3/model_checkpoints/Nemotron-Content-Safety-Reasoning-4B}"

# =============================================================================
# Lean4 sandbox (for math_formal_lean)
# =============================================================================
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-sandbox-latest.sqsh}"
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
# Triggered when at least one tarball is missing OR stale (cache_write has newer files).
# _stale_tarballs=()
# for _tar_name in inductor_cache triton_cache "vllm_compile_cache_${_vllm_cache_precision}"; do
#   _tar="${CACHE_READ_DIR}/${_tar_name}.tar.zst"
#   _wd="${CACHE_WRITE_DIR}/${_tar_name}"
#   [ -d "$_wd" ] && [ -n "$(ls -A "$_wd" 2>/dev/null)" ] || continue
#   if [ ! -f "$_tar" ]; then
#     _stale_tarballs+=("$_tar_name")
#   elif find "$_wd" -type f -newer "$_tar" -print -quit 2>/dev/null | grep -q .; then
#     _stale_tarballs+=("$_tar_name")
#   fi
# done

# if (( ${#_stale_tarballs[@]} > 0 )); then
#   echo "[CACHE] Missing or stale tarballs: ${_stale_tarballs[*]}"
#   echo "[CACHE] Generating via srun on a compute node..."
#   _promo_script="${CACHE_WRITE_DIR}/.promote_tarballs_$$.sh"
#   cat > "$_promo_script" <<'PROMOSCRIPT'
# #!/bin/bash
# set -euo pipefail
# CACHE_READ_DIR="$1"; CACHE_WRITE_DIR="$2"; shift 2
# for _tar_name in "$@"; do
#   _read_tar="${CACHE_READ_DIR}/${_tar_name}.tar.zst"
#   _write_dir="${CACHE_WRITE_DIR}/${_tar_name}"
#   [ -d "$_write_dir" ] && [ -n "$(ls -A "$_write_dir" 2>/dev/null)" ] || continue
#   _needs=0
#   if [ ! -f "$_read_tar" ]; then
#     _needs=1
#   elif find "$_write_dir" -type f -newer "$_read_tar" -print -quit 2>/dev/null | grep -q .; then
#     _needs=1
#   fi
#   if (( _needs )); then
#     echo "Creating/refreshing ${_tar_name}.tar.zst..."
#     tar --zstd -cf "${_read_tar}.tmp.$$" --blocking-factor=8192 -C "$_write_dir" --exclude='tmp*' --exclude='.tmp_*' --exclude='.*' --exclude='*/.*' . \
#       && mv "${_read_tar}.tmp.$$" "$_read_tar" \
#       && echo "Done: $(du -sh "$_read_tar" | cut -f1)" \
#       || { rm -f "${_read_tar}.tmp.$$"; echo "Failed: ${_tar_name}"; }
#   else
#     echo "${_tar_name}: tarball up to date"
#   fi
# done
# PROMOSCRIPT
#   chmod +x "$_promo_script"
#   srun -N1 -n1 -t 00:30:00 -A "${SLURM_ACCOUNT}" -p cpu \
#     -q cpu-normal \
#     bash "$_promo_script" "${CACHE_READ_DIR}" "${CACHE_WRITE_DIR}" \
#       inductor_cache triton_cache "vllm_compile_cache_${_vllm_cache_precision}" \
#     && echo "[CACHE] srun tarball generation complete" \
#     || echo "[CACHE] srun tarball generation failed (non-fatal, first job will compile from scratch)"
#   rm -f "$_promo_script"
# fi

# =============================================================================
# Code snapshot
# =============================================================================
USE_SNAPSHOT="${USE_SNAPSHOT:-1}"

if [[ "${USE_SNAPSHOT}" == "1" ]]; then
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
if [[ -d "${OVERLAY_SOURCE}/nemo_rl" ]]; then
  MOUNTS="${MOUNTS},${OVERLAY_SOURCE}/nemo_rl:/opt/nemo-rl/nemo_rl"
  echo "  Mount: nemo_rl → /opt/nemo-rl/nemo_rl"
fi
if [[ -d "${OVERLAY_SOURCE}/examples/configs" ]]; then
  MOUNTS="${MOUNTS},${OVERLAY_SOURCE}/examples/configs:/opt/nemo-rl/examples/configs"
  echo "  Mount: configs → /opt/nemo-rl/examples/configs"
fi

if [[ "${USE_SNAPSHOT}" == "1" ]]; then
  MOUNTS="${MOUNTS},${SNAPSHOT_DIR}:${SNAPSHOT_DIR}"
fi

if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
  MOUNTS="${MOUNTS},${EXTRA_MOUNTS}"
  echo "  Extra mounts: ${EXTRA_MOUNTS}"
fi

export MOUNTS

# =============================================================================
# Resolve ray.sub
# =============================================================================
RAY_SUB="${RAY_SUB:-${PROJECT_ROOT}/ray.sub}"
if [[ ! -f "${RAY_SUB}" ]]; then
  echo "ERROR: ray.sub not found at ${RAY_SUB}"
  exit 1
fi

# =================================================================================================================
# Per-node cache seeding
# =================================================================================================================
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
TRAIN_CMD="cd ${CODE_ROOT} && date ; \
${VLLM_ENV_SOURCE}\
OMP_NUM_THREADS=16 \
RAY_DEDUP_LOGS=1 \
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
HF_HOME=${HF_HOME} \
HF_TOKEN=${HF_TOKEN:-} \
NRL_USE_FASTOKENS=${NRL_USE_FASTOKENS:-1} \
uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
--config examples/configs/grpo_ultra_768n4g_${PRECISION_RECIPE}.yaml \
policy.model_name=${NRL_MODEL_PATH} \
cluster.gpus_per_node=4 \
cluster.num_nodes=${NUM_ACTOR_NODES} \
policy.generation.colocated.enabled=False \
policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
policy.generation.colocated.resources.gpus_per_node=4 \
env.nemo_gym.genrm_model.responses_api_models.genrm_model.model=${NRL_GENRM_MODEL_PATH} \
env.nemo_gym.nl2bash_judge_model.responses_api_models.local_vllm_model.model=${NRL_NL2BASH_JUDGE_MODEL_PATH} \
env.nemo_gym.safety_judge_model.responses_api_models.local_vllm_model.model=${NRL_SAFETY_MODEL_PATH} \
env.nemo_gym.nemo_gym_log_dir=\${LOG_DIR}/nemo_gym \
data.train.data_path=${NRL_TRAIN_PATH} \
data.validation.data_path=${NRL_VAL_PATH} \
checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
logger.log_dir=${LOG_DIR} \
logger.wandb_enabled=True \
logger.wandb.name=${WANDB_NAME} \
logger.wandb.project=${WANDB_PROJ} \
${NRL_MAX_STEPS:+grpo.max_num_steps=${NRL_MAX_STEPS}} \
${MTP_EXTRA_ARGS} \
${*}"

export COMMAND="${TRAIN_CMD}"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "================================================================"
echo "  GRPO Ultra V3 — ${NUM_TOTAL_NODES}-node scale run"
echo "================================================================"
echo "  Job name:    ${JOB_NAME}  (singleton — only one runs at a time)"
echo "  Nodes:       ${NUM_TOTAL_NODES} total  (segment=${SEGMENT_SIZE})"
echo "    Training:  ${NUM_TRAIN_NODES}  ($((NUM_TRAIN_NODES * GPUS_PER_NODE)) GPUs)"
echo "    vLLM gen:  ${NUM_GEN_NODES}  ($((NUM_GEN_NODES * GPUS_PER_NODE)) GPUs)"
echo "    Gym:       ${NUM_GYM_NODES}  ($((NUM_GYM_NODES * GPUS_PER_NODE)) GPUs)"
echo "  Walltime:    ${WALLTIME}"
echo ""
echo "  Checkpoints: ${CHECKPOINT_DIR}  (stable — auto-resumes across jobs)"
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
# Submit
# =============================================================================
SBATCH_OUTPUT=$(sbatch \
  --nodes="${NUM_TOTAL_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --job-name="${JOB_NAME}" \
  --partition="${PARTITION}" \
  --time="${WALLTIME}" \
  --gres=gpu:4 \
  --exclusive \
  --mem=0 \
  --dependency=singleton \
  --segment="${SEGMENT_SIZE}" \
  --output="${SLURM_LOG_DIR}/%j.out" \
  --error="${SLURM_LOG_DIR}/%j.err" \
  ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
  ${RESERVATION:+--reservation="${RESERVATION}"} \
  "${RAY_SUB}")

echo "${SBATCH_OUTPUT}"
JOB_ID=$(echo "${SBATCH_OUTPUT}" | grep -oP '\d+$')

if [[ -n "${JOB_ID}" ]]; then
  echo ""
  echo "  Ray logs:    ${BASE_LOG_DIR}/${JOB_ID}-logs/"
  echo ""
fi
