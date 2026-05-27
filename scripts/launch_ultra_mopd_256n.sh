#!/bin/bash
set -euo pipefail

# =============================================================================
# launch_ultra_256n.sh
#
# GRPO Ultra V3 — 256-node GB200 NVL72 scale test with NeMo Gym
#
# Batch-only launch script. All logs, checkpoints, and slurm output are written under a shared Lustre directory.
# Uses --dependency=singleton with a fixed job name to prevent concurrent runs.
#
# Usage:
#   ./launch_ultra_256n.sh
#   WALLTIME=4:00:00 ./launch_ultra_256n.sh
#   NRL_MAX_STEPS=10 ./launch_ultra_256n.sh
#   DRY_RUN=1 ./launch_ultra_256n.sh           # print resolved config, don't submit
#
# Adjust node allocation:
#   NUM_TRAIN_NODES=32 NUM_GEN_NODES=80 NUM_GYM_NODES=8 ./launch_ultra_256n.sh
#
# Mount local code into the container (default: container built-in, no overlays):
#   EXTRA_MOUNTS="/path/to/nemo_rl:/opt/nemo-rl/nemo_rl" ./launch_ultra_256n.sh
#
# Extra positional arguments are forwarded as Hydra overrides:
#   ./launch_ultra_256n.sh grpo.max_num_steps=2 policy.precision=float32
#
# Enable MTP (multi-token prediction) speculative decoding for vLLM:
#   ENABLE_MTP_INFERENCE=1 ./launch_ultra_256n.sh
#   ENABLE_MTP_INFERENCE=1 NUM_SPECULATIVE_TOKENS=3 ./launch_ultra_256n.sh
#
# =============================================================================

# =============================================================================
# Personal settings
# =============================================================================
export HF_HOME=/lustre/fsw/portfolios/llmservice/users/jiaqiz/hf_home
export WANDB_API_KEY=
export GITLAB_PAT=

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
#PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)
PROJECT_ROOT="/lustre/fsw/portfolios/llmservice/users/jiaqiz/launch_scripts_ultra_v3/async-mopd/nemo-rl-internal"
cd "${PROJECT_ROOT}"
PRECISION_RECIPE=${PRECISION_RECIPE:-bf16}

# =============================================================================
# Model and Training configuration 
# =============================================================================
# ---------- Model Configuration ----------
TP="${TP:-8}"
CP="${CP:-8}"
EP="${EP:-64}"
PP="${PP:-1}"
ETP="${ETP:-1}"
VLLM_TP="${VLLM_TP:-8}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
MAX_LENGTH="${MAX_LENGTH:-65536}"

# ---------- Teacher Model Configuration ----------
TEACHER_TP="${TEACHER_TP:-8}"
TEACHER_CP="${TEACHER_CP:-2}"
TEACHER_PP="${TEACHER_PP:-1}"
TEACHER_EP="${TEACHER_EP:-16}"

# ---------- Training ----------
NRL_MAX_STEPS="${NRL_MAX_STEPS:-1000000}"
VAL_PERIOD="${VAL_PERIOD:-10000}"
SAVE_PERIOD="${SAVE_PERIOD:-6}"
LR="${LR:-2e-6}"
MIN_LR="${MIN_LR:-2e-6}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-10}"
KL="${KL:-0}"

# ---------- GRPO ----------
PPS="${PPS:-1024}"
GPP="${GPP:-1}"
GBS="${GBS:-1024}"
TIS_THRESHOLD="${TIS_THRESHOLD:-5}"
SEQ_LOGPROB_ERROR_THRESHOLD="${SEQ_LOGPROB_ERROR_THRESHOLD:-2}"

# ---------- Async GRPO ----------
ASYNC_GRPO=True
MAX_TRAJECTORY_AGE_STEPS=1
IN_FLIGHT_WEIGHT_UPDATES=True
RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES=False
COLOCATED_INFERENCE=False

# =============================================================================
# Job identity — fixed name for singleton.
# Must be deterministic so that queued submissions with
# --dependency=singleton correctly serialise instead of running in parallel.
# =============================================================================
JOB_PREFIX="${JOB_PREFIX:-production-ultra-mopd}"
#JOB_NAME="${JOB_PREFIX}-256n-${PRECISION_RECIPE}"
EXP_SUFFIX="${JOB_PREFIX}-student_general_rl_step170_multienv_v38v39_agentic_v9v12_tp${TP}_cp${CP}_ep${EP}_pp${PP}_gpp${GPP}_pps${PPS}_gbs${GBS}-20260504-tkonuk"
JOB_NAME="${EXP_SUFFIX}"

# =============================================================================
# Output directories
# =============================================================================
# ray.sub reads BASE_LOG_DIR and creates $BASE_LOG_DIR/$SLURM_JOB_ID-logs/ for
# ray infrastructure logs (ray-head.log, ray-driver.log, ray-worker-*.log,
# topology probes, attach scripts, etc.).  Shared project path so all job logs
# are easy to find regardless of who submitted or from where.
export BASE_LOG_DIR="${BASE_LOG_DIR:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/nemo_rl/logs}"
export BASE_LOG_DIR="${BASE_LOG_DIR}/${EXP_SUFFIX}"

# Checkpoints and per-submission run dirs live under the submission directory.
RESULTS_DIR="${RESULTS_DIR:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/results/ultra-v3-posttraining-mopd/${EXP_SUFFIX}}"

# Checkpoint dir is constant across runs so GRPO auto-resumes from the latest
# checkpoint after preemption or resubmission. The CheckpointManager scans
# this directory for the most recent checkpoint on startup.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/results/ultra-v3-posttraining-mopd/${JOB_NAME}}"

# Per-submission dirs for logs and slurm output (timestamped for history).
RUN_DIR="${RESULTS_DIR}/runs/$(date +%Y%m%d-%H%M)"
LOG_DIR="${RUN_DIR}/logs"
SLURM_LOG_DIR="${RUN_DIR}/slurm"
mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${SLURM_LOG_DIR}"
ln -sfn "${RUN_DIR}" "${RESULTS_DIR}/runs/latest"

# =============================================================================
# SLURM configuration
# =============================================================================
SLURM_ACCOUNT="${SLURM_ACCOUNT:-llmservice_nemotron_ultra}"
PARTITION="${PARTITION:-batch_long}"
SLURM_QOS="${SLURM_QOS:-normal}"
WALLTIME="${WALLTIME:-24:00:00}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

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
export HF_HUB_CACHE=/lustre/fsw/portfolios/llmservice/users/jiaqiz/hf_home/hub
export HF_DATASETS_CACHE=/lustre/fsw/portfolios/llmservice/users/jiaqiz/hf_home/hub

# =============================================================================
# W&B configuration
# =============================================================================
WANDB_PROJ="${WANDB_PROJ:-ultra-v3-posttraining}"
WANDB_ENTITY="${WANDB_ENTITY:-nvidia}"
WANDB_NAME="${EXP_SUFFIX}"
export WANDB_API_KEY
export WANDB_ENTITY

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
#   Training: 128 nodes (512 GPUs)  — Megatron training backend
#   vLLM:     118 nodes (472 GPUs)  — 59 instances at TP=8 EP=8
#   Gym:       10 nodes ( 40 GPUs)  — judges (GenRM, NL2Bash, Safety)
#
# =============================================================================
NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-128}"
NUM_GEN_NODES="${NUM_GEN_NODES:-82}"
NUM_GYM_NODES="${NUM_GYM_NODES:-10}"

NUM_UNIQUE_TEACHERS="${NUM_UNIQUE_TEACHERS:-9}"
NUM_NODES_PER_TEACHER="${NUM_NODES_PER_TEACHER:-4}"
NUM_TEACHER_NODES=$((NUM_UNIQUE_TEACHERS * NUM_NODES_PER_TEACHER))

NUM_ACTOR_NODES=$((NUM_TRAIN_NODES + NUM_GEN_NODES + NUM_TEACHER_NODES))
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
NRL_TRAIN_PATH="${NRL_TRAIN_PATH:-/lustre/fsw/portfolios/llmservice/users/tkonuk/data/gym/datasets/mopd/production-20260501/curriculum_v38v39_agentic_v9v12_merged.train.v2.shuffled.jsonl}"
NRL_SWE_TRAIN_PATH="${NRL_SWE_TRAIN_PATH:-/lustre/fsw/portfolios/llmservice/users/tkonuk/data/gym/datasets/mopd/production-20260501/large_root_cause_curriculum_with_mercor_ots_plus_singlefile_swerebench_overlap_fix.jsonl}"

NRL_VAL_PATH="${NRL_VAL_PATH:-/lustre/fsw/portfolios/llmservice/users/tkonuk/data/gym/datasets/mopd/production-20260501/curriculum_v38v39_agentic_v9v12_merged.train.v2.shuffled.jsonl}"
NRL_MODEL_PATH="${NRL_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/models/ultra_stage2sft_step300}"
NRL_GENRM_MODEL_PATH="${NRL_GENRM_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/qwen235b_principle_comparison_genrm_step1230}"
NRL_NL2BASH_JUDGE_MODEL_PATH="${NRL_NL2BASH_JUDGE_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
NRL_SAFETY_MODEL_PATH="${NRL_SAFETY_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/super_v3/model_checkpoints/Nemotron-Content-Safety-Reasoning-4B}"

# ---------- Teachers ----------
NRL_GENERAL_TEACHER_PATH="${NRL_GENERAL_TEACHER_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/results/ultra-v3-pipeclean/pipeclean-ultra-rl-prod_ultra_stage2sft300_fixlc_tp8_cp8_ep64_pp1_gpp16_pps512_gbs8192-20260417-jiaqi-resumestep128-65k/eval/step_152/hf}"
NRL_AGENTIC_TEACHER_PATH="${NRL_AGENTIC_TEACHER_PATH:-/lustre/fsw/portfolios/llmservice/users/tkonuk/models/ultra-v3/mopd/ultra-v3-grpo_agentic-sft-step1175_wondrous-robin_tp8_cp16_ep64_pp1_gpp16_pps128_gbs2048_maxlen69632-20260416-venkats_step_77}"
NRL_SWE_TEACHER_PATH="${NRL_SWE_TEACHER_PATH:-/lustre/fsw/portfolios/llmservice/users/tkonuk/models/ultra-v3/mopd/e2e_swe_0427_step_83}"
NRL_SEARCH_TEACHER_PATH="${NRL_SEARCH_TEACHER_PATH:-/lustre/fsw/portfolios/llmservice/users/rgala/repos/0427-create-search-sft-dataset/sft_chkpts/sft_blend_v2/checkpoints/search-sft-v1/78_hf/hf}"
NRL_TERMINAL_STIRRUP_TEACHER_PATH="${NRL_TERMINAL_STIRRUP_TEACHER_PATH:-/lustre/fsw/portfolios/llmservice/users/tkonuk/models/ultra-v3/mopd/ultra-v3-grpo_mongooseV14_sft1175_chattmpl_tkonukEP64_allAgents_tp8_cp8_ep64_pp1_gpp16_pps128_gbs2048_maxlen49152-20260425c-venkats_step_98}"

NRL_IFBENCH_TEACHER_PATH="${NRL_IFBENCH_TEACHER_PATH:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/adithyare/code/prod_run/checkpoints/pipeclean-ultra-rl-prod_ultra_stage2sft300_v38_tp8_cp8_ep64_pp1_gpp16_pps512_gbs8192-20260417-adithyare-lenprofiledpenalties-from-step80-try2/step_125/hf/hf}"
NRL_OMNISCIENCE_TEACHER_PATH="${NRL_OMNISCIENCE_TEACHER_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/results/ultra-v3-pipeclean/pipeclean-ultra-rl-prod_ultra_stage2sft300_fixlc_tp8_cp8_ep64_pp1_gpp16_pps512_gbs8192-20260417-jiaqi-resumestep136/eval/step_152/hf}"
NRL_CODE_TEACHER_PATH="${NRL_CODE_TEACHER_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/results/ultra-v3-pipeclean/pipeclean-ultra-rl-prod_ultra_stage2sft300_fixlc_tp8_cp8_ep64_pp1_gpp16_pps512_gbs8192-20260417-jiaqi-resumestep128-65k/eval/step_144/hf}"
NRL_NS_TOOLS_TEACHER_PATH="${NRL_NS_TOOLS_TEACHER_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/results/ultra-v3-pipeclean/pipeclean-ultra-rl-prod_ultra_stage2sft300_fixlc_tp8_cp8_ep64_pp1_gpp16_pps512_gbs8192-20260417-jiaqi-resumestep128-65k/eval/step_170/hf}"

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
# Lustre holds the warm persistent cache. At job start, SETUP_COMMAND clears
# stale /tmp caches then seeds node-local /tmp from Lustre. JIT writes go to
# /tmp to avoid Lustre metadata contention from parallel compilation.
if [[ -z "${PERSISTENT_CACHE:-}" ]]; then
  _access_group="${SLURM_ACCOUNT%%_*}"
  PERSISTENT_CACHE="/lustre/fsw/portfolios/${_access_group}/users/${USER}/.cache/nemotron_ultra_new"
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
# Snapshot the git-tracked source tree so the code is frozen at submission time.
# This guarantees we know exactly which code was used for a given experiment.
# The snapshot directory path is recorded in the summary output and logs.
#
# Set USE_SNAPSHOT=0 to skip (runs from container built-in or live checkout).
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
# By default, nemo_rl (Python package) and examples/configs (YAML configs) from
# the code snapshot are overlaid into the container. Everything else uses the
# container's built-in code at /opt/nemo-rl.
#
# To overlay additional components, use EXTRA_MOUNTS with explicit host:container
# pairs. Examples:
#
#   # Mount Megatron-LM (will shadow prebuilt venvs — expect slow startup)
#   EXTRA_MOUNTS="/path/to/Megatron-LM:/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM" ./scripts/launch_ultra_256n.sh
#
# Container paths for reference:
#   /opt/nemo-rl/nemo_rl                                              — Python package
#   /opt/nemo-rl/examples/configs                                     — YAML configs
#   /opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM          — Megatron-LM
#   /opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge   — Megatron-Bridge
#   /opt/nemo-rl/3rdparty/Gym-workspace/Gym                           — NeMo-Gym
#   /opt/nemo-rl/3rdparty/vllm                                        — vLLM
# =============================================================================
if [[ -d "${OVERLAY_SOURCE}/nemo_rl" ]]; then
  MOUNTS="${MOUNTS},${OVERLAY_SOURCE}/nemo_rl:/opt/nemo-rl/nemo_rl"
  echo "  Mount: nemo_rl → /opt/nemo-rl/nemo_rl"
fi
if [[ -d "${OVERLAY_SOURCE}/examples" ]]; then
  MOUNTS="${MOUNTS},${OVERLAY_SOURCE}/examples:/opt/nemo-rl/examples"
  echo "  Mount: examples → /opt/nemo-rl/examples"
fi
if [[ -d "${OVERLAY_SOURCE}/examples/configs" ]]; then
  MOUNTS="${MOUNTS},${OVERLAY_SOURCE}/examples/configs:/opt/nemo-rl/examples/configs"
  echo "  Mount: configs → /opt/nemo-rl/examples/configs"
fi
if [[ -d "${OVERLAY_SOURCE}/3rdparty/Gym-workspace/Gym" ]]; then
  MOUNTS="${MOUNTS},${OVERLAY_SOURCE}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
  echo "  Mount: Gym → /opt/nemo-rl/3rdparty/Gym-workspace/Gym"
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
# Triton, Inductor, and FlashInfer cubins compile/download to node-local /tmp to avoid Lustre race conditions
# and file lock contention during concurrent JIT compilation and cubin fetching.
# To avoid cold-start penalties, we seed /tmp from a warm Lustre cache before Ray starts (SETUP_COMMAND).
#
# IMPORTANT: Stale /tmp caches from previous jobs can cause hangs (e.g. triton_bundler
# skipping non-empty temp dirs). We rm -rf /tmp caches first, then seed fresh from Lustre.
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

# ++env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.dataset_path=$NRL_SWE_TRAIN_PATH \
# ++env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.dataset_path=$NRL_SWE_TRAIN_PATH \
# ++env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.openhands_should_log=True \
# ++env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.openhands_should_log=True \
# on_policy_distillation.teacher_model_by_agent_name.swe_agents_train=${NRL_SWE_TEACHER_PATH} \
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
NEMO_GYM_VENV_DIR=${NEMO_GYM_VENV_DIR} \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
VLLM_FLASHINFER_MOE_BACKEND=latency \
NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800 \
NRL_WG_USE_RAY_REF=1 \
HF_HOME=${HF_HOME} \
HF_TOKEN=${HF_TOKEN:-} \
NRL_USE_FASTOKENS=${NRL_USE_FASTOKENS:-1} \
UV_INDEX_FLASHINFER_INTERNAL_PYPI_USERNAME=__token__ \
UV_INDEX_FLASHINFER_INTERNAL_PYPI_PASSWORD=${GITLAB_PAT} \
uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
--config examples/configs/mopd_ultra_256n4g_${PRECISION_RECIPE}.yaml \
policy.model_name=${NRL_MODEL_PATH} \
cluster.gpus_per_node=4 \
cluster.num_nodes=${NUM_ACTOR_NODES} \
grpo.val_period=${VAL_PERIOD} \
grpo.num_prompts_per_step=${PPS} \
grpo.num_generations_per_prompt=${GPP} \
grpo.seq_logprob_error_threshold=${SEQ_LOGPROB_ERROR_THRESHOLD} \
grpo.async_grpo.enabled=${ASYNC_GRPO} \
grpo.async_grpo.max_trajectory_age_steps=${MAX_TRAJECTORY_AGE_STEPS} \
grpo.async_grpo.in_flight_weight_updates=${IN_FLIGHT_WEIGHT_UPDATES} \
grpo.async_grpo.recompute_kv_cache_after_weight_updates=${RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES} \
policy.generation.colocated.enabled=False \
policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
policy.generation.colocated.resources.gpus_per_node=4 \
policy.train_global_batch_size=${GBS} \
policy.max_total_sequence_length=${MAX_LENGTH} \
policy.megatron_cfg.tensor_model_parallel_size=${TP} \
policy.megatron_cfg.context_parallel_size=${CP} \
policy.megatron_cfg.expert_model_parallel_size=${EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${PP} \
policy.megatron_cfg.expert_tensor_parallel_size=${ETP} \
policy.megatron_cfg.optimizer.lr=${LR} \
policy.megatron_cfg.optimizer.min_lr=${MIN_LR} \
policy.megatron_cfg.scheduler.lr_warmup_iters=${LR_WARMUP_ITERS} \
policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP} \
policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_GPU_UTIL} \
policy.generation.vllm_cfg.max_model_len=${MAX_LENGTH} \
policy.generation.colocated.enabled=${COLOCATED_INFERENCE} \
policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
env.nemo_gym.genrm_model.responses_api_models.genrm_model.model=${NRL_GENRM_MODEL_PATH} \
env.nemo_gym.nl2bash_judge_model.responses_api_models.local_vllm_model.model=${NRL_NL2BASH_JUDGE_MODEL_PATH} \
env.nemo_gym.safety_judge_model.responses_api_models.local_vllm_model.model=${NRL_SAFETY_MODEL_PATH} \
env.nemo_gym.nemo_gym_log_dir=${LOG_DIR}/nemo_gym \
on_policy_distillation.teacher_model_by_agent_name.math_with_judge_simple_agent=${NRL_NS_TOOLS_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.lc_judge_simple_agent=${NRL_GENERAL_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.terminal_multi_harness_stirrup_agent=${NRL_TERMINAL_STIRRUP_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.terminus_judge_string_only_simple_agent=${NRL_NS_TOOLS_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.search_pivot_single_step_tool_use_with_argument_comparison_agent=${NRL_SEARCH_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.swe_pivot_single_step_tool_use_with_argument_comparison_agent=${NRL_SWE_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.swe_pivot_tool_simulation_agent=${NRL_SWE_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.structured_outputs_v3_simple_agent=${NRL_AGENTIC_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.terminal_multi_harness_opencode_agent=${NRL_AGENTIC_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.terminal_multi_harness_agent006_agent=${NRL_AGENTIC_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.terminal_multi_harness_codex_agent=${NRL_AGENTIC_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.single_step_tool_use_with_argument_comparison_agent=${NRL_AGENTIC_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.droid_pivot_single_step_tool_use_with_argument_comparison_agent=${NRL_AGENTIC_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.freeform_formatting_simple_agent=${NRL_AGENTIC_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.citation_format_simple_agent=${NRL_AGENTIC_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.instruction_following_simple_agent=${NRL_IFBENCH_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.abstention_simple_agent=${NRL_OMNISCIENCE_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.code_gen_simple_agent=${NRL_CODE_TEACHER_PATH} \
on_policy_distillation.teacher_model_by_agent_name.ns_tools_simple_agent=${NRL_NS_TOOLS_TEACHER_PATH} \
on_policy_distillation.default_teacher_alias=lc_judge_simple_agent \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.tensor_model_parallel_size=${TEACHER_TP} \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.context_parallel_size=${TEACHER_CP} \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.pipeline_model_parallel_size=${TEACHER_PP} \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.expert_model_parallel_size=${TEACHER_EP} \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.num_nodes=${NUM_NODES_PER_TEACHER} \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.gpus_per_node=4 \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.precision=bf16 \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.micro_batch_size=1 \
++on_policy_distillation.non_colocated_teachers.default_teacher_cfg.megatron_cfg_overrides.moe_token_dispatcher_type=alltoall \
++on_policy_distillation.non_colocated_teachers.default_teacher_cfg.megatron_cfg_overrides.moe_flex_dispatcher_backend=deepep \
data.train.data_path=${NRL_TRAIN_PATH} \
data.validation.data_path=${NRL_VAL_PATH} \
checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
checkpointing.save_period=${SAVE_PERIOD} \
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
echo "  GRPO Ultra V3 — ${NUM_TOTAL_NODES}-node scale test"
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
  --reservation=sla_res_nemotron \
  ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
  "${RAY_SUB}")

echo "${SBATCH_OUTPUT}"
JOB_ID=$(echo "${SBATCH_OUTPUT}" | grep -oP '\d+$')

if [[ -n "${JOB_ID}" ]]; then
  echo ""
  echo "  Ray logs:    ${BASE_LOG_DIR}/${JOB_ID}-logs/"
  echo ""
fi
