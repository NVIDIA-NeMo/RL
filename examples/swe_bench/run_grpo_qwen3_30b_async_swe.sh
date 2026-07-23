#!/bin/bash
# =============================================================================
# Async GRPO on SWE-bench (NeMo-Gym / OpenHands) — Qwen3-30B-A3B, SLURM launcher.
#
# Submits a 16-node (8 train + 8 generation, non-colocated) async GRPO run via
# ray.sub. Supports both entrypoints:
#   SC_MODE=1 (default)  examples/run_grpo_single_controller.py
#                        (single-controller + TransferQueue data plane)
#   SC_MODE=0            examples/nemo_gym/run_grpo_nemo_gym.py
#                        (classic async GRPO; async comes from the yaml's
#                        grpo.async_grpo block)
#
# This file is environment-agnostic: no secrets and no user-specific paths.
# Site-specific values come from the environment (see REQUIRED below) — wrap
# this script with your own launcher that exports them (see
# test_assets/SWE/grpo_swe_tests.sh for an example wrapper).
#
# ---------------------------------------------------------------------------
# REQUIRED environment:
#   ACCOUNT           SLURM account
#   CONTAINER         enroot .sqsh/.squashfs image (a recent NeMo-RL nightly;
#                     must postdate the TransferQueue pyproject dependency)
#   MODEL_PATH        HF checkpoint dir to train from
#   TRAIN_DATA_PATH   SWE train jsonl
#
# Common optional environment (see defaults inline for the full list):
#   PARTITION (batch), NUM_NODES (16), NUM_GEN_NODES (8), TIME (4:0:0)
#   VAL_DATA_PATH (=TRAIN_DATA_PATH), CONFIG_FILE (yaml next to this script)
#   TP/EP/CP/PP/VLLM_TP, SEQLEN, PPS/GPP/GBS, LR, MAX_NUM_STEPS
#   SC_MODE (1), MIN_PROMPT_GROUPS (=PPS), MIP (=PPS; async_rl.max_inflight_prompts)
#   OVER_SAMPLING (false), FORCE_IN_ORDER (true)
#       streaming-1 (default): OVER_SAMPLING=false FORCE_IN_ORDER=true
#           — no over-generation, each step consumes the groups dispatched
#           for it; 1:1 repro of the classic async_grpo (age=1) dispatch.
#       streaming-2:           OVER_SAMPLING=true  FORCE_IN_ORDER=false
#           — generation keeps producing, steps consume any groups within
#           the staleness window; stale groups get evicted (wasted).
#   PERSISTENT_CACHE  compile/uv cache root on shared fs (~/.cache/... default)
#   GYM_VENV_DIR      /tmp/nemo_gym_venvs default; /opt/gym_venvs if your image
#                     has the gym server venvs baked in
#   MOUNTS / EXTRA_MOUNTS   container mounts (make model/data paths visible!)
#   WANDB_API_KEY + WANDB_PROJ + EXP_SUFFIX   logging & naming
#   HUGGINGFACE_TOKEN / GITHUB_TOKEN / GITLAB_TOKEN   passed through if set
#   REAPER_COMMENT    SLURM --comment payload (cluster-specific; empty default)
#   DRY_RUN=1         print everything, submit nothing
#
# Usage (from a repo/snapshot root that contains ray.sub):
#   ACCOUNT=... CONTAINER=... MODEL_PATH=... TRAIN_DATA_PATH=... \
#     bash examples/swe_bench/run_grpo_qwen3_30b_async_swe.sh
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# Locate the tree to submit from: this script lives at examples/swe_bench/
# inside the repo (or a code snapshot); ray.sub must exist at its root.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${RUN_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
if [ ! -f "${RUN_DIR}/ray.sub" ]; then
    echo "Error: ${RUN_DIR}/ray.sub not found — RUN_DIR must be a NeMo-RL checkout/snapshot root." >&2
    exit 1
fi
RUN_COMMIT="unknown"
if git -C "${RUN_DIR}" rev-parse --short HEAD >/dev/null 2>&1; then
    RUN_COMMIT="$(git -C "${RUN_DIR}" rev-parse --short HEAD)"
elif [ -f "${RUN_DIR}/commit.txt" ]; then
    RUN_COMMIT="$(cut -c1-7 "${RUN_DIR}/commit.txt")"
fi

# ---------------------------------------------------------------------------
# Required site-specific inputs — fail fast with a clear message.
# ---------------------------------------------------------------------------
missing=""
[ -n "${ACCOUNT:-}" ]         || missing+=" ACCOUNT"
[ -n "${CONTAINER:-}" ]       || missing+=" CONTAINER"
[ -n "${MODEL_PATH:-}" ]      || missing+=" MODEL_PATH"
[ -n "${TRAIN_DATA_PATH:-}" ] || missing+=" TRAIN_DATA_PATH"
if [ -n "${missing}" ]; then
    echo "Error: missing required environment variables:${missing}" >&2
    echo "See the header of this script for the full contract." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Cluster / submission (site-specific, all overridable)
# ---------------------------------------------------------------------------
PARTITION="${PARTITION:-batch}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CPUS_PER_WORKER="${CPUS_PER_WORKER:-114}"
# SLURM --comment payload (e.g. idle-GPU-reaper exemptions on some clusters).
REAPER_COMMENT="${REAPER_COMMENT:-}"

# ---------------------------------------------------------------------------
# Scale / walltime
# ---------------------------------------------------------------------------
NUM_NODES="${NUM_NODES:-16}"          # total allocation
NUM_GEN_NODES="${NUM_GEN_NODES:-8}"   # carved out of NUM_NODES for generation
TIME="${TIME:-4:0:0}"
MAX_NUM_STEPS="${MAX_NUM_STEPS:-}"    # empty => use the yaml's value

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/grpo_qwen3_30b_async_swe.yaml}"
VAL_DATA_PATH="${VAL_DATA_PATH:-${TRAIN_DATA_PATH}}"

# ============================ Parallelism ============================
TP="${TP:-4}"; EP="${EP:-8}"; CP="${CP:-4}"; PP="${PP:-2}"; VLLM_TP="${VLLM_TP:-2}"
MIN_PAD=1
[ "${CP}" -gt 1 ] && MIN_PAD=$((MIN_PAD * CP * 2))
[ "${TP}" -gt 1 ] && MIN_PAD=$((MIN_PAD * TP))
MAKE_SEQ_DIVISIBLE_BY="${MIN_PAD}"
SEQUENCE_PACKING=True

# ===================== Sequence length =====================
SEQLEN="${SEQLEN:-131072}"

# ===== Single-controller async-RL knobs (SC_MODE=1: data_plane + async_rl) =====
# Maps the classic async_grpo (age=1) 1:1 to async_rl.
MAX_TRAJECTORY_AGE_STEPS="${MAX_TRAJECTORY_AGE_STEPS:-1}"  # -> async_rl.max_weight_staleness_versions
BATCH_SELECTION_STRATEGY=staleness_window
# Dispatch semantics ("streaming" modes). Defaults = strict 1:1 repro of the
# classic async_grpo age-1 behavior (streaming-1): no over-generation, rollouts
# consumed strictly by their dispatch-target step. streaming-2 = the code
# defaults (over_sampling=true force_in_order=false): generation keeps
# producing (stale groups get evicted/wasted) and steps consume any groups
# inside the staleness window, out of order.
OVER_SAMPLING="${OVER_SAMPLING:-false}"
FORCE_IN_ORDER="${FORCE_IN_ORDER:-true}"
FORCE_ON_POLICY_RATIO=True
SEQ_LOGPROB_ERROR_THRESHOLD=null
COLOCATED_ENABLED=False
VLLM_GPU_UTIL=0.8
OVERLAP_GRAD_REDUCE=False
ADVANTAGE_CLIP_LOW=-100
ADVANTAGE_CLIP_HIGH=100
TIS_THRESHOLD=5

# ========================= GRPO / sampling =========================
PPS="${PPS:-8}"; GPP="${GPP:-8}"; GBS="${GBS:-64}"
# Intra-step dispatch granularity (SC only): the sampler releases work to the
# trainer once this many complete prompt groups are ready (gradient
# accumulation; the optimizer still steps only after PPS groups). Not passed in
# => same as PPS = fully synchronous within the step; smaller values overlap
# trainer compute with the generation tail.
MIN_PROMPT_GROUPS="${MIN_PROMPT_GROUPS:-${PPS}}"
# Cap on concurrent in-flight rollout dispatches (SC only): async_rl.max_inflight_prompts,
# the semaphore bounding how many prompt rollouts the rollout pump runs at once.
# Default = PPS (one step's worth in flight). Larger values overlap more generation.
MIP="${MIP:-${PPS}}"
NORMALIZE_REWARDS=True
OVERLONG_FILTERING=True
VAL_PERIOD="${VAL_PERIOD:-1000}"

# ========================== Loss function ==========================
KL=0
CLIP_MIN=0.2
CLIP_MAX=0.28
USE_ON_POLICY_KL_APPROXIMATION=True
IMPORTANCE_SAMPLING_CORRECTION=True
SEQ_LEVEL_IS=False
TOKEN_LEVEL_LOSS=True

# ============================ Optimizer ============================
LR="${LR:-1e-06}"

# =============================== MoE ===============================
MOE_FREEZE_ROUTER=True
MOE_PERMUTE_FUSION=True
MOE_ENABLE_DEEPEP=False
MOE_TOKEN_DISPATCHER_TYPE="alltoall"
MOE_AUX_LOSS_COEFF=0
MOE_ROUTER_LOAD_BALANCING_TYPE="none"
MOE_ROUTER_BIAS_UPDATE_RATE="1e-3"

# ======================= Generation / vLLM =======================
TEMPERATURE=1.0

# =================== Checkpointing & validation ===================
SAVE_PERIOD="${SAVE_PERIOD:-5}"
KEEP_TOP_K="${KEEP_TOP_K:-2}"
MUST_SAVE_BY="${MUST_SAVE_BY:-00:03:35:00}"   # graceful save+exit before TIME

# ============================ SWE agent ============================
AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-200}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-1800}"

# ============================== Logging ==============================
WANDB_PROJ="${WANDB_PROJ:-nemo-rl-swe-bench}"
LOG_GYM_RESPONSES=true

# ========================= Experiment naming =========================
SYNC_MODE="async-age${MAX_TRAJECTORY_AGE_STEPS}"
EXP_SUFFIX="${EXP_SUFFIX:-swe-sc@${RUN_COMMIT}-${SYNC_MODE}-pps${PPS}-mip${MIP}-gpp${GPP}-gbs${GBS}-lr${LR}-tp${TP}}"
WANDB_NAME="${EXP_SUFFIX}"
EXP_NAME="${EXP_SUFFIX}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RUN_DIR}/results}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CHECKPOINT_ROOT}/${EXP_SUFFIX}}"
BASE_LOG_DIR="${BASE_LOG_DIR:-${RUN_DIR}/logs/slurm}"

# ========================= Runtime env =========================
# Secrets are passed through from the caller's environment; never set here.
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
export GITHUB_TOKEN="${GITHUB_TOKEN:-}"
export GITLAB_TOKEN="${GITLAB_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
# Node-local uv cache (ephemeral). Each node builds its own venv independently —
# a shared-filesystem uv cache corrupts under concurrent multi-node builds.
# Pre-warm LUSTRE_UV_CACHE once with a single process to skip long compiles.
export UV_CACHE_DIR=/tmp/uv_cache
export UV_LOCK_TIMEOUT=3600
export RAY_DEDUP_LOGS=1
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"
export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

# ===================== Shared-fs compile caches =====================
# Runtime caches stay node-local (/tmp); the shared-fs copies are only seeded
# from (read) at startup and written back by a periodic sidecar, so repeat runs
# skip the ~20min triton MoE JIT.
PERSISTENT_CACHE="${PERSISTENT_CACHE:-${HOME}/.cache/nemo_rl_swe_bench}"
export LUSTRE_VLLM_CACHE="${PERSISTENT_CACHE}/vllm_compile_cache"
export LUSTRE_INDUCTOR_CACHE="${PERSISTENT_CACHE}/inductor_cache"
export LUSTRE_TRITON_CACHE="${PERSISTENT_CACHE}/triton_cache"
export LUSTRE_UV_CACHE="${PERSISTENT_CACHE}/uv_cache"
export NRL_VLLM_LOCAL_CACHE_DIR="/tmp/nemo_rl_vllm_cache"
export NRL_VLLM_CACHE_SEED_DIR="/tmp/nemo_rl_vllm_cache_warm"
export INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
export TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
export CACHE_SYNC_FREQUENCY="${CACHE_SYNC_FREQUENCY:-120}"
mkdir -p "${LUSTRE_VLLM_CACHE}" "${LUSTRE_INDUCTOR_CACHE}" "${LUSTRE_TRITON_CACHE}" "${LUSTRE_UV_CACHE}"

# ===================== NeMo-Gym server venvs =====================
# Default: node-local build (safe everywhere). If your container has the gym
# server venvs baked in (vllm_model + swe_agents), export GYM_VENV_DIR=/opt/gym_venvs.
# Do NOT point this at a shared filesystem: the editable nemo-gym build hangs on
# uv's flock there, and interrupted builds leave empty venv shells that
# skip-if-present then reuses -> the gym policy_model server crashes on import.
GYM_VENV_DIR="${GYM_VENV_DIR:-/tmp/nemo_gym_venvs}"
case "${GYM_VENV_DIR}" in /opt/*|/tmp/*) ;; *) mkdir -p "${GYM_VENV_DIR}" ;; esac

# ===== SETUP_COMMAND: install apptainer + seed caches + uv sync =====
# Runs on all nodes before Ray starts (consumed by ray.sub).
read -r -d '' SETUP_COMMAND <<SETUPEOF || true
echo "[SETUP] Installing apptainer for SWE sandbox..."
apt-get update && apt-get install -y git build-essential gcc wget 2>/dev/null || true
RET=1
RETRIES=3
for attempt in \$(seq 1 \$RETRIES); do
  if command -v apptainer >/dev/null 2>&1 || command -v singularity >/dev/null 2>&1; then
    echo "[SETUP] singularity/apptainer already available"
    RET=0
    break
  fi
  cd /tmp && \
  wget --no-check-certificate -q https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_amd64.deb && \
  apt install -y ./apptainer_1.3.1_amd64.deb && \
  ln -sf /usr/bin/apptainer /usr/bin/singularity
  if command -v apptainer >/dev/null 2>&1; then
    echo "[SETUP] apptainer installed successfully"
    RET=0
    break
  fi
  echo "[SETUP] apptainer install attempt \$attempt failed, retrying..."
  sleep 10
done
if [ \$RET -ne 0 ]; then
  echo "[SETUP] WARNING: apptainer installation failed after \$RETRIES attempts"
fi

echo "[CACHE SEED] Clearing stale /tmp caches and seeding from shared fs..."
rm -rf /tmp/nemo_rl_vllm_cache /tmp/nemo_rl_vllm_cache_*
rm -rf "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
mkdir -p "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

find "${LUSTRE_INDUCTOR_CACHE}" -maxdepth 1 -name '.tmp_*' -mmin +30 -exec rm -rf {} + 2>/dev/null || true
find "${LUSTRE_TRITON_CACHE}" -maxdepth 1 -name '.tmp_*' -mmin +30 -exec rm -rf {} + 2>/dev/null || true

_seed_cache() {
  local lustre="\$1" local_dir="\$2" name="\$3"
  if [ -d "\$lustre" ] && [ "\$(ls -A "\$lustre" 2>/dev/null)" ]; then
    rsync -a --exclude '.tmp_*' "\$lustre/" "\$local_dir/" 2>/dev/null \
      && echo "[CACHE SEED] \$name: seeded from shared fs" \
      || echo "[CACHE SEED] \$name: seed failed (non-fatal)"
  else
    echo "[CACHE SEED] \$name: no warm cache on shared fs yet"
  fi
}

_seed_cache "${LUSTRE_INDUCTOR_CACHE}" "${INDUCTOR_CACHE_DIR}" "Inductor"
_seed_cache "${LUSTRE_TRITON_CACHE}" "${TRITON_CACHE_DIR}" "Triton"
# uv cache: read-only rsync of single-process-prebuilt wheels into node-local
# /tmp so the uv sync below is a cache hit, not a ~28min build.
# (NOTE: no backticks/'\$(...)' in this heredoc body -- they would be command-
# substituted on the LOGIN node when SETUP_COMMAND is read.)
mkdir -p "${UV_CACHE_DIR}"
_seed_cache "${LUSTRE_UV_CACHE}" "${UV_CACHE_DIR}" "uv (prebuilt wheels)"
echo "[CACHE SEED] Done."

# ===== Compile-cache WRITE-BACK sidecar =====
# The seed above only READS shared fs -> /tmp. This sidecar periodically rsyncs
# /tmp -> shared fs (and on TERM/INT) so compiled kernels persist across runs;
# --ignore-existing makes concurrent per-node writes first-writer-wins.
_sync_cache_one() {
  local src="\$1" dst="\$2" name="\$3"
  mkdir -p "\$dst"
  if [ -d "\$src" ] && [ "\$(ls -A "\$src" 2>/dev/null)" ]; then
    rsync -a --ignore-existing --exclude '.tmp_*' --exclude 'tmp*' "\$src/" "\$dst/" 2>/dev/null \
      && echo "[CACHE SYNC] \$name: /tmp -> shared fs" \
      || echo "[CACHE SYNC] \$name: sync failed (non-fatal)"
  fi
}
_sync_compile_caches_to_lustre() {
  _sync_cache_one "${INDUCTOR_CACHE_DIR}" "${LUSTRE_INDUCTOR_CACHE}" "Inductor"
  _sync_cache_one "${TRITON_CACHE_DIR}" "${LUSTRE_TRITON_CACHE}" "Triton"
}
_start_cache_sync_sidecar() {
  local pidfile="/tmp/nemo_rl_compile_cache_sync.pid"
  if [ -f "\$pidfile" ] && kill -0 "\$(cat "\$pidfile" 2>/dev/null)" 2>/dev/null; then
    echo "[CACHE SYNC] sidecar already running (pid=\$(cat "\$pidfile" 2>/dev/null))"
    return
  fi
  (
    set +e
    trap '_sync_compile_caches_to_lustre; exit 0' TERM INT
    echo "[CACHE SYNC] sidecar started, frequency=${CACHE_SYNC_FREQUENCY}s"
    while true; do
      sleep "${CACHE_SYNC_FREQUENCY}"
      _sync_compile_caches_to_lustre
    done
  ) > /tmp/nemo_rl_compile_cache_sync.log 2>&1 &
  echo "\$!" > "\$pidfile"
  echo "[CACHE SYNC] sidecar pid=\$!"
}
if [ "${CACHE_SYNC_FREQUENCY}" -gt 0 ] 2>/dev/null; then
  _start_cache_sync_sidecar
else
  echo "[CACHE SYNC] disabled (CACHE_SYNC_FREQUENCY=${CACHE_SYNC_FREQUENCY})"
fi

UV_HTTP_TIMEOUT=3600 \
  uv sync --frozen --extra mcore
SETUPEOF
export SETUP_COMMAND

# Optional extra grpo overrides (only emitted when set, so empty == use yaml).
EXTRA_GRPO=""
[ -n "${MAX_NUM_STEPS}" ] && EXTRA_GRPO="grpo.max_num_steps=${MAX_NUM_STEPS}"

# ===== Entrypoint switch: SC (single-controller) vs classic async GRPO =====
# SC_MODE=1 (default): examples/run_grpo_single_controller.py + data_plane/async_rl overrides.
# SC_MODE=0: examples/nemo_gym/run_grpo_nemo_gym.py — the gym-dedicated classic
#            entry; async comes from the yaml's native grpo.async_grpo block.
#            (The generic examples/run_grpo.py is NOT wired for nemo-gym: it
#            discards the gym actor instead of binding task_to_env["nemo_gym"],
#            and its configure_generation_config eos injection trips the gym
#            rollout stop-criteria assert.)
SC_MODE="${SC_MODE:-1}"
if [ "${SC_MODE}" = "1" ]; then
  ENTRYPOINT="./examples/run_grpo_single_controller.py"
  SC_OVERRIDES="++data_plane.enabled=true \
  ++data_plane.impl=transfer_queue \
  ++data_plane.backend=simple \
  ++data_plane.storage_capacity=1000000 \
  ++data_plane.num_storage_units=2 \
  ++data_plane.claim_meta_poll_interval_s=0.5 \
  ++data_plane.global_segment_size=549755813888 \
  ++data_plane.local_buffer_size=68719476736 \
  ++async_rl.max_weight_staleness_versions=${MAX_TRAJECTORY_AGE_STEPS} \
  ++async_rl.min_prompt_groups_per_batch=${MIN_PROMPT_GROUPS} \
  ++async_rl.max_inflight_prompts=${MIP} \
  ++async_rl.max_buffered_rollouts=$((PPS * (MAX_TRAJECTORY_AGE_STEPS + 1))) \
  ++async_rl.batch_selection_strategy=${BATCH_SELECTION_STRATEGY} \
  ++async_rl.over_sampling=${OVER_SAMPLING} \
  ++async_rl.force_in_order=${FORCE_IN_ORDER}"
else
  ENTRYPOINT="./examples/nemo_gym/run_grpo_nemo_gym.py"
  SC_OVERRIDES=""
fi

# ===== Training command =====
export COMMAND="NRL_VLLM_USE_V1=1 \
  NRL_WG_USE_RAY_REF=1 \
  WANDB_API_KEY=${WANDB_API_KEY} \
  HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
  HF_HOME=${HF_HOME} \
  HF_DATASETS_CACHE=${HF_DATASETS_CACHE} \
  UV_CACHE_DIR=${UV_CACHE_DIR} \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  VLLM_CACHE_ROOT=${LUSTRE_VLLM_CACHE} \
  DG_JIT_CACHE_DIR=${LUSTRE_VLLM_CACHE}/deep_gemm \
  VLLM_DEEP_GEMM_WARMUP=skip \
  NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-false} \
  NRL_SKIP_TQ_RUNTIME_ENV_PATCH=${NRL_SKIP_TQ_RUNTIME_ENV_PATCH:-1} \
  NRL_IGNORE_VERSION_MISMATCH=1 \
  RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
  UV_HTTP_TIMEOUT=3600 \
  UV_LOCK_TIMEOUT=900 \
  TORCH_CUDA_ARCH_LIST='9.0 10.0' \
  NEMO_GYM_SKIP_VENV_IF_PRESENT=1 \
  NEMO_GYM_VENV_DIR=${GYM_VENV_DIR} \
  uv run --frozen --extra mcore ${ENTRYPOINT} \
  --config=${CONFIG_FILE} \
  cluster.num_nodes=${NUM_NODES} \
  cluster.gpus_per_node=${GPUS_PER_NODE} \
  ++data.train.data_path=${TRAIN_DATA_PATH} \
  ++data.validation.data_path=${VAL_DATA_PATH} \
  grpo.num_prompts_per_step=${PPS} \
  grpo.num_generations_per_prompt=${GPP} \
  grpo.val_at_start=False \
  grpo.normalize_rewards=${NORMALIZE_REWARDS} \
  grpo.overlong_filtering=${OVERLONG_FILTERING} \
  grpo.val_period=${VAL_PERIOD} \
  grpo.seq_logprob_error_threshold=${SEQ_LOGPROB_ERROR_THRESHOLD} \
  ${EXTRA_GRPO} \
  ${SC_OVERRIDES} \
  ++policy.draft.enabled=false \
  ++policy.draft.model_name=null \
  ++policy.draft.loss_weight=0.1 \
  ++policy.draft.num_layers=null \
  ++policy.draft.aux_layer_indices=null \
  env.should_log_nemo_gym_responses=${LOG_GYM_RESPONSES} \
  policy.generation.colocated.enabled=${COLOCATED_ENABLED} \
  policy.model_name=${MODEL_PATH} \
  policy.max_total_sequence_length=${SEQLEN} \
  policy.dynamic_batching.enabled=False \
  policy.train_global_batch_size=${GBS} \
  policy.make_sequence_length_divisible_by=${MAKE_SEQ_DIVISIBLE_BY} \
  policy.offload_optimizer_for_logprob=true \
  policy.sequence_packing.enabled=${SEQUENCE_PACKING} \
  policy.megatron_cfg.tensor_model_parallel_size=${TP} \
  policy.megatron_cfg.expert_model_parallel_size=${EP} \
  policy.megatron_cfg.context_parallel_size=${CP} \
  policy.megatron_cfg.pipeline_model_parallel_size=${PP} \
  policy.megatron_cfg.sequence_parallel=True \
  policy.megatron_cfg.bias_activation_fusion=False \
  ++policy.megatron_cfg.use_fused_weighted_squared_relu=false \
  policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=${OVERLAP_GRAD_REDUCE} \
  policy.megatron_cfg.moe_permute_fusion=${MOE_PERMUTE_FUSION} \
  policy.megatron_cfg.moe_enable_deepep=${MOE_ENABLE_DEEPEP} \
  policy.megatron_cfg.moe_token_dispatcher_type=${MOE_TOKEN_DISPATCHER_TYPE} \
  policy.megatron_cfg.moe_aux_loss_coeff=${MOE_AUX_LOSS_COEFF} \
  policy.megatron_cfg.moe_router_load_balancing_type=${MOE_ROUTER_LOAD_BALANCING_TYPE} \
  policy.megatron_cfg.moe_router_bias_update_rate=${MOE_ROUTER_BIAS_UPDATE_RATE} \
  policy.megatron_cfg.freeze_moe_router=${MOE_FREEZE_ROUTER} \
  policy.megatron_cfg.optimizer.lr=${LR} \
  policy.megatron_cfg.optimizer.min_lr=${LR} \
  policy.megatron_cfg.optimizer.weight_decay=0 \
  policy.megatron_cfg.empty_unused_memory_level=2 \
  policy.megatron_cfg.activation_checkpointing=True \
  policy.generation.temperature=${TEMPERATURE} \
  policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP} \
  policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_GPU_UTIL} \
  policy.generation.vllm_cfg.skip_tokenizer_init=False \
  loss_fn.reference_policy_kl_penalty=${KL} \
  loss_fn.ratio_clip_min=${CLIP_MIN} \
  loss_fn.ratio_clip_max=${CLIP_MAX} \
  loss_fn.use_on_policy_kl_approximation=${USE_ON_POLICY_KL_APPROXIMATION} \
  loss_fn.use_importance_sampling_correction=${IMPORTANCE_SAMPLING_CORRECTION} \
  loss_fn.sequence_level_importance_ratios=${SEQ_LEVEL_IS} \
  loss_fn.token_level_loss=${TOKEN_LEVEL_LOSS} \
  loss_fn.force_on_policy_ratio=${FORCE_ON_POLICY_RATIO} \
  checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
  checkpointing.save_period=${SAVE_PERIOD} \
  checkpointing.keep_top_k=${KEEP_TOP_K} \
  ++checkpointing.metric_name=train:total_reward/mean \
  ++checkpointing.checkpoint_must_save_by=${MUST_SAVE_BY} \
  logger.wandb_enabled=True \
  logger.wandb.name=${WANDB_NAME} \
  logger.wandb.project=${WANDB_PROJ}"

# Async non-colocated: generation cluster + clipping + agent turn/timeout knobs.
export COMMAND="${COMMAND} \
  policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
  policy.generation.colocated.resources.gpus_per_node=${GPUS_PER_NODE} \
  grpo.advantage_clip_low=${ADVANTAGE_CLIP_LOW} \
  grpo.advantage_clip_high=${ADVANTAGE_CLIP_HIGH} \
  loss_fn.truncated_importance_sampling_ratio=${TIS_THRESHOLD} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT}"

# ---------------------------------------------------------------------------
# Mounts: the run tree at its own path (so ./examples/... resolves) and the
# gym source over the container's bundled copy. Export MOUNTS to replace, or
# EXTRA_MOUNTS to append (e.g. the filesystem holding MODEL_PATH/data).
# ---------------------------------------------------------------------------
GYM_CODE="${RUN_DIR}/3rdparty/Gym-workspace/Gym"
MOUNTS="${MOUNTS:-${RUN_DIR}:${RUN_DIR},${GYM_CODE}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym}"
[ -n "${EXTRA_MOUNTS:-}" ] && MOUNTS="${MOUNTS},${EXTRA_MOUNTS}"

mkdir -p "${CHECKPOINT_DIR}" "${BASE_LOG_DIR}"

# ray.sub reads these from the environment.
export CONTAINER MOUNTS COMMAND SETUP_COMMAND GPUS_PER_NODE CPUS_PER_WORKER BASE_LOG_DIR
[ -n "${UV_CACHE_DIR_OVERRIDE:-}" ] && export UV_CACHE_DIR_OVERRIDE

sbatch_args=(
    --nodes="${NUM_NODES}"
    --account="${ACCOUNT}"
    --job-name="${EXP_NAME}"
    --partition="${PARTITION}"
    --time="${TIME}"
    --gres=gpu:"${GPUS_PER_NODE}"
    --exclusive
    --dependency=singleton
    --output="${BASE_LOG_DIR}/slurm-%j.out"
)
[ -n "${REAPER_COMMENT}" ] && sbatch_args+=(--comment="${REAPER_COMMENT}")
# shellcheck disable=SC2206
[ -n "${SBATCH_EXTRA_ARGS:-}" ] && sbatch_args+=(${SBATCH_EXTRA_ARGS})

echo "=========================================="
echo "SWE async GRPO | Experiment: ${EXP_SUFFIX}"
echo "Entrypoint: ${ENTRYPOINT} (SC_MODE=${SC_MODE})"
echo "Run tree:   ${RUN_DIR} @ ${RUN_COMMIT}"
echo "Account:    ${ACCOUNT} / ${PARTITION}"
echo "Nodes: ${NUM_NODES} total (generation carves out ${NUM_GEN_NODES})    Time: ${TIME}"
echo "Container:  ${CONTAINER}"
echo "Parallelism: TP=${TP}, EP=${EP}, CP=${CP}, PP=${PP}, vLLM_TP=${VLLM_TP}, pad=${MAKE_SEQ_DIVISIBLE_BY}"
echo "Training: PPS=${PPS}, GPP=${GPP}, GBS=${GBS}, LR=${LR}, seqlen=${SEQLEN}, max_steps=${MAX_NUM_STEPS:-<yaml>}, min_prompt_groups=${MIN_PROMPT_GROUPS}, max_inflight_prompts=${MIP}"
echo "Streaming: over_sampling=${OVER_SAMPLING}, force_in_order=${FORCE_IN_ORDER}, age=${MAX_TRAJECTORY_AGE_STEPS}"
echo "Model: ${MODEL_PATH}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "WandB: ${WANDB_PROJ}/${WANDB_NAME}"
echo "=========================================="

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY_RUN] sbatch ${sbatch_args[*]} ${RUN_DIR}/ray.sub"
    echo "[DRY_RUN] COMMAND (first 400 chars): ${COMMAND:0:400}..."
    exit 0
fi

cd "${RUN_DIR}"
out=$(sbatch "${sbatch_args[@]}" "${RUN_DIR}/ray.sub")
echo "${out}"
job_id=$(echo "${out}" | grep -oE '[0-9]+' | head -1)
if [ -n "${job_id}" ]; then
    echo "Job ID: ${job_id}"
    echo "Monitor:  squeue -j ${job_id}"
    echo "Driver log: ${BASE_LOG_DIR}/${job_id}-logs/ray-driver.log"
fi
