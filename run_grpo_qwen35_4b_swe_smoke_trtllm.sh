#!/bin/bash
# ============================================================================
# Async GRPO SWE smoke launcher: Qwen3.5-4B on
# oci-hsg-cs-001 / nemotron_sw_post (GB200, aarch64, 4 GPU/node).
#
# Reorganized to match test_assets/qwen-30B/run_grpo_qwen3_30b_swe_scale_gen.sh
# conventions (REPO_ROOT auto-derive, aarch64 container, ${HOME} env source,
# Lustre cache seeding, arm64 apptainer, sm_100 arch, 4 GPU/node). The Qwen3.5-4B
# model, dense-model parallelism, and per-run smoke knobs are kept.
#
# Geometry (fixed, fits 4-GPU GB200 nodes):
#   TRAIN_NODES = NUM_ACTOR_NODES - NUM_GENERATION_NODES   (non-colocated async)
#   train world = TRAIN_NODES * NUM_GPU,  train DP = train_world / TP
#   gen replicas = NUM_GENERATION_NODES * NUM_GPU / VLLM_TP
#
# Usage:  bash test_assets/qwen35-4B/run_grpo_qwen35_4b_swe_smoke.sh
# Optional env: NUM_NODES, NUM_GEN_NODES, EXP_SUFFIX, MODEL_PATH, CONTAINER,
#               MAX_NUM_STEPS, SBATCH_TIME, PERSISTENT_CACHE, BASE_LOG_DIR.
# ============================================================================

set -e

# ============================ Paths ============================
# REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# lyris: repo lives under coreai_comparch_trtllm (oci-hsg path was portfolios/coreai)
REPO_ROOT="/lustre/fsw/coreai_comparch_trtllm/erinh/RL"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/grpo_qwen3.5_4b_async_swe_smoke_arm_trtllm.yaml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/results}"
# lyris: SWE blend jsonls copied into the repo root (oci-hsg path was sdevare/repos/ultra/...)
# lyris-covered subsets (only instances with a present .sif): 11 train / 182 val.
# Full blends barely overlap the lyris gym_sifs (11/921, 182/405) + shuffle is off, so the
# full files would fail on the first uncovered instance. Use the subsets for the smoke.
# 2026-06-11: switched train to SWE-bench_Verified instances with a present SIF (202 tasks,
# all SIF-backed) so a 50-step run has distinct tasks (the old subset had only 10 -> cycled).
# Built by make_swebench_jsonl.py (format-validated against gym data/example.jsonl). See
# run_logs_lyris.md "Train jsonl: SWE-bench_Verified all-SIF". Old subset (10 tasks):
#   ${REPO_ROOT}/balanced_language_lyris_subset.jsonl
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${REPO_ROOT}/swebench_verified_lyris_sif.jsonl}"
VAL_DATA_PATH="${VAL_DATA_PATH:-${REPO_ROOT}/swe_public_datasets_val_swebench_lyris_subset.jsonl}"
# lyris: swapped from Qwen3.5-4B (Qwen3-Next GDN/MoE -> trtllm partial-weight-refit
# unsupported, see run_logs_lyris.md) to dense Qwen3-4B-Instruct-2507 for trtllm pipe-clean.
DEFAULT_MODEL_PATH="/lustre/fsw/coreai_comparch_trtllm/erinh/llm-models/Qwen3-4B-Instruct-2507"
MODEL_PATH="${1:-${MODEL_PATH:-${DEFAULT_MODEL_PATH}}}"

# ================ Container and mount config ================
# GB200 (aarch64) baked image: apptainer + /opt/nemo_rl_venv with --extra mcore
# (sm_100), built by test_assets/SWE/build_swe_bench_combined.sh.
# lyris: Shiki's trtllm sqsh (NOT ruit's prebaked SWE image). It has mcore but NOT apptainer,
# so apptainer is installed at runtime via the SETUP_COMMAND PPA block below.
# [manual run] NOT needed — you're already attached inside the enroot container on a running
# 2-node Ray cluster, so no container launch / bind-mounts happen here. Kept for reference.
# export CONTAINER=${CONTAINER:-/lustre/fsw/coreai_comparch_trtllm/shikiw/images/nemo-rl-py313-trtllm.sqsh}
# GYM_CODE="${REPO_ROOT}/3rdparty/Gym-workspace/Gym"
# export MOUNTS="/lustre:/lustre,$PWD:$PWD,/dev/fuse:/dev/fuse,${GYM_CODE}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"

# ======================= Cluster / resources =======================
NUM_GPU=4                                          # GB200: ray.sub asserts == gres gpu:4
export GPUS_PER_NODE=${NUM_GPU}
export CPUS_PER_WORKER=${CPUS_PER_WORKER:-140}     # GB200 nodes have 144 CPUs
# Ray 2.54 turns OpenTelemetry metrics ON by default; its recorder segfaults at core-worker
# startup on this aarch64 build (getenv race in OpenTelemetryMetricRecorder::Start). Disable it.
# NeMo-RL forwards the driver's os.environ to all workers (virtual_cluster.py:110), so exporting
# here reaches every Ray worker process.
export RAY_enable_open_telemetry=0
# Qwen3.5 is a hybrid Gated-Delta-Net model. TRT-LLM's default flashinfer Blackwell (sm_100) GDN
# prefill kernel fails to JIT-compile (cutlass-dsl ICE: tcgen05.make_tmem_copy "failed to legalize"
# in flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py). Force the vendored Triton
# chunk_gated_delta_rule fallback instead (tensorrt_llm/_torch/modules/mamba/gdn_mixer.py:42).
export TLLM_USE_FLASHINFER_GDN_PREFILL=0
NUM_ACTOR_NODES=${NUM_NODES:-2}
NUM_GENERATION_NODES=${NUM_GEN_NODES:-1}           # only used in async (non-colocated) mode

# ============================ Parallelism ============================
TP=2
EP=1
# 2026-06-11: 0611h/0611i OOM'd at the logprob log-softmax (`_compute_distributed_log_softmax`,
# vocab_parallel_logits.exp()) on long SWE-bench trajectories — ~29GB spike, killed at step 12-14.
# max_total_sequence_length does NOT shrink the logprob forward (only masks loss), so seqlen caps
# were ineffective (29.5GB at 65k ≈ 28GB at 131k). Context parallelism shards the sequence dim
# across CP GPUs → ~halves the per-GPU logits. Train world = 1 node x 4 GPU; TP2 x CP2 = 4 (DP1).
# 2026-06-11 REVERTED to CP=1: CP requires sequence_packing, and packing bin = max_total_seq_len/CP,
# so CP halves the bin → an 83k SWE-bench trajectory can't fit unless seqlen is raised, which re-OOMs
# the per-rank logprob (circular on 4 GPUs; CP=4 impossible at tp2). Proven CP1/no-packing reaches
# step ~14; cap MAX_NUM_STEPS below that for a clean e2e verification. Full 50-step long-context
# training needs more GPUs (CP4/PP) or chunked-logprob — a supervised effort.
CP=1
PP=1
VLLM_TP=1
MAKE_SEQ_DIVISIBLE_BY=8

# ===================== Sequence length & packing =====================
# 2026-06-11: 65536 FAILED on real SWE-bench_Verified prompts — TRT-LLM raised
# `RequestError: prompt length (66042) should not exceed max_num_tokens (65536)` → 500 loop
# (see run_logs_lyris.md). Bumped to 131072 per project memory (SWE-bench agent rollouts need
# max_num_tokens=max_seq_len=131k; 32k/65k fail late). NOTE: TRT-LLM's max_num_tokens does NOT
# derive from max_seq_len (default stays 65536) — it must be set EXPLICITLY via
# trtllm_cfg.max_num_tokens=${SEQLEN} (added to the overrides below); bumping max_model_len alone
# is insufficient (verified: 0611g had max_seq_len=131072 but max_num_tokens=65536 → same crash).
SEQLEN=131072
# 2026-06-11: training-side logprob logits OOM'd at 131k (tp2). Fix = CP=2 + sequence_packing (below),
# which shards the forward across CP GPUs. With packing, max_total_sequence_length is the PACK BIN and
# must be >= the longest trajectory; generation caps trajectories at 131072, so set the bin to 131072
# (fits all — a 65k bin rejected an 83016-token trajectory in 0611k). CP=2 shards ~65k/GPU → fits.
TRAIN_SEQLEN=131072
# 2026-06-11: required True for CP=2 (0611j died: "Sequence Packing must be enabled to use Context
# Parallelism with MCore"). Also caps the forward at the pack size (max_total_sequence_length), which
# with CP=2 shards the logprob logits → fixes the long-SWE-bench-trajectory OOM (steps 12-14).
# 2026-06-11 REVERTED to False (CP reverted to 1; see CP comment). Proven regime.
SEQUENCE_PACKING=False

# ================= Sync/Async mode & async GRPO settings =================
ASYNC_GRPO_ENABLED=True
MAX_TRAJECTORY_AGE_STEPS=1
FORCE_ON_POLICY_RATIO=True
INFLIGHT_WEIGHT_UPDATE=False
RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES=False
SEQ_LOGPROB_ERROR_THRESHOLD=null
if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  COLOCATED_ENABLED=False
  VLLM_GPU_UTIL=0.8
  OVERLAP_GRAD_REDUCE=False
  TIS_THRESHOLD=5
else
  COLOCATED_ENABLED=True
  VLLM_GPU_UTIL=0.5
  OVERLAP_GRAD_REDUCE=True
fi

# ========================= GRPO / sampling =========================
PPS=1
GPP=4
GBS=4
MAX_NUM_STEPS="${MAX_NUM_STEPS:-50}"
NORMALIZE_REWARDS=True
OVERLONG_FILTERING=True

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

# ======================= Generation / vLLM =======================
TEMPERATURE=1.0

# =================== Checkpointing & validation ===================
SAVE_PERIOD=5
VAL_PERIOD=50
KEEP_TOP_K=2

# ============================ SWE agent ============================
AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-50}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-1800}"

# ============================== Logging ==============================
WANDB_PROJ="${WANDB_PROJ:-nemo-rl-swe-benchmark-smoke-erinh}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-4b-2507-gb200-swe-smoke}"
LOG_GYM_RESPONSES=true

# ========================= SLURM submission =========================
# [manual run] NOT needed — no sbatch allocation here (cluster already up). Kept for reference.
# SBATCH_ACCOUNT="coreai_comparch_trtllm"
# SBATCH_PARTITION="gb200"   # lyris (oci-hsg was 'batch')
# SBATCH_TIME="${SBATCH_TIME:-4:0:0}"

# ========================= Experiment naming =========================
if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  SYNC_MODE="async-age${MAX_TRAJECTORY_AGE_STEPS}"
else
  SYNC_MODE="sync"
fi
EXP_SUFFIX="${EXP_SUFFIX:-qwen3-4b-2507-gb200-swe-smoke-trtllm-${SYNC_MODE}-64k-steps${MAX_NUM_STEPS}-turns${AGENT_MAX_TURNS}-nodes${NUM_ACTOR_NODES}-gen${NUM_GENERATION_NODES}-tp${TP}-pps${PPS}-gpp${GPP}-gbs${GBS}-lr${LR}}"
WANDB_NAME="${EXP_SUFFIX}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${EXP_SUFFIX}"
LOG_DIR="logs/exp_${EXP_SUFFIX}"
SNAPSHOT_DIR="${REPO_ROOT}"

mkdir -p "${CHECKPOINT_DIR}"

# ============= Unified SLURM/Ray log location =============
export BASE_LOG_DIR="${BASE_LOG_DIR:-${SNAPSHOT_DIR}/logs/qwen35_4b_swe_smoke}"
mkdir -p "${BASE_LOG_DIR}"

# ========================= Environment variables =========================
if [ -f "/lustre/fsw/coreai_comparch_trtllm/erinh/launch_scripts/env.sh" ]; then
  # shellcheck disable=SC1090
  source "/lustre/fsw/coreai_comparch_trtllm/erinh/launch_scripts/env.sh"
fi
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN}}"
export GITLAB_TOKEN="${GITLAB_TOKEN:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export UV_CACHE_DIR=/tmp/uv_cache
export LUSTRE_UV_CACHE_SEED="${LUSTRE_UV_CACHE_SEED:-}"
export UV_LOCK_TIMEOUT=3600
export RAY_DEDUP_LOGS=1
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export OMP_NUM_THREADS=16

# ========================= Node-local cache config =========================
# HOME has a 10G quota on this cluster -> persistent caches live on Lustre.
PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fsw/coreai_comparch_trtllm/erinh/.cache/qwen3.5_4b_swe}"
export LUSTRE_VLLM_CACHE="${PERSISTENT_CACHE}/vllm_compile_cache"
export LUSTRE_INDUCTOR_CACHE="${PERSISTENT_CACHE}/inductor_cache"
export LUSTRE_TRITON_CACHE="${PERSISTENT_CACHE}/triton_cache"
export NRL_VLLM_LOCAL_CACHE_DIR="/tmp/nemo_rl_vllm_cache"
export NRL_VLLM_CACHE_SEED_DIR="/tmp/nemo_rl_vllm_cache_warm"
export INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
export TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
export CACHE_SYNC_FREQUENCY=120
mkdir -p "${LUSTRE_VLLM_CACHE}" "${LUSTRE_INDUCTOR_CACHE}" "${LUSTRE_TRITON_CACHE}"

# ============================== Summary ==============================
echo "=========================================="
echo "Qwen3.5-4B GB200 SWE smoke | Experiment: ${EXP_SUFFIX}"
echo "Mode: ${SYNC_MODE}, Colocated: ${COLOCATED_ENABLED}"
echo "Nodes: ${NUM_ACTOR_NODES} (gen=${NUM_GENERATION_NODES}, train=$((NUM_ACTOR_NODES - NUM_GENERATION_NODES))), GPUs/node: ${NUM_GPU}"
echo "Parallelism: TP=${TP}, EP=${EP}, CP=${CP}, PP=${PP}, vLLM_TP=${VLLM_TP}, pad=${MAKE_SEQ_DIVISIBLE_BY}"
echo "Training: PPS=${PPS}, GPP=${GPP}, GBS=${GBS}, LR=${LR}, max_steps=${MAX_NUM_STEPS}, seqlen=${SEQLEN}"
echo "Model: ${MODEL_PATH}"
echo "Container: ${CONTAINER}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "=========================================="

cd "${SNAPSHOT_DIR}"

# ================ SETUP_COMMAND (self-skips apptainer install if baked; seed caches) ================
read -r -d '' SETUP_COMMAND <<SETUPEOF || true
echo "[SETUP] Ensuring apptainer (arm64) for SWE sandbox..."
RET=1
RETRIES=3
for attempt in \$(seq 1 \$RETRIES); do
  if command -v apptainer >/dev/null 2>&1 || command -v singularity >/dev/null 2>&1; then
    echo "[SETUP] singularity/apptainer already available"
    RET=0
    break
  fi
  # lyris: NO arm64 .deb exists on apptainer GitHub releases (amd64 only) -> the v1.3.1 .deb
  # URL 404s. Use the apptainer PPA (same as Rayen's build script). add-apt-repository's shebang
  # python (3.13) lacks apt_pkg, so run it under the system python3.12.
  export DEBIAN_FRONTEND=noninteractive
  apt-get update && apt-get install -y git build-essential gcc wget python3-apt software-properties-common 2>/dev/null || true
  /usr/bin/python3.12 /usr/bin/add-apt-repository -y ppa:apptainer/ppa && \
  apt-get update && \
  apt-get install -y apptainer apptainer-suid && \
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
  echo "[SETUP] WARNING: apptainer not available after \$RETRIES attempts"
fi

echo "[CACHE SEED] Clearing stale /tmp caches and seeding from Lustre..."
rm -rf /tmp/nemo_rl_vllm_cache /tmp/nemo_rl_vllm_cache_*
rm -rf "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
mkdir -p "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

find "${LUSTRE_INDUCTOR_CACHE}" -maxdepth 1 -name '.tmp_*' -mmin +30 -exec rm -rf {} + 2>/dev/null || true
find "${LUSTRE_TRITON_CACHE}" -maxdepth 1 -name '.tmp_*' -mmin +30 -exec rm -rf {} + 2>/dev/null || true

_seed_cache() {
  local lustre="\$1" local_dir="\$2" name="\$3"
  if [ -d "\$lustre" ] && [ "\$(ls -A "\$lustre" 2>/dev/null)" ]; then
    rsync -a --exclude '.tmp_*' "\$lustre/" "\$local_dir/" 2>/dev/null \
      && echo "[CACHE SEED] \$name: seeded from Lustre" \
      || echo "[CACHE SEED] \$name: seed failed (non-fatal)"
  else
    echo "[CACHE SEED] \$name: no warm cache on Lustre yet"
  fi
}

_seed_cache "${LUSTRE_INDUCTOR_CACHE}" "${INDUCTOR_CACHE_DIR}" "Inductor"
_seed_cache "${LUSTRE_TRITON_CACHE}" "${TRITON_CACHE_DIR}" "Triton"
mkdir -p /tmp/uv_cache
_seed_cache "${LUSTRE_UV_CACHE_SEED}" "/tmp/uv_cache" "uv (prebuilt transformer-engine)"
echo "[CACHE SEED] Done."

UV_HTTP_TIMEOUT=3600 \
  uv sync --frozen --extra mcore
SETUPEOF
export SETUP_COMMAND

# ================ Training command ================
# vLLM-only env vars removed for the trtllm backend (can't be #-commented inside the
# COMMAND string). Originals, for reference:
#   NRL_VLLM_USE_V1=1
#   VLLM_ATTENTION_BACKEND=FLASH_ATTN
#   VLLM_CACHE_ROOT=${LUSTRE_VLLM_CACHE}
#   DG_JIT_CACHE_DIR=${LUSTRE_VLLM_CACHE}/deep_gemm
#   VLLM_DEEP_GEMM_WARMUP=skip
export COMMAND="NRL_WG_USE_RAY_REF=1 \
  WANDB_API_KEY=${WANDB_API_KEY} \
  HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
  GITHUB_TOKEN=${GITHUB_TOKEN} \
  GITLAB_TOKEN=${GITLAB_TOKEN} \
  HF_HOME=${HF_HOME} \
  HF_DATASETS_CACHE=${HF_DATASETS_CACHE} \
  UV_CACHE_DIR=${UV_CACHE_DIR} \
  NRL_FORCE_REBUILD_VENVS=false \
  NRL_IGNORE_VERSION_MISMATCH=1 \
  RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
  UV_HTTP_TIMEOUT=3600 \
  UV_LOCK_TIMEOUT=900 \
  TORCH_CUDA_ARCH_LIST='10.0' \
  NEMO_GYM_SKIP_VENV_IF_PRESENT=1 \
  uv run --no-sync --extra mcore ./examples/nemo_gym/run_grpo_nemo_gym.py \
  --config=${CONFIG_FILE} \
  cluster.num_nodes=${NUM_ACTOR_NODES} \
  cluster.gpus_per_node=${NUM_GPU} \
  ++data.train.data_path=${TRAIN_DATA_PATH} \
  ++data.validation.data_path=${VAL_DATA_PATH} \
  logger.log_dir=${LOG_DIR} \
  logger.wandb_enabled=True \
  logger.wandb.name=${WANDB_NAME} \
  logger.wandb.project=${WANDB_PROJ} \
  ++logger.wandb.group=${WANDB_GROUP} \
  grpo.max_num_steps=${MAX_NUM_STEPS} \
  grpo.num_prompts_per_step=${PPS} \
  grpo.num_generations_per_prompt=${GPP} \
  grpo.val_at_start=False \
  grpo.val_at_end=False \
  grpo.normalize_rewards=${NORMALIZE_REWARDS} \
  grpo.overlong_filtering=${OVERLONG_FILTERING} \
  grpo.val_period=${VAL_PERIOD} \
  grpo.seq_logprob_error_threshold=${SEQ_LOGPROB_ERROR_THRESHOLD} \
  grpo.async_grpo.enabled=${ASYNC_GRPO_ENABLED} \
  grpo.async_grpo.in_flight_weight_updates=${INFLIGHT_WEIGHT_UPDATE} \
  grpo.async_grpo.recompute_kv_cache_after_weight_updates=${RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES} \
  grpo.async_grpo.max_trajectory_age_steps=${MAX_TRAJECTORY_AGE_STEPS} \
  env.should_log_nemo_gym_responses=${LOG_GYM_RESPONSES} \
  policy.generation.colocated.enabled=${COLOCATED_ENABLED} \
  policy.generation.colocated.resources.num_nodes=${NUM_GENERATION_NODES} \
  policy.generation.colocated.resources.gpus_per_node=${NUM_GPU} \
  policy.model_name=${MODEL_PATH} \
  policy.max_total_sequence_length=${TRAIN_SEQLEN} \
  policy.dynamic_batching.enabled=False \
  policy.train_global_batch_size=${GBS} \
  policy.train_micro_batch_size=1 \
  policy.logprob_batch_size=1 \
  policy.make_sequence_length_divisible_by=${MAKE_SEQ_DIVISIBLE_BY} \
  policy.sequence_packing.enabled=${SEQUENCE_PACKING} \
  policy.megatron_cfg.tensor_model_parallel_size=${TP} \
  policy.megatron_cfg.pipeline_model_parallel_size=${PP} \
  policy.megatron_cfg.expert_model_parallel_size=${EP} \
  policy.megatron_cfg.expert_tensor_parallel_size=1 \
  policy.megatron_cfg.context_parallel_size=${CP} \
  policy.megatron_cfg.sequence_parallel=True \
  policy.megatron_cfg.apply_rope_fusion=False \
  policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=${OVERLAP_GRAD_REDUCE} \
  policy.megatron_cfg.optimizer.lr=${LR} \
  policy.megatron_cfg.optimizer.min_lr=${LR} \
  policy.megatron_cfg.optimizer.weight_decay=0 \
  policy.megatron_cfg.empty_unused_memory_level=2 \
  policy.megatron_cfg.activation_checkpointing=True \
  policy.generation.backend=trtllm \
  policy.generation.temperature=${TEMPERATURE} \
  policy.generation.trtllm_cfg.tensor_parallel_size=${VLLM_TP} \
  policy.generation.trtllm_cfg.gpu_memory_utilization=${VLLM_GPU_UTIL} \
  policy.generation.trtllm_cfg.max_model_len=${SEQLEN} \
  policy.generation.trtllm_cfg.max_num_tokens=${SEQLEN} \
  policy.generation.trtllm_cfg.async_engine=True \
  policy.generation.trtllm_cfg.expose_http_server=True \
  policy.generation.trtllm_cfg.in_flight_weight_updates=${INFLIGHT_WEIGHT_UPDATE} \
  policy.generation.trtllm_cfg.recompute_kv_cache_after_weight_updates=${RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES} \
  loss_fn.reference_policy_kl_penalty=${KL} \
  loss_fn.ratio_clip_min=${CLIP_MIN} \
  loss_fn.ratio_clip_max=${CLIP_MAX} \
  loss_fn.use_on_policy_kl_approximation=${USE_ON_POLICY_KL_APPROXIMATION} \
  loss_fn.use_importance_sampling_correction=${IMPORTANCE_SAMPLING_CORRECTION} \
  loss_fn.sequence_level_importance_ratios=${SEQ_LEVEL_IS} \
  loss_fn.token_level_loss=${TOKEN_LEVEL_LOSS} \
  loss_fn.force_on_policy_ratio=${FORCE_ON_POLICY_RATIO} \
  loss_fn.truncated_importance_sampling_ratio=${TIS_THRESHOLD} \
  checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
  checkpointing.save_period=${SAVE_PERIOD} \
  checkpointing.keep_top_k=${KEEP_TOP_K} \
  ++checkpointing.metric_name=train:total_reward/mean \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT}"

# ================ Run directly on the EXISTING Ray cluster (manual, in-enroot) ================
# [manual run] No node allocation / container launch here — you're already attached inside the
# enroot container on the head of a running 2-node Ray cluster, so we just launch the driver
# (it connects to the running cluster). Prereq: source node_init first so PATH has uv +
# /opt/nemo_rl_venv and CPATH/CUDA are set.
#
# --- Original sbatch + ray.sub allocation (commented out; that's for launching from the login node) ---
# sbatch \
#   --nodes="${NUM_ACTOR_NODES}" \
#   --account="${SBATCH_ACCOUNT}" \
#   --job-name="${WANDB_NAME}" \
#   --partition="${SBATCH_PARTITION}" \
#   --time="${SBATCH_TIME}" \
#   --gres=gpu:${NUM_GPU} \
#   --output="${BASE_LOG_DIR}/slurm-%j.out" \
#   --exclusive \
#   --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"data_loading","description":"Async GRPO Qwen3.5-4B GB200 SWE smoke"}}' \
#   ray.sub | tee /dev/stderr | grep -o '[0-9]\+' > latest_qwen35_4b_swe_smoke_job_id.txt

# Optional one-time per-node setup (cache seed + uv sync; apptainer is already installed). Run
# once on the head before the first launch if needed:  eval "$SETUP_COMMAND"

echo "=========================================="
echo "Launching driver on existing Ray cluster | ${EXP_SUFFIX}"
echo "Config: ${CONFIG_FILE}"
echo "wandb:  ${WANDB_PROJ} / ${WANDB_GROUP} / ${WANDB_NAME}"
echo "=========================================="
bash -c "${COMMAND}"
