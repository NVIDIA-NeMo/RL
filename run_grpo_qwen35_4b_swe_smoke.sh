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
REPO_ROOT="/lustre/fsw/portfolios/coreai/users/erinh/RL"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/grpo_qwen3.5_4b_async_swe_smoke_arm.yaml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/results}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/lustre/fsw/portfolios/llmservice/users/sdevare/repos/ultra/datasets/swe/blends/balanced_language.jsonl}"
VAL_DATA_PATH="${VAL_DATA_PATH:-/lustre/fsw/portfolios/llmservice/users/sdevare/repos/ultra/datasets/swe/swe_public_datasets_val_swebench.jsonl}"
DEFAULT_MODEL_PATH="/lustre/fsw/portfolios/nemotron/users/ruit/hf_models/Qwen3.5-4B"
MODEL_PATH="${1:-${MODEL_PATH:-${DEFAULT_MODEL_PATH}}}"

# ================ Container and mount config ================
# GB200 (aarch64) baked image: apptainer + /opt/nemo_rl_venv with --extra mcore
# (sm_100), built by test_assets/SWE/build_swe_bench_combined.sh.
export CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/nemotron/users/ruit/enroot-images/ruit-swe_bench-6dc8fabea-aarch64-060426-mcore-apptainer.squashfs}
GYM_CODE="${REPO_ROOT}/3rdparty/Gym-workspace/Gym"
export MOUNTS="/lustre:/lustre,$PWD:$PWD,${GYM_CODE}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"

# ======================= Cluster / resources =======================
NUM_GPU=4                                          # GB200: ray.sub asserts == gres gpu:4
export GPUS_PER_NODE=${NUM_GPU}
export CPUS_PER_WORKER=${CPUS_PER_WORKER:-140}     # GB200 nodes have 144 CPUs
NUM_ACTOR_NODES=${NUM_NODES:-2}
NUM_GENERATION_NODES=${NUM_GEN_NODES:-1}           # only used in async (non-colocated) mode

# ============================ Parallelism ============================
TP=2
EP=1
CP=1
PP=1
VLLM_TP=1
MAKE_SEQ_DIVISIBLE_BY=8

# ===================== Sequence length & packing =====================
SEQLEN=65536
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
WANDB_GROUP="${WANDB_GROUP:-qwen3.5-4b-gb200-swe-smoke}"
LOG_GYM_RESPONSES=true

# ========================= SLURM submission =========================
SBATCH_ACCOUNT="coreai_comparch_trtllm"
SBATCH_PARTITION="batch"
SBATCH_TIME="${SBATCH_TIME:-4:0:0}"

# ========================= Experiment naming =========================
if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  SYNC_MODE="async-age${MAX_TRAJECTORY_AGE_STEPS}"
else
  SYNC_MODE="sync"
fi
EXP_SUFFIX="${EXP_SUFFIX:-qwen3.5-4b-gb200-swe-smoke-vllm-${SYNC_MODE}-64k-steps${MAX_NUM_STEPS}-turns${AGENT_MAX_TURNS}-nodes${NUM_ACTOR_NODES}-gen${NUM_GENERATION_NODES}-tp${TP}-pps${PPS}-gpp${GPP}-gbs${GBS}-lr${LR}}"
WANDB_NAME="${EXP_SUFFIX}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${EXP_SUFFIX}"
LOG_DIR="logs/exp_${EXP_SUFFIX}"
SNAPSHOT_DIR="${REPO_ROOT}"

mkdir -p "${CHECKPOINT_DIR}"

# ============= Unified SLURM/Ray log location =============
export BASE_LOG_DIR="${BASE_LOG_DIR:-${SNAPSHOT_DIR}/logs/qwen35_4b_swe_smoke}"
mkdir -p "${BASE_LOG_DIR}"

# ========================= Environment variables =========================
if [ -f "/lustre/fsw/portfolios/coreai/users/erinh/script/env.sh" ]; then
  # shellcheck disable=SC1090
  source "/lustre/fsw/portfolios/coreai/users/erinh/script/env.sh"
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
PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fsw/portfolios/nemotron/users/ruit/.cache/qwen3.5_4b_swe}"
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
  apt-get update && apt-get install -y git build-essential gcc wget 2>/dev/null || true
  cd /tmp && \
  wget --no-check-certificate -q https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_arm64.deb && \
  apt install -y ./apptainer_1.3.1_arm64.deb && \
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
export COMMAND="NRL_VLLM_USE_V1=1 \
  NRL_WG_USE_RAY_REF=1 \
  WANDB_API_KEY=${WANDB_API_KEY} \
  HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
  GITHUB_TOKEN=${GITHUB_TOKEN} \
  GITLAB_TOKEN=${GITLAB_TOKEN} \
  HF_HOME=${HF_HOME} \
  HF_DATASETS_CACHE=${HF_DATASETS_CACHE} \
  UV_CACHE_DIR=${UV_CACHE_DIR} \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  VLLM_CACHE_ROOT=${LUSTRE_VLLM_CACHE} \
  DG_JIT_CACHE_DIR=${LUSTRE_VLLM_CACHE}/deep_gemm \
  VLLM_DEEP_GEMM_WARMUP=skip \
  NRL_FORCE_REBUILD_VENVS=false \
  NRL_IGNORE_VERSION_MISMATCH=1 \
  RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
  UV_HTTP_TIMEOUT=3600 \
  UV_LOCK_TIMEOUT=900 \
  TORCH_CUDA_ARCH_LIST='10.0' \
  NEMO_GYM_SKIP_VENV_IF_PRESENT=1 \
  uv run --frozen --extra mcore ./examples/nemo_gym/run_grpo_nemo_gym.py \
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
  policy.max_total_sequence_length=${SEQLEN} \
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
  policy.generation.temperature=${TEMPERATURE} \
  policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP} \
  policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_GPU_UTIL} \
  policy.generation.vllm_cfg.max_model_len=${SEQLEN} \
  policy.generation.vllm_cfg.skip_tokenizer_init=False \
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

# ================ Submit job ================
sbatch \
  --nodes="${NUM_ACTOR_NODES}" \
  --account="${SBATCH_ACCOUNT}" \
  --job-name="${WANDB_NAME}" \
  --partition="${SBATCH_PARTITION}" \
  --time="${SBATCH_TIME}" \
  --gres=gpu:${NUM_GPU} \
  --output="${BASE_LOG_DIR}/slurm-%j.out" \
  --exclusive \
  --dependency=singleton \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"data_loading","description":"Async GRPO Qwen3.5-4B GB200 SWE smoke"}}' \
  ray.sub | tee /dev/stderr | grep -o '[0-9]\+' > latest_qwen35_4b_swe_smoke_job_id.txt

JOB_ID="$(cat latest_qwen35_4b_swe_smoke_job_id.txt)"
echo "=========================================="
echo "Job submitted: ${EXP_SUFFIX}"
echo "Job ID: ${JOB_ID}"
echo "wandb group: ${WANDB_GROUP}"
echo "Monitor with: squeue -j ${JOB_ID}"
echo "Ray/SLURM logs: ${BASE_LOG_DIR}/${JOB_ID}-logs/"
echo "Checkpoints: ${CHECKPOINT_DIR}/"
echo "=========================================="

cd - > /dev/null
