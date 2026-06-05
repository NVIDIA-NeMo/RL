#!/bin/bash
# ============================================================================
# GENERATION-SCALING launcher for async SWE GRPO (derived from
# run_grpo_repro_bihu_swe2.sh / bihu dc3m70us).
#
# Single knob:  NUM_VLLM_REPLICAS (R)  -> number of vLLM generation replicas.
# Everything else is auto-derived to hold these invariants constant so that
# runs at different R are directly comparable:
#   - per generation-replica workload : samples/replica/step = 2
#   - per training-GPU workload       : GBS / train_DP       = 32
#   - train:gen node ratio            : 1:1 (matches the bihu 8+8 baseline)
#
# Derivation (REPLICAS_PER_NODE = gpus_per_node / VLLM_TP = 8/2 = 4):
#   GEN_NODES   = R / 4
#   TRAIN_NODES = R / 4                 (linear follow; override with TRAIN_NODES=)
#   TOTAL_NODES = TRAIN_NODES+GEN_NODES = R/2   -> sbatch --nodes & cluster.num_nodes
#   PPS         = 2*R / GPP             = R/4
#   GBS         = PPS*GPP               = 2*R   (force_on_policy_ratio requires ==)
#   CONCURRENCY = max(768, GBS*age)
# R must be a multiple of 16 (train world = 2R must satisfy Megatron
# model-parallel & expert-parallel divisibility; gen must fill whole nodes).
# R=32 exactly reproduces the bihu repro (16 nodes = 8+8, PPS=8, GBS=64).
#
# All runs of this sweep share one wandb group (WANDB_GROUP) under project
# swe-benchmark for easy comparison.
#
# Usage:
#   NUM_VLLM_REPLICAS=64 bash examples/swe_bench/run_grpo_swe2_scale_gen.sh
#   NUM_VLLM_REPLICAS=64 DRY_RUN=1 bash examples/swe_bench/run_grpo_swe2_scale_gen.sh   # print config, no submit
#   SKIP_TRAINING=1 NUM_VLLM_REPLICAS=4 bash examples/swe_bench/run_grpo_swe2_scale_gen.sh  # generation-only (no-op train, 1 node, R%4)
# Optional env: SKIP_TRAINING, TRAIN_NODES, WANDB_GROUP, EXP_SUFFIX, MODEL_PATH, CONTAINER,
#               MAX_NUM_STEPS, SBATCH_TIME, PERSISTENT_CACHE, BASE_LOG_DIR
# Credentials are NOT sourced here — export HF_HOME / HF_TOKEN / WANDB_API_KEY yourself.
# ============================================================================

set -e

# ============================ Paths ============================
# Auto-detected from this script's location (examples/swe_bench/), so it works from
# any clone of the repo. Override by exporting REPO_ROOT.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_FILE="${REPO_ROOT}/examples/swe_bench/grpo_qwen3_30b_async_swe.yaml"
CHECKPOINT_ROOT="${REPO_ROOT}/results"
TRAIN_DATA_PATH="/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/nano/dataset/rl/swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl"
VAL_DATA_PATH="${TRAIN_DATA_PATH}"
# SWE1 step_230 HF checkpoint (exactly what dc3m70us trained from).
DEFAULT_MODEL_PATH="/lustre/fsw/portfolios/coreai/users/bihu/repos/nemo-rl-async-swe/results/qwen3-30b-thinking-swe1-async-age1-pps64-gpp8-gbs512-lr1e-06/step_230_hf"
MODEL_PATH="${1:-${MODEL_PATH:-${DEFAULT_MODEL_PATH}}}"

# ================ Container and mount config ================
# SWE training container (mcore + apptainer, working hermes tool parser).
export CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/coreai/users/ruit/enroot-images/docker_images:ruit-swe_bench-6de99f772-x86_64-060326-mcore-apptainer.squashfs}
GYM_CODE="${REPO_ROOT}/3rdparty/Gym-workspace/Gym"
export MOUNTS="/lustre:/lustre,$PWD:$PWD,${GYM_CODE}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"

# ======================= Cluster / resources =======================
NUM_GPU=8
export GPUS_PER_NODE=${NUM_GPU}
export CPUS_PER_WORKER=114

# ============================ Parallelism ============================
# SKIP_TRAINING=1 -> generation-only benchmark: training is a no-op on a SINGLE node
# (no optimizer, weights frozen, refit every step + keep-alive matmul). Training
# parallelism must fit 1 node, so model_parallel = TP*CP*PP must divide gpus_per_node(=8).
SKIP_TRAINING="${SKIP_TRAINING:-0}"
if [ "${SKIP_TRAINING}" = "1" ]; then
  TP=8; EP=8; CP=1; PP=1; ETP=1     # model_parallel = 8 (fits 1 node), train_DP=1
else
  TP=4; EP=8; CP=4; PP=2; ETP=1     # linear-train default (model_parallel=32)
fi
VLLM_TP=2
MIN_PAD=1
if [ ${CP} -gt 1 ]; then MIN_PAD=$((MIN_PAD * CP * 2)); fi
if [ ${TP} -gt 1 ]; then MIN_PAD=$((MIN_PAD * TP)); fi
MAKE_SEQ_DIVISIBLE_BY=${MIN_PAD}

# ================= Generation-scaling: derive all sizes from R =================
GPP=8                                            # generations per prompt (fixed)
SAMPLES_PER_REPLICA=2                             # invariant: samples/replica/step
BASE_CONCURRENCY=768                              # nemo-gym fan-out floor
REPLICAS_PER_NODE=$(( NUM_GPU / VLLM_TP ))        # = 4
MODEL_PARALLEL=$(( TP * CP * PP ))                # = 32
EXPERT_TMP=$(( ETP * EP * PP ))                   # = 16

NUM_VLLM_REPLICAS="${NUM_VLLM_REPLICAS:-}"
if [ -z "${NUM_VLLM_REPLICAS}" ]; then
  echo "ERROR: NUM_VLLM_REPLICAS is required (number of vLLM replicas). e.g. NUM_VLLM_REPLICAS=64" >&2
  exit 1
fi

# Smallest valid step for R.
gcd() { local a=$1 b=$2 t; while [ ${b} -ne 0 ]; do t=${b}; b=$(( a % b )); a=${t}; done; echo ${a}; }
lcm() { echo $(( $1 / $(gcd $1 $2) * $2 )); }
if [ "${SKIP_TRAINING}" = "1" ]; then
  # train fixed at 1 node (train_world=8, divisible by model_parallel=8); only gen
  # must fill whole nodes -> R need only be a multiple of REPLICAS_PER_NODE (=4).
  R_STEP=${REPLICAS_PER_NODE}
else
  # linear train: train_world=2R must be divisible by model-parallel & expert sizes.
  L=$(lcm ${MODEL_PARALLEL} ${EXPERT_TMP})          # train-world divisor
  R_STEP_TRAIN=$(( L / $(gcd 2 ${L}) ))             # since train_world = 2R
  R_STEP=$(lcm ${R_STEP_TRAIN} ${REPLICAS_PER_NODE})
fi
if [ $(( NUM_VLLM_REPLICAS % R_STEP )) -ne 0 ] || [ ${NUM_VLLM_REPLICAS} -lt ${R_STEP} ]; then
  echo "ERROR: NUM_VLLM_REPLICAS must be a positive multiple of ${R_STEP} (got ${NUM_VLLM_REPLICAS})." >&2
  exit 1
fi

GEN_NODES=$(( NUM_VLLM_REPLICAS / REPLICAS_PER_NODE ))
if [ "${SKIP_TRAINING}" = "1" ]; then
  TRAIN_NODES="${TRAIN_NODES:-1}"                 # no-op training: single node
else
  TRAIN_NODES="${TRAIN_NODES:-${GEN_NODES}}"      # linear 1:1 follow by default
fi
TOTAL_NODES=$(( TRAIN_NODES + GEN_NODES ))
PPS=$(( SAMPLES_PER_REPLICA * NUM_VLLM_REPLICAS / GPP ))
GBS=$(( PPS * GPP ))
CONCURRENCY=$(( GBS * 1 ))                         # GBS * max_trajectory_age_steps(=1)
if [ ${CONCURRENCY} -lt ${BASE_CONCURRENCY} ]; then CONCURRENCY=${BASE_CONCURRENCY}; fi

# Sanity: training divisibility (also re-checks any TRAIN_NODES override).
TRAIN_WORLD=$(( TRAIN_NODES * NUM_GPU ))
if [ $(( TRAIN_WORLD % MODEL_PARALLEL )) -ne 0 ] || [ $(( TRAIN_WORLD % EXPERT_TMP )) -ne 0 ]; then
  echo "ERROR: train world ${TRAIN_WORLD} (TRAIN_NODES=${TRAIN_NODES}) not divisible by model-parallel ${MODEL_PARALLEL} / expert ${EXPERT_TMP}." >&2
  exit 1
fi
TRAIN_DP=$(( TRAIN_WORLD / MODEL_PARALLEL ))
if [ $(( GBS % TRAIN_DP )) -ne 0 ]; then
  echo "ERROR: GBS ${GBS} not divisible by train DP ${TRAIN_DP}." >&2
  exit 1
fi
PER_GPU_BATCH=$(( GBS / TRAIN_DP ))
PER_REPLICA_SAMPLES=$(( GBS / NUM_VLLM_REPLICAS ))

# ===================== Sequence length & packing =====================
SEQLEN=131072
SEQUENCE_PACKING=True

# ================= Sync/Async mode & async GRPO settings =================
ASYNC_GRPO_ENABLED=True
MAX_TRAJECTORY_AGE_STEPS=1
FORCE_ON_POLICY_RATIO=True
INFLIGHT_WEIGHT_UPDATE=True
RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES=False
SEQ_LOGPROB_ERROR_THRESHOLD=null
if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  COLOCATED_ENABLED=False
  VLLM_GPU_UTIL=0.8
  OVERLAP_GRAD_REDUCE=False
  ADVANTAGE_CLIP_LOW=-100
  ADVANTAGE_CLIP_HIGH=100
  TIS_THRESHOLD=5
else
  COLOCATED_ENABLED=True
  VLLM_GPU_UTIL=0.5
  OVERLAP_GRAD_REDUCE=True
fi

# ========================= GRPO / sampling =========================
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
LR="1e-06"

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
SAVE_PERIOD=5
VAL_PERIOD=1000
KEEP_TOP_K=2

# ============================ SWE agent ============================
AGENT_MAX_TURNS=200
AGENT_TIMEOUT=1800

# ============================== Logging ==============================
WANDB_PROJ="swe-benchmark"
# Shared group for the whole generation-scaling sweep (compare runs by R).
WANDB_GROUP="${WANDB_GROUP:-swe-gen-scale-linear}"
# Log full trajectories to wandb so we can verify function_call items appear.
LOG_GYM_RESPONSES=true

# ========================= SLURM submission =========================
SBATCH_ACCOUNT="nemotron_sw_post"
SBATCH_PARTITION="batch"
SBATCH_TIME="${SBATCH_TIME:-4:0:0}"
# Optional smoke-test knob: cap training steps (appended as ++grpo.max_num_steps). Empty = use YAML default.
MAX_NUM_STEPS="${MAX_NUM_STEPS:-}"

# ========================= Experiment naming =========================
if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  SYNC_MODE="async-age${MAX_TRAJECTORY_AGE_STEPS}"
else
  SYNC_MODE="sync"
fi
EXP_SUFFIX="${EXP_SUFFIX:-swe-genscale-${SYNC_MODE}-genrep${NUM_VLLM_REPLICAS}-nodes${TOTAL_NODES}-pps${PPS}-gpp${GPP}-gbs${GBS}-lr${LR}}"
WANDB_NAME="${EXP_SUFFIX}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${EXP_SUFFIX}"
SNAPSHOT_DIR="${REPO_ROOT}"

mkdir -p "${CHECKPOINT_DIR}"

# ============= Unified SLURM/Ray log location =============
export BASE_LOG_DIR="${BASE_LOG_DIR:-${SNAPSHOT_DIR}/logs/swe_bench_scale}"
mkdir -p "${BASE_LOG_DIR}"

# ========================= Environment variables =========================
# Credentials are NOT sourced here. Export these yourself before submitting:
#   HF_HOME, HF_TOKEN, WANDB_API_KEY  (and GITHUB_TOKEN / GITLAB_TOKEN if needed)
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN}}"
export GITLAB_TOKEN="${GITLAB_TOKEN:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export UV_CACHE_DIR=/tmp/uv_cache
# Safe TE persistence (option B, seed-style — NO /root/.cache/uv override, so ray is untouched):
# the SETUP_COMMAND below rsyncs this Lustre seed (a harvested /tmp/uv_cache that already has the
# compiled transformer-engine wheel) into /tmp/uv_cache before the run, so the COMMAND's uv finds
# the prebuilt TE and skips the ~20-40min recompile. Empty seed => harmless (falls back to compile).
export LUSTRE_UV_CACHE_SEED="${LUSTRE_UV_CACHE_SEED:-}"
export UV_LOCK_TIMEOUT=3600
export RAY_DEDUP_LOGS=1
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export OMP_NUM_THREADS=16

# ========================= Node-local cache config =========================
PERSISTENT_CACHE="${PERSISTENT_CACHE:-${HOME}/.cache/qwen3_30b_thinking_swe_scale}"
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
echo "SWE generation-scaling | Experiment: ${EXP_SUFFIX}"
echo "Mode: ${SYNC_MODE}, Colocated: ${COLOCATED_ENABLED}"
echo "wandb: project=${WANDB_PROJ}, group=${WANDB_GROUP}, name=${WANDB_NAME}"
echo "------------------------------------------"
echo "Scaling input:  NUM_VLLM_REPLICAS = ${NUM_VLLM_REPLICAS}  (R-step=${R_STEP})"
echo "  replicas/node = ${REPLICAS_PER_NODE} (vllm_tp=${VLLM_TP})"
echo "  GEN_NODES     = ${GEN_NODES}"
echo "  TRAIN_NODES   = ${TRAIN_NODES}   (train_DP=${TRAIN_DP})"
echo "  TOTAL_NODES   = ${TOTAL_NODES}"
echo "  PPS           = ${PPS}"
echo "  GPP           = ${GPP}"
echo "  GBS           = ${GBS}"
echo "  CONCURRENCY   = ${CONCURRENCY}"
echo "  invariants    : samples/replica=${PER_REPLICA_SAMPLES}, batch/train-GPU=${PER_GPU_BATCH}"
echo "Parallelism: TP=${TP}, EP=${EP}, CP=${CP}, PP=${PP}, vLLM_TP=${VLLM_TP}, pad=${MAKE_SEQ_DIVISIBLE_BY}"
echo "Model: ${MODEL_PATH}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "=========================================="

cd "${SNAPSHOT_DIR}"

# ================ SETUP_COMMAND (bihu's: install apptainer + seed caches + uv sync) ================
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

# ================ Training command (bihu-style: uv run --frozen, no --extra mcore) ================
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
  TORCH_CUDA_ARCH_LIST='9.0 10.0' \
  NEMO_GYM_SKIP_VENV_IF_PRESENT=1 \
  uv run --frozen --extra mcore ./examples/nemo_gym/run_grpo_nemo_gym.py \
  --config=${CONFIG_FILE} \
  cluster.num_nodes=${TOTAL_NODES} \
  cluster.gpus_per_node=${NUM_GPU} \
  ++data.train.data_path=${TRAIN_DATA_PATH} \
  ++data.validation.data_path=${VAL_DATA_PATH} \
  grpo.num_prompts_per_step=${PPS} \
  grpo.num_generations_per_prompt=${GPP} \
  grpo.val_at_start=False \
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
  ++checkpointing.checkpoint_must_save_by=00:03:35:00 \
  logger.wandb_enabled=True \
  logger.wandb.name=${WANDB_NAME} \
  logger.wandb.project=${WANDB_PROJ} \
  ++logger.wandb.group=${WANDB_GROUP}"

if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  export COMMAND="${COMMAND} \
  policy.generation.colocated.resources.num_nodes=${GEN_NODES} \
  policy.generation.colocated.resources.gpus_per_node=${NUM_GPU} \
  grpo.advantage_clip_low=${ADVANTAGE_CLIP_LOW} \
  grpo.advantage_clip_high=${ADVANTAGE_CLIP_HIGH} \
  loss_fn.truncated_importance_sampling_ratio=${TIS_THRESHOLD} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.concurrency=${CONCURRENCY} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.concurrency=${CONCURRENCY}"
fi

# Optional: cap training steps (smoke test).
if [ -n "${MAX_NUM_STEPS}" ]; then
  export COMMAND="${COMMAND} grpo.max_num_steps=${MAX_NUM_STEPS}"
fi

# Generation-only benchmark: no-op training (no optimizer) + disable checkpoint saving.
if [ "${SKIP_TRAINING}" = "1" ]; then
  export COMMAND="${COMMAND} ++grpo.gen_benchmark_skip_training=true checkpointing.enabled=false"
fi

# ================ Submit job (skipped under DRY_RUN=1) ================
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ""
  echo "[DRY_RUN] Not submitting. Would run:"
  echo "[DRY_RUN]   sbatch --nodes=${TOTAL_NODES} --account=${SBATCH_ACCOUNT} --partition=${SBATCH_PARTITION} --time=${SBATCH_TIME} --gres=gpu:${NUM_GPU} ... ray.sub"
  cd - > /dev/null
  exit 0
fi

sbatch \
  --nodes="${TOTAL_NODES}" \
  --account="${SBATCH_ACCOUNT}" \
  --job-name="${WANDB_NAME}" \
  --partition="${SBATCH_PARTITION}" \
  --time="${SBATCH_TIME}" \
  --gres=gpu:${NUM_GPU} \
  --output="${BASE_LOG_DIR}/slurm-%j.out" \
  --exclusive \
  --dependency=singleton \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"data_loading","description":"Async GRPO SWE generation-scaling benchmark"}}' \
  ray.sub | tee /dev/stderr | grep -o '[0-9]\+' > latest_scale_gen_job_id.txt

JOB_ID="$(cat latest_scale_gen_job_id.txt)"
echo "=========================================="
echo "Job submitted: ${EXP_SUFFIX}"
echo "Job ID: ${JOB_ID}"
echo "wandb group: ${WANDB_GROUP}"
echo "Monitor with: squeue -j ${JOB_ID}"
echo "Ray/SLURM logs: ${BASE_LOG_DIR}/${JOB_ID}-logs/"
echo "Checkpoints: ${CHECKPOINT_DIR}/"
echo "=========================================="

cd - > /dev/null
