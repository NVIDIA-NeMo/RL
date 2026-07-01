#!/bin/bash
# ============================================================================
# SCALABLE async SWE GRPO launcher for Qwen3-30B-A3B-Thinking-2507 on
# oci-hsg-cs-001 / nemotron_sw_post (GB200, aarch64, 4 GPU/node).
#
# Purpose: confirm the SWE async-GRPO stack runs end-to-end on GB200 with a small
# model before scaling to 235B. Same single-knob design as the 235B launcher.
#
# Single knob:  NUM_VLLM_REPLICAS (R)  -> everything else is auto-derived.
# INVARIANT held across all R:   GPP*PPS/R == GBS/R == SAMPLES_PER_REPLICA (== 2 for 30B).
# GPP is held fixed (GRPO group size); PPS is derived from R to keep the invariant.
# Training geometry is FIXED & small (30B fits 1 node); R only scales generation:
#   GEN_NODES                = R * VLLM_TP / NUM_GPU
#   TRAIN_NODES (default)    = (TP*CP*PP) / NUM_GPU = 4/4 = 1   (override w/ TRAIN_NODES=)
#   TOTAL_NODES              = TRAIN_NODES + GEN_NODES   -> sbatch --nodes & cluster.num_nodes
#   PPS                      = SAMPLES_PER_REPLICA * R / GPP    (= R/4 with the defaults below)
#   GBS                      = PPS * GPP = SAMPLES_PER_REPLICA * R
#   CONCURRENCY              = max(768, GBS * max_trajectory_age_steps)
# Non-colocated carve-out (grpo.py:527): cluster.num_nodes - gen_nodes = train_nodes.
#
# Examples (defaults GPP=8, SAMPLES_PER_REPLICA=2 -> GPP*PPS/R = 2):
#   NUM_VLLM_REPLICAS=4 bash test_assets/qwen-30B/run_grpo_qwen3_30b_swe_scale_gen.sh
#       -> GEN_NODES=2 + TRAIN_NODES=1 = 3 nodes, PPS=1, GPP=8, GBS=8
#   NUM_VLLM_REPLICAS=8  -> GEN_NODES=4 + TRAIN_NODES=1 = 5 nodes, PPS=2, GBS=16
#   NUM_VLLM_REPLICAS=4 SEQLEN=16384 MAX_NUM_STEPS=2 DRY_RUN=1 bash .../run_grpo_qwen3_30b_swe_scale_gen.sh
#   SKIP_TRAINING=1 NUM_VLLM_REPLICAS=4 bash .../run_grpo_qwen3_30b_swe_scale_gen.sh   # gen-only
#
# ALIGN_BASELINE=1: reproduce baseline dc3m70us GPU/batch geometry. baseline ran on
#   x86 8-GPU nodes: 64 gen GPU (32 vLLM replicas) + 64 train GPU, GBS=64. On GB200
#   (4 GPU/node) the same GPU count is 16 gen + 16 train = 32 nodes. The switch just
#   pins NUM_VLLM_REPLICAS=32 (-> 16 gen nodes, GBS=64) and TRAIN_NODES=16 (train_DP=16).
#   Both remain overridable. Example:
#   ALIGN_BASELINE=1 bash .../run_grpo_qwen3_30b_swe_scale_gen.sh   # 32 nodes, GBS=64
#
# Optional env: ALIGN_BASELINE, SKIP_TRAINING, TRAIN_NODES, GPP, WANDB_GROUP, EXP_SUFFIX,
#               MODEL_PATH, CONTAINER, MAX_NUM_STEPS, SBATCH_TIME, PERSISTENT_CACHE,
#               BASE_LOG_DIR, TP, CP, EP, PP, VLLM_TP, SEQLEN (advanced overrides).
# ============================================================================

set -e

# ============================ Paths ============================
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/examples/swe_bench/grpo_qwen3_30b_async_swe_hsg.yaml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/results}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/lustre/fsw/portfolios/llmservice/users/sdevare/repos/ultra/datasets/swe/blends/balanced_language.jsonl}"
VAL_DATA_PATH="${VAL_DATA_PATH:-/lustre/fsw/portfolios/llmservice/users/sdevare/repos/ultra/datasets/swe/swe_public_datasets_val_swebench.jsonl}"
DEFAULT_MODEL_PATH="/lustre/fsw/portfolios/nemotron/users/ruit/evolution_rl/test_assets/qwen-30B/bihu/qwen3-30b-thinking-swe1-async-age1-pps64-gpp8-gbs512-lr1e-06/step_230_hf"
MODEL_PATH="${1:-${MODEL_PATH:-${DEFAULT_MODEL_PATH}}}"

# ================ Container and mount config ================
# GB200 (aarch64) baked image: apptainer + /opt/nemo_rl_venv with --extra mcore
# (sm_100), built by test_assets/SWE/build_swe_bench_combined.sh.
# Baked image: nightly-063026 + /opt/gym_venvs (gym server venvs prebuilt) so gym
# spinup hits skip_venv_if_present and does NOT concurrently build venvs (which
# deadlocks on the uv cache lock). Built by test_assets/SWE/build_swe_bench_combined.sh.
# Older non-baked images (e.g. nightly-062326) WILL hang at gym spinup — do not use.
export CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/nemotron/users/ruit/enroot-images/nemo-rl:nightly-063026-gymvenvs.squashfs}
GYM_CODE="${REPO_ROOT}/3rdparty/Gym-workspace/Gym"
export MOUNTS="/lustre:/lustre,$PWD:$PWD,${GYM_CODE}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"

# ======================= Cluster / resources =======================
NUM_GPU=4                                          # GB200: ray.sub asserts == gres gpu:4
export GPUS_PER_NODE=${NUM_GPU}
export CPUS_PER_WORKER=${CPUS_PER_WORKER:-96}      # GB200: 144 CPUs but ~720GB FreeMem; ray.sub worker srun is
                                                   # --exact + CR_CORE_MEMORY so step mem = RealMem*cpus/144.
                                                   # 140 -> ~916GB (> free) = intermittent step-creation OOM; 96 -> ~628GB safe.

# ============================ Parallelism (FIXED, fits 1 node) ============================
SKIP_TRAINING="${SKIP_TRAINING:-0}"
if [ "${SKIP_TRAINING}" = "1" ]; then
  TP="${TP:-4}"; EP="${EP:-4}"; CP="${CP:-1}"; PP="${PP:-1}"; ETP="${ETP:-1}"   # model_parallel=4 (1 node)
else
  TP="${TP:-4}"; EP="${EP:-4}"; CP="${CP:-1}"; PP="${PP:-1}"; ETP="${ETP:-1}"   # model_parallel=4 (1 node)
fi
VLLM_TP="${VLLM_TP:-2}"
MIN_PAD=1
if [ ${CP} -gt 1 ]; then MIN_PAD=$((MIN_PAD * CP * 2)); fi
if [ ${TP} -gt 1 ]; then MIN_PAD=$((MIN_PAD * TP)); fi
MAKE_SEQ_DIVISIBLE_BY=${MIN_PAD}

# ================= Generation-scaling: derive all sizes from R =================
# Invariant held across R:  GPP*PPS/R == GBS/R == SAMPLES_PER_REPLICA (constant).
# GPP (GRPO group size) is held fixed; PPS is derived from R so the invariant holds.
# 30B baseline anchors the invariant at 2: GPP=8, SAMPLES_PER_REPLICA=2
#   -> PPS = 2*R/8 = R/4,  GBS = 2*R   (R=4 => PPS1/GPP8/GBS8).
GPP="${GPP:-8}"                                    # generations per prompt (GRPO group size, fixed)
SAMPLES_PER_REPLICA="${SAMPLES_PER_REPLICA:-2}"    # invariant GBS/R = GPP*PPS/R (const = 2)
BASE_CONCURRENCY=768
MODEL_PARALLEL=$(( TP * CP * PP ))
EXPERT_TMP=$(( ETP * EP * PP ))

# ALIGN_BASELINE: reproduce baseline dc3m70us GPU/batch geometry (GBS=64, 32 vLLM
# replicas, 64 train GPU). On GB200 (4 GPU/node) that is 16 gen + 16 train = 32 nodes.
# Only seeds defaults -> NUM_VLLM_REPLICAS / TRAIN_NODES still win if set explicitly.
ALIGN_BASELINE="${ALIGN_BASELINE:-0}"
if [ "${ALIGN_BASELINE}" = "1" ]; then
  NUM_VLLM_REPLICAS="${NUM_VLLM_REPLICAS:-32}"   # 32 replicas -> GEN_NODES=16, GBS=2*32=64
  TRAIN_NODES="${TRAIN_NODES:-16}"               # 64 train GPU -> train_DP=64/MODEL_PARALLEL=16
fi

NUM_VLLM_REPLICAS="${NUM_VLLM_REPLICAS:-}"
if [ -z "${NUM_VLLM_REPLICAS}" ]; then
  echo "ERROR: NUM_VLLM_REPLICAS is required (number of vLLM replicas). e.g. NUM_VLLM_REPLICAS=4" >&2
  exit 1
fi

gcd() { local a=$1 b=$2 t; while [ ${b} -ne 0 ]; do t=${b}; b=$(( a % b )); a=${t}; done; echo ${a}; }
lcm() { echo $(( $1 / $(gcd $1 $2) * $2 )); }

R_STEP_GEN=$(( NUM_GPU / $(gcd ${VLLM_TP} ${NUM_GPU}) ))
R_STEP_PPS=$(( GPP / $(gcd ${SAMPLES_PER_REPLICA} ${GPP}) ))
R_STEP=$(lcm ${R_STEP_GEN} ${R_STEP_PPS})
if [ $(( NUM_VLLM_REPLICAS % R_STEP )) -ne 0 ] || [ ${NUM_VLLM_REPLICAS} -lt ${R_STEP} ]; then
  echo "ERROR: NUM_VLLM_REPLICAS must be a positive multiple of ${R_STEP} (got ${NUM_VLLM_REPLICAS})." >&2
  exit 1
fi

GEN_GPUS=$(( NUM_VLLM_REPLICAS * VLLM_TP ))
GEN_NODES=$(( GEN_GPUS / NUM_GPU ))
if [ "${SKIP_TRAINING}" = "1" ]; then
  TRAIN_NODES="${TRAIN_NODES:-1}"
else
  TRAIN_NODES="${TRAIN_NODES:-$(( MODEL_PARALLEL / NUM_GPU ))}"
  if [ ${TRAIN_NODES} -lt 1 ]; then TRAIN_NODES=1; fi
fi
TOTAL_NODES=$(( TRAIN_NODES + GEN_NODES ))
PPS=$(( SAMPLES_PER_REPLICA * NUM_VLLM_REPLICAS / GPP ))
if [ ${PPS} -lt 1 ]; then PPS=1; fi
GBS=$(( PPS * GPP ))
CONCURRENCY=$(( GBS * 1 ))
if [ ${CONCURRENCY} -lt ${BASE_CONCURRENCY} ]; then CONCURRENCY=${BASE_CONCURRENCY}; fi

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
SEQLEN="${SEQLEN:-131072}"
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
SAVE_PERIOD=5
VAL_PERIOD=1000
KEEP_TOP_K=2

# ============================ SWE agent ============================
AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-200}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-1800}"

# ============================== Logging ==============================
WANDB_PROJ="${WANDB_PROJ:-swe-benchmark}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-30b-gb200-swe-gen-scale}"
LOG_GYM_RESPONSES=true

# ========================= SLURM submission =========================
SBATCH_ACCOUNT="nemotron_sw_post"
SBATCH_PARTITION="batch"
SBATCH_TIME="${SBATCH_TIME:-4:0:0}"
MAX_NUM_STEPS="${MAX_NUM_STEPS:-3}"
# Free-form passthrough appended verbatim to COMMAND (extra ++policy.megatron_cfg.* knobs, etc.). Empty by default.
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ========================= Experiment naming =========================
if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  SYNC_MODE="async-age${MAX_TRAJECTORY_AGE_STEPS}"
else
  SYNC_MODE="sync"
fi
EXP_SUFFIX="${EXP_SUFFIX:-qwen3-30b-gb200-swe-genscale-${SYNC_MODE}-genrep${NUM_VLLM_REPLICAS}-nodes${TOTAL_NODES}-tp${TP}cp${CP}ep${EP}pp${PP}-pps${PPS}-gpp${GPP}-gbs${GBS}-seq${SEQLEN}-lr${LR}}"
WANDB_NAME="${EXP_SUFFIX}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${EXP_SUFFIX}"
SNAPSHOT_DIR="${REPO_ROOT}"

mkdir -p "${CHECKPOINT_DIR}"

# ============= Unified SLURM/Ray log location =============
export BASE_LOG_DIR="${BASE_LOG_DIR:-${SNAPSHOT_DIR}/logs/qwen3_30b_swe_scale}"
mkdir -p "${BASE_LOG_DIR}"

# ========================= Environment variables =========================
# Credentials are NOT sourced here. Before running, export your own (e.g. in your
# shell or a personal env script you source yourself):
#   HF_HOME=...            # HuggingFace cache dir
#   HF_TOKEN=...           # HuggingFace token (used for HUGGINGFACE_TOKEN below)
#   WANDB_API_KEY=...      # Weights & Biases API key
#   GITHUB_TOKEN=... GITLAB_TOKEN=...   # optional, for git-dep rate limits
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
PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fsw/portfolios/nemotron/users/ruit/.cache/qwen3_30b_thinking_swe_scale}"
export LUSTRE_VLLM_CACHE="${PERSISTENT_CACHE}/vllm_compile_cache"
export LUSTRE_INDUCTOR_CACHE="${PERSISTENT_CACHE}/inductor_cache"
export LUSTRE_TRITON_CACHE="${PERSISTENT_CACHE}/triton_cache"
# Seed the driver's node-local /tmp/uv_cache from a warm lustre copy so `uv run`
# hits the prebuilt transformer-engine wheel instead of recompiling (~20-30min).
# Populate this dir once from a completed head-node /tmp/uv_cache (see repo notes).
export LUSTRE_UV_CACHE_SEED="${LUSTRE_UV_CACHE_SEED:-${PERSISTENT_CACHE}/uv_cache_seed}"
export NRL_VLLM_LOCAL_CACHE_DIR="/tmp/nemo_rl_vllm_cache"
export NRL_VLLM_CACHE_SEED_DIR="/tmp/nemo_rl_vllm_cache_warm"
export INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
export TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
export CACHE_SYNC_FREQUENCY=120
mkdir -p "${LUSTRE_VLLM_CACHE}" "${LUSTRE_INDUCTOR_CACHE}" "${LUSTRE_TRITON_CACHE}"

# ============================== Summary ==============================
echo "=========================================="
echo "Qwen3-30B GB200 SWE generation-scaling | Experiment: ${EXP_SUFFIX}"
echo "Mode: ${SYNC_MODE}, Colocated: ${COLOCATED_ENABLED}, SkipTraining: ${SKIP_TRAINING}, AlignBaseline: ${ALIGN_BASELINE}"
echo "wandb: project=${WANDB_PROJ}, group=${WANDB_GROUP}, name=${WANDB_NAME}"
echo "------------------------------------------"
echo "Scaling input:  NUM_VLLM_REPLICAS = ${NUM_VLLM_REPLICAS}  (R-step=${R_STEP})"
echo "  vllm_tp       = ${VLLM_TP}  (nodes/replica = ${VLLM_TP}/${NUM_GPU})"
echo "  GEN_NODES     = ${GEN_NODES}"
echo "  TRAIN_NODES   = ${TRAIN_NODES}   (train_world=${TRAIN_WORLD}, model_parallel=${MODEL_PARALLEL}, train_DP=${TRAIN_DP})"
echo "  TOTAL_NODES   = ${TOTAL_NODES}"
echo "  PPS           = ${PPS}"
echo "  GPP           = ${GPP}"
echo "  GBS           = ${GBS}"
echo "  CONCURRENCY   = ${CONCURRENCY}"
echo "  invariants    : samples/replica=${PER_REPLICA_SAMPLES}, batch/train-GPU=${PER_GPU_BATCH}"
echo "Parallelism: TP=${TP}, EP=${EP}, CP=${CP}, PP=${PP}, ETP=${ETP}, vLLM_TP=${VLLM_TP}, pad=${MAKE_SEQ_DIVISIBLE_BY}"
echo "SeqLen: ${SEQLEN}"
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
  policy.megatron_cfg.expert_tensor_parallel_size=${ETP} \
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

if [ -n "${MAX_NUM_STEPS}" ]; then
  export COMMAND="${COMMAND} grpo.max_num_steps=${MAX_NUM_STEPS}"
fi

if [ "${SKIP_TRAINING}" = "1" ]; then
  export COMMAND="${COMMAND} ++grpo.gen_benchmark_skip_training=true checkpointing.enabled=false"
fi

# Free-form extra Hydra overrides (appended last so they can override anything above).
if [ -n "${EXTRA_ARGS}" ]; then
  export COMMAND="${COMMAND} ${EXTRA_ARGS}"
fi

# ================ Submit job (skipped under DRY_RUN=1) ================
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ""
  echo "[DRY_RUN] Not submitting. Would run:"
  echo "[DRY_RUN]   sbatch --nodes=${TOTAL_NODES} --account=${SBATCH_ACCOUNT} --partition=${SBATCH_PARTITION} --time=${SBATCH_TIME} --gres=gpu:${NUM_GPU} ... ray.sub"
  echo ""
  echo "[DRY_RUN] COMMAND:"
  echo "${COMMAND}"
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
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"data_loading","description":"Async GRPO Qwen3-30B GB200 SWE generation-scaling"}}' \
  ray.sub | tee /dev/stderr | grep -o '[0-9]\+' > latest_30b_scale_gen_job_id.txt

JOB_ID="$(cat latest_30b_scale_gen_job_id.txt)"
echo "=========================================="
echo "Job submitted: ${EXP_SUFFIX}"
echo "Job ID: ${JOB_ID}"
echo "wandb group: ${WANDB_GROUP}"
echo "Monitor with: squeue -j ${JOB_ID}"
echo "Ray/SLURM logs: ${BASE_LOG_DIR}/${JOB_ID}-logs/"
echo "Checkpoints: ${CHECKPOINT_DIR}/"
echo "=========================================="

cd - > /dev/null
