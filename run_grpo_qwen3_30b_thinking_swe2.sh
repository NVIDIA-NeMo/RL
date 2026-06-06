#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# NeMo-RL Async GRPO SWE2 RL Training: Qwen3-30B-A3B-Thinking (after SWE1)
#
# Model:      Qwen3-30B-A3B-Thinking SWE1 checkpoint (step_230 converted to HF)
# Train data: R2E-Gym (r2e-gym subset, 4518 samples)
# Eval data:  SWE-bench Verified
# Mode:       Async GRPO with non-colocated generation
# Env:        swe_agents (OpenHands agent + singularity sandbox)
#
# Continued training after SWE1 (SWE2), starting from SWE1 step_230 HF checkpoint
#
# Usage:
#   bash run_grpo_qwen3_30b_thinking_swe2.sh [MODEL_PATH]
#
# Override:
#   bash run_grpo_qwen3_30b_thinking_swe2.sh /path/to/checkpoint   # positional arg
#   MODEL_PATH=/path/to/checkpoint bash run_grpo_qwen3_30b_thinking_swe2.sh   # env var
#   NUM_NODES=16 NUM_GEN_NODES=8 bash run_grpo_qwen3_30b_thinking_swe2.sh
# ============================================================================

set -e

# ============================ Paths ============================
REPO_ROOT="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/joyang/RL"
CONFIG_FILE="${REPO_ROOT}/test_assets/SWE/grpo_qwen3_30b_async_swe.yaml"
CHECKPOINT_ROOT="${REPO_ROOT}/results"
TRAIN_DATA_PATH="/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/nano/dataset/rl/swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl"
VAL_DATA_PATH="${TRAIN_DATA_PATH}"
# Model checkpoint to train from. Override priority: positional arg $1 > MODEL_PATH env > default.
# SWE2 continues RL from the SWE1 checkpoint. Use bihu's SWE1 step_230 HF export — the same
# checkpoint ruit's repro-bihu recipe trains from (keeps this run consistent with ruit).
DEFAULT_MODEL_PATH="/lustre/fsw/portfolios/coreai/users/bihu/repos/nemo-rl-async-swe/results/qwen3-30b-thinking-swe1-async-age1-pps64-gpp8-gbs512-lr1e-06/step_230_hf"
MODEL_PATH="${1:-${MODEL_PATH:-${DEFAULT_MODEL_PATH}}}"

# ================ Container and mount config ================
# nliang's image is the proven SWE repro container: bakes vLLM 0.17.1 (working hermes
# tool parser) AND apptainer (so SETUP_COMMAND can stay disabled). The old
# nemo-rl-nightly-053026-apptainer-mcore image baked vLLM 0.20.0, whose hermes parser
# emits zero function_calls -> 0.0 reward. This image's prefetched venvs are from a
# different nemo-rl commit than HEAD, so the container-fingerprint check prints a
# (non-fatal) mismatch warning; NRL_IGNORE_VERSION_MISMATCH=1 in COMMAND only silences
# that expected warning. vLLM 0.17.1 was verified by hand to match uv.lock.
export CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/coreai/users/nliang/enroot-images/docker_images:nliang-qwen3-swe-training-e19dee3ba-x86_64-051626.squashfs}
GYM_CODE="${REPO_ROOT}/3rdparty/Gym-workspace/Gym"
export MOUNTS="/lustre:/lustre,$PWD:$PWD,${GYM_CODE}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"

# ======================= Cluster / resources =======================
NUM_ACTOR_NODES=${NUM_NODES:-16}
NUM_GENERATION_NODES=${NUM_GEN_NODES:-8}   # only used in async (non-colocated) mode
NUM_GPU=8
export GPUS_PER_NODE=${NUM_GPU}
export CPUS_PER_WORKER=114

# ============================ Parallelism ============================
TP=4
EP=8
CP=4
PP=2
VLLM_TP=2
# Auto-compute make_sequence_length_divisible_by (to satisfy the CP/TP/SP assert)
# minimum_pad_factor = (cp_size * 2) * tp_size when both CP>1 and TP>1+SP
MIN_PAD=1
if [ ${CP} -gt 1 ]; then MIN_PAD=$((MIN_PAD * CP * 2)); fi
if [ ${TP} -gt 1 ]; then MIN_PAD=$((MIN_PAD * TP)); fi
MAKE_SEQ_DIVISIBLE_BY=${MIN_PAD}

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
# Settings that differ between async (non-colocated) and sync (colocated) modes.
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
PPS=8
GPP=8
GBS=64
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
# (VLLM_TP set under Parallelism; VLLM_GPU_UTIL set under Sync/Async mode)

# =================== Checkpointing & validation ===================
SAVE_PERIOD=5
VAL_PERIOD=1000
KEEP_TOP_K=2

# ============================ SWE agent ============================
AGENT_MAX_TURNS=200
AGENT_TIMEOUT=1800
CONCURRENCY=768

# ============================== Logging ==============================
WANDB_PROJ="swe-benchmark-harness"

# ========================= SLURM submission =========================
SBATCH_ACCOUNT="coreai_dlalgo_nemorl"
SBATCH_PARTITION="batch"
SBATCH_TIME="4:0:0"

# ========================= Experiment naming =========================
if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  SYNC_MODE="async-age${MAX_TRAJECTORY_AGE_STEPS}"
else
  SYNC_MODE="sync"
fi
# Derive a short tag from MODEL_PATH for naming. Works for HF hub IDs
# (e.g. Qwen/Qwen3-30B-A3B-Thinking-2507) and local checkpoint paths.
MODEL_TAG="$(basename "${MODEL_PATH%/}")"
# If the basename is just a checkpoint/step folder, use the parent dir name instead.
case "${MODEL_TAG}" in
  step_*|iter_*|hf|*_hf|checkpoint|checkpoints|global_step*|policy)
    MODEL_TAG="$(basename "$(dirname "${MODEL_PATH%/}")")" ;;
esac
# Sanitize: lowercase, keep alnum, collapse the rest into single dashes.
MODEL_TAG="$(printf '%s' "${MODEL_TAG}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' | sed -E 's/-+/-/g; s/^-+//; s/-+$//')"
MODEL_TAG="${MODEL_TAG_OVERRIDE:-${MODEL_TAG}}"

EXP_SUFFIX="${EXP_SUFFIX:-${MODEL_TAG}-SWEbench-seqlen${SEQLEN}-turn${AGENT_MAX_TURNS}-concurrency${CONCURRENCY}}"
WANDB_NAME="${EXP_SUFFIX}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${EXP_SUFFIX}"
SNAPSHOT_DIR="${REPO_ROOT}"

mkdir -p "${CHECKPOINT_DIR}"

# ============= Unified SLURM/Ray log location (not repo root) =============
# ray.sub writes per-job logs to ${BASE_LOG_DIR}/${SLURM_JOB_ID}-logs (see ray.sub).
# The sbatch stdout/stderr (slurm-%j.out) and the monitor log below go here too.
export BASE_LOG_DIR="${BASE_LOG_DIR:-${SNAPSHOT_DIR}/logs/slurm}"
mkdir -p "${BASE_LOG_DIR}"
# ray.sub runs with `set -x` and traces the COMMAND (which embeds WANDB/HF/GitHub/GitLab
# tokens) into slurm-%j.out. Lock the log dir so those plaintext secrets stay readable
# only by the owner (does not affect checkpoint/results visibility).
chmod 700 "${BASE_LOG_DIR}" 2>/dev/null || true

# ========================= Environment variables =========================
# Use joyang's OWN private credentials/caches. Tokens (HF_TOKEN, WANDB_API_KEY,
# GITHUB_TOKEN) are inherited from the shell profile; do NOT source ruit's env.
export HF_HOME="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/joyang/hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN}}"
export GITLAB_TOKEN="${GITLAB_TOKEN:-}"
# Fail fast if private credentials aren't present in the environment.
: "${HF_TOKEN:?HF_TOKEN not set in env}"
: "${WANDB_API_KEY:?WANDB_API_KEY not set in env}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"
# Persistent uv cache on Lustre so wheels/venvs aren't re-downloaded+rebuilt every job
# (was /tmp/uv_cache, which is node-local and wiped between jobs -> cold start each time).
# /lustre is bind-mounted into the container, and ray.sub forwards UV_CACHE_DIR via COMMAND.
export UV_CACHE_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/joyang/uv_cache
# Cache (Lustre) and venvs (/opt, in-container) live on different filesystems; copy instead
# of hardlink to avoid uv's repeated "failed to hardlink, falling back to copy" warnings.
export UV_LINK_MODE=copy
mkdir -p "${UV_CACHE_DIR}"
export UV_LOCK_TIMEOUT=3600
export RAY_DEDUP_LOGS=1
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export OMP_NUM_THREADS=16

# ========================= Node-local cache config =========================
PERSISTENT_CACHE="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/joyang/.cache/qwen3_30b_thinking_swe"
export LUSTRE_VLLM_CACHE="${PERSISTENT_CACHE}/vllm_compile_cache"
export LUSTRE_INDUCTOR_CACHE="${PERSISTENT_CACHE}/inductor_cache"
export LUSTRE_TRITON_CACHE="${PERSISTENT_CACHE}/triton_cache"
export INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
export TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
mkdir -p "${LUSTRE_VLLM_CACHE}" "${LUSTRE_INDUCTOR_CACHE}" "${LUSTRE_TRITON_CACHE}"

# ============================== Summary ==============================
echo "=========================================="
echo "Experiment: ${EXP_SUFFIX}"
echo "Mode: ${SYNC_MODE}, Colocated: ${COLOCATED_ENABLED}"
echo "Nodes: ${NUM_ACTOR_NODES}, GPUs/node: ${NUM_GPU}"
echo "Parallelism: TP=${TP}, EP=${EP}, CP=${CP}, PP=${PP}, vLLM_TP=${VLLM_TP}"
echo "Training: PPS=${PPS}, GPP=${GPP}, GBS=${GBS}, LR=${LR}"
echo "SeqLen: ${SEQLEN}"
echo "Agent: max_turns=${AGENT_MAX_TURNS}, timeout=${AGENT_TIMEOUT}s"
echo "Model: ${MODEL_PATH}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "=========================================="

cd "${SNAPSHOT_DIR}"

# ================ SETUP_COMMAND ================
# read -r -d '' SETUP_COMMAND <<SETUPEOF || true
# echo "[SETUP] Installing apptainer for SWE sandbox..."
# apt-get update && apt-get install -y git build-essential gcc wget 2>/dev/null || true
# RET=1
# RETRIES=3
# for attempt in \$(seq 1 \$RETRIES); do
#   if command -v apptainer >/dev/null 2>&1 || command -v singularity >/dev/null 2>&1; then
#     echo "[SETUP] singularity/apptainer already available"
#     RET=0
#     break
#   fi
#   cd /tmp && \
#   wget --no-check-certificate -q https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_amd64.deb && \
#   apt install -y ./apptainer_1.3.1_amd64.deb && \
#   ln -sf /usr/bin/apptainer /usr/bin/singularity
#   if command -v apptainer >/dev/null 2>&1; then
#     echo "[SETUP] apptainer installed successfully"
#     RET=0
#     break
#   fi
#   echo "[SETUP] apptainer install attempt \$attempt failed, retrying..."
#   sleep 10
# done
# if [ \$RET -ne 0 ]; then
#   echo "[SETUP] WARNING: apptainer installation failed after \$RETRIES attempts"
# fi
#
# echo "[CACHE SEED] Clearing stale /tmp caches and seeding from Lustre..."
# rm -rf "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
# mkdir -p "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
#
# find "${LUSTRE_INDUCTOR_CACHE}" -maxdepth 1 -name '.tmp_*' -mmin +30 -exec rm -rf {} + 2>/dev/null || true
# find "${LUSTRE_TRITON_CACHE}" -maxdepth 1 -name '.tmp_*' -mmin +30 -exec rm -rf {} + 2>/dev/null || true
#
# _seed_cache() {
#   local lustre="\$1" local_dir="\$2" name="\$3"
#   if [ -d "\$lustre" ] && [ "\$(ls -A "\$lustre" 2>/dev/null)" ]; then
#     rsync -a --exclude '.tmp_*' "\$lustre/" "\$local_dir/" 2>/dev/null \
#       && echo "[CACHE SEED] \$name: seeded from Lustre" \
#       || echo "[CACHE SEED] \$name: seed failed (non-fatal)"
#   else
#     echo "[CACHE SEED] \$name: no warm cache on Lustre yet"
#   fi
# }
#
# _seed_cache "${LUSTRE_INDUCTOR_CACHE}" "${INDUCTOR_CACHE_DIR}" "Inductor"
# _seed_cache "${LUSTRE_TRITON_CACHE}" "${TRITON_CACHE_DIR}" "Triton"
# echo "[CACHE SEED] Done."
#
# UV_HTTP_TIMEOUT=3600 \
#   uv sync --frozen
# SETUPEOF
# export SETUP_COMMAND

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
  UV_LINK_MODE=${UV_LINK_MODE} \
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
  uv run --frozen --extra mcore ./examples/nemo_gym/run_grpo_nemo_gym.py \
  --config=${CONFIG_FILE} \
  cluster.num_nodes=${NUM_ACTOR_NODES} \
  cluster.gpus_per_node=${NUM_GPU} \
  ++data.train.data_path=${TRAIN_DATA_PATH} \
  ++data.validation.data_path=${VAL_DATA_PATH} \
  grpo.num_prompts_per_step=${PPS} \
  grpo.num_generations_per_prompt=${GPP} \
  grpo.val_at_start=False \
  grpo.max_num_steps=3 \
  grpo.normalize_rewards=${NORMALIZE_REWARDS} \
  grpo.overlong_filtering=${OVERLONG_FILTERING} \
  grpo.val_period=${VAL_PERIOD} \
  grpo.seq_logprob_error_threshold=${SEQ_LOGPROB_ERROR_THRESHOLD} \
  grpo.async_grpo.enabled=${ASYNC_GRPO_ENABLED} \
  grpo.async_grpo.in_flight_weight_updates=${INFLIGHT_WEIGHT_UPDATE} \
  grpo.async_grpo.recompute_kv_cache_after_weight_updates=${RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES} \
  grpo.async_grpo.max_trajectory_age_steps=${MAX_TRAJECTORY_AGE_STEPS} \
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
  checkpointing.enabled=False \
  checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
  checkpointing.save_period=${SAVE_PERIOD} \
  checkpointing.keep_top_k=${KEEP_TOP_K} \
  ++checkpointing.metric_name=train:total_reward/mean \
  ++checkpointing.checkpoint_must_save_by=00:03:35:00 \
  logger.wandb_enabled=True \
  logger.wandb.name=${WANDB_NAME} \
  logger.wandb.project=${WANDB_PROJ}"

if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  export COMMAND="${COMMAND} \
  policy.generation.colocated.resources.num_nodes=${NUM_GENERATION_NODES} \
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
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Async GRPO RL training: training GPUs idle during rollout collection (~30min) and validation each step"}}' \
  ray.sub | tee /dev/stderr | grep -o '[0-9]\+' > latest_thinking_swe2_job_id.txt

JOB_ID="$(cat latest_thinking_swe2_job_id.txt)"
echo "=========================================="
echo "Job submitted: ${EXP_SUFFIX}"
echo "Job ID: ${JOB_ID}"
echo "Monitor with: squeue -j ${JOB_ID}"
echo "Ray/SLURM logs: ${BASE_LOG_DIR}/${JOB_ID}-logs/ (and slurm-${JOB_ID}.out)"
echo "Checkpoints: ${CHECKPOINT_DIR}/"
echo "=========================================="

# ================ Background monitoring process ================
(
  echo "[$(date)] Waiting for job $JOB_ID to start running..."
  MAX_WAIT_ITERATIONS=100000000
  for i in $(seq 1 $MAX_WAIT_ITERATIONS); do
    if squeue -j $JOB_ID -h -o "%T" 2>/dev/null | grep -q "RUNNING"; then
      echo "[$(date)] Job $JOB_ID is now RUNNING."
      break
    fi
    if ! squeue -j $JOB_ID &>/dev/null; then
      echo "[$(date)] Job $JOB_ID is no longer in queue."
      exit 0
    fi
    sleep 60
  done

  LOG_DIR="${BASE_LOG_DIR}/${JOB_ID}-logs"
  RAY_DRIVER_LOG="${LOG_DIR}/ray-driver.log"

  for minute in $(seq 1 30); do
    if ! squeue -j $JOB_ID &>/dev/null; then
      echo "[$(date)] Job $JOB_ID is no longer running."
      exit 0
    fi

    if [ -f "$RAY_DRIVER_LOG" ]; then
      if grep -q "AssertionError: Attempting to report device id" "$RAY_DRIVER_LOG" 2>/dev/null; then
        echo "[$(date)] Found vLLM initialization error. Killing job $JOB_ID."
        scancel $JOB_ID
        exit 0
      fi
      if grep -q "Failed to build.*mamba-ssm" "$RAY_DRIVER_LOG" 2>/dev/null; then
        echo "[$(date)] Found mamba-ssm build failure. Killing job $JOB_ID."
        scancel $JOB_ID
        exit 0
      fi
      if grep -q "RuntimeError: Engine core initialization failed" "$RAY_DRIVER_LOG" 2>/dev/null; then
        echo "[$(date)] Found engine core initialization failure. Killing job $JOB_ID."
        scancel $JOB_ID
        exit 0
      fi
      if grep -q "CUDA error: an illegal memory access was encountered" "$RAY_DRIVER_LOG" 2>/dev/null || \
         grep -q "RayTaskError(AcceleratorError)" "$RAY_DRIVER_LOG" 2>/dev/null; then
        echo "[$(date)] Found CUDA illegal memory access / AcceleratorError. Killing job $JOB_ID."
        scancel $JOB_ID
        exit 0
      fi
      if grep -q "ERROR:nemo_rl.utils.venvs:Failed to create venv" "$RAY_DRIVER_LOG" 2>/dev/null; then
        echo "[$(date)] Venv creation failure. Killing job $JOB_ID."
        scancel $JOB_ID
        exit 0
      fi
    fi

    if [ $minute -lt 30 ]; then
      sleep 60
    fi
  done

  echo "[$(date)] Monitoring complete. Job $JOB_ID appears stable."
) >> "${BASE_LOG_DIR}/monitor_thinking_swe2_${JOB_ID}.log" 2>&1 &

MONITOR_PID=$!
echo "Started monitoring process (PID: $MONITOR_PID)"

cd - > /dev/null
