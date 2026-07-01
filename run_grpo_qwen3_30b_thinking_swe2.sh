#!/bin/bash
# ============================================================================
# NeMo-RL Async GRPO SWE2 RL Training: Qwen3-30B-A3B-Thinking (after SWE1)
#
# Model:      Qwen3-30B-A3B-Thinking SWE1 checkpoint (step_241 converted to HF)
# Train data: R2E-Gym (r2e-gym subset, 4518 samples)
# Eval data:  SWE-bench Verified
# Mode:       Async GRPO with non-colocated generation
# Env:        swe_agents (OpenHands agent + singularity sandbox)
#
# 这是 SWE1 训练后的继续训练 (SWE2)，使用 SWE1 step_241 的 HF checkpoint
#
# Usage:
#   bash run_grpo_qwen3_30b_thinking_swe2.sh
#
# Override:
#   NUM_NODES=16 NUM_GEN_NODES=8 bash run_grpo_qwen3_30b_thinking_swe2.sh
#   MODEL_PATH=/path/to/checkpoint bash run_grpo_qwen3_30b_thinking_swe2.sh
# ============================================================================

set -e

REPO_ROOT="/lustre/fsw/portfolios/coreai/users/erinh/RL"
CONFIG_FILE="${REPO_ROOT}/grpo_qwen3_30b_async_swe.yaml"
CHECKPOINT_ROOT="${REPO_ROOT}/results"

# ================ Scaling 实验核心参数 ================
PPS=8
GPP=8
GBS=64
LR="1e-06"
AGENT_MAX_TURNS=200
AGENT_TIMEOUT=1800

# ================ Sync/Async 模式选择 ================
ASYNC_GRPO_ENABLED=True
MAX_TRAJECTORY_AGE_STEPS=1

# ================ 根据 Sync/Async 自动配置 ================
NUM_ACTOR_NODES=${NUM_NODES:-16}
FORCE_ON_POLICY_RATIO=True
INFLIGHT_WEIGHT_UPDATE=True
RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES=False
SEQ_LOGPROB_ERROR_THRESHOLD=null

if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  COLOCATED_ENABLED=False
  VLLM_GPU_UTIL=0.8
  NUM_GENERATION_NODES=${NUM_GEN_NODES:-8}
  OVERLAP_GRAD_REDUCE=False
  ADVANTAGE_CLIP_LOW=-100
  ADVANTAGE_CLIP_HIGH=100
  TIS_THRESHOLD=5
else
  COLOCATED_ENABLED=True
  VLLM_GPU_UTIL=0.5
  OVERLAP_GRAD_REDUCE=True
fi

# ================ 固定参数 ================
SEQLEN=131072
NUM_GPU=8
export GPUS_PER_NODE=${NUM_GPU}
export CPUS_PER_WORKER=114

TP=4
EP=8
CP=4
PP=2

VLLM_TP=2

SEQUENCE_PACKING=True
TOKEN_LEVEL_LOSS=True

# 自动计算 make_sequence_length_divisible_by (满足 CP/TP/SP 的 assert)
# minimum_pad_factor = (cp_size * 2) * tp_size when both CP>1 and TP>1+SP
MIN_PAD=1
if [ ${CP} -gt 1 ]; then MIN_PAD=$((MIN_PAD * CP * 2)); fi
if [ ${TP} -gt 1 ]; then MIN_PAD=$((MIN_PAD * TP)); fi
MAKE_SEQ_DIVISIBLE_BY=${MIN_PAD}
SEQ_LEVEL_IS=False
NORMALIZE_REWARDS=True
OVERLONG_FILTERING=True

USE_ON_POLICY_KL_APPROXIMATION=True
IMPORTANCE_SAMPLING_CORRECTION=True
KL=0
CLIP_MIN=0.2
CLIP_MAX=0.28
TEMPERATURE=1.0

SAVE_PERIOD=5
VAL_PERIOD=1000
KEEP_TOP_K=2

MOE_FREEZE_ROUTER=True
MOE_PERMUTE_FUSION=True
MOE_ENABLE_DEEPEP=False
MOE_TOKEN_DISPATCHER_TYPE="alltoall"
MOE_AUX_LOSS_COEFF=0
MOE_ROUTER_LOAD_BALANCING_TYPE="none"
MOE_ROUTER_BIAS_UPDATE_RATE="1e-3"

# ================ 数据/模型路径 ================
TRAIN_DATA_PATH="/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/nano/dataset/rl/swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl"
VAL_DATA_PATH="${TRAIN_DATA_PATH}"
# Smoke test: start from the base model (no SWE1 checkpoint). Override with MODEL_PATH=... for a real SWE2 run.
MODEL_PATH=${MODEL_PATH:-"/lustre/fsw/portfolios/llmservice/users/igitman/hf_models/Qwen3-30B-A3B-Thinking-2507"}

# ================ 实验命名 ================
WANDB_PROJ="binhu-nemo-rl"
if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  SYNC_MODE="async-age${MAX_TRAJECTORY_AGE_STEPS}"
else
  SYNC_MODE="sync"
fi
EXP_SUFFIX="main-qwen3-30b-a3b-thinking-swe2-afterswe1-${SYNC_MODE}-pps${PPS}-gpp${GPP}-gbs${GBS}-lr${LR}"
WANDB_NAME="${EXP_SUFFIX}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${EXP_SUFFIX}"
SNAPSHOT_DIR="${REPO_ROOT}"

mkdir -p "${CHECKPOINT_DIR}"

# ================ 环境变量 ================
source "/lustre/fsw/portfolios/coreai/users/erinh/env.sh"
export UV_CACHE_DIR=/tmp/uv_cache
export UV_LOCK_TIMEOUT=3600
export RAY_DEDUP_LOGS=1
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export OMP_NUM_THREADS=16

# ================ Node-local cache 配置 ================
PERSISTENT_CACHE="/lustre/fsw/portfolios/coreai/users/erinh/.cache/qwen3_30b_thinking_swe"
export LUSTRE_VLLM_CACHE="${PERSISTENT_CACHE}/vllm_compile_cache"
export LUSTRE_INDUCTOR_CACHE="${PERSISTENT_CACHE}/inductor_cache"
export LUSTRE_TRITON_CACHE="${PERSISTENT_CACHE}/triton_cache"
export NRL_VLLM_LOCAL_CACHE_DIR="/tmp/nemo_rl_vllm_cache"
export NRL_VLLM_CACHE_SEED_DIR="/tmp/nemo_rl_vllm_cache_warm"
export INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
export TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
export CACHE_SYNC_FREQUENCY=120

mkdir -p "${LUSTRE_VLLM_CACHE}" "${LUSTRE_INDUCTOR_CACHE}" "${LUSTRE_TRITON_CACHE}"

SBATCH_ACCOUNT="coreai_dlalgo_llm"
SBATCH_PARTITION="batch"
SBATCH_TIME="4:0:0"

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

_found_warm=""
if [ -n "${LUSTRE_VLLM_CACHE}" ]; then
  _base="\$(basename "${LUSTRE_VLLM_CACHE}")"
  _parent="\$(dirname "${LUSTRE_VLLM_CACHE}")"
  _found_warm="\$(
    ls -1dt "\${_parent}/\${_base}_"* 2>/dev/null \
      | while IFS= read -r d; do
          [ -d "\$d" ] && [ "\$(ls -A "\$d" 2>/dev/null)" ] && echo "\$d" && break
        done
  )"
fi
if [ -n "\$_found_warm" ]; then
  rm -rf "${NRL_VLLM_CACHE_SEED_DIR}"
  _seed_cache "\$_found_warm" "${NRL_VLLM_CACHE_SEED_DIR}" "vLLM (from \$(basename "\$_found_warm"))"
else
  echo "[CACHE SEED] vLLM: no warm cache on Lustre yet"
  rm -rf "${NRL_VLLM_CACHE_SEED_DIR}"
fi
echo "[CACHE SEED] Done."

UV_HTTP_TIMEOUT=3600 \
  uv sync --frozen
SETUPEOF
export SETUP_COMMAND

# ================ 训练命令 ================
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
  uv run --frozen ./examples/nemo_gym/run_grpo_nemo_gym.py \
  --config=${CONFIG_FILE} \
  cluster.num_nodes=${NUM_ACTOR_NODES} \
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
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT}"
fi

# ================ 容器和挂载配置 ================
# main-ToT image: torch 2.11.0 / vLLM 0.20.0, with nemo_gym + async-GRPO actor venvs prefetched.
# (Apptainer is installed at runtime by SETUP_COMMAND; Gym is bind-mounted below.)
# Pinned copy of nliang's 2026-05-31 nightly (nemo-rl-nightly-latest.sqsh) for reproducibility.
export CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/coreai/users/erinh/enroot-images/nemo-rl-nightly-vllm0.20.0-torch2.11-20260531.sqsh}
GYM_CODE="${REPO_ROOT}/3rdparty/Gym-workspace/Gym"
export MOUNTS="/lustre:/lustre,$PWD:$PWD,${GYM_CODE}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"

# ================ 提交任务 ================
sbatch \
  --nodes="${NUM_ACTOR_NODES}" \
  --account="${SBATCH_ACCOUNT}" \
  --job-name="${WANDB_NAME}" \
  --partition="${SBATCH_PARTITION}" \
  --time="${SBATCH_TIME}" \
  --gres=gpu:${NUM_GPU} \
  --exclusive \
  --dependency=singleton \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Async GRPO RL training: training GPUs idle during rollout collection (~30min) and validation each step"}}' \
  ray.sub | tee /dev/stderr | grep -o '[0-9]\+' > latest_thinking_swe2_job_id.txt

JOB_ID="$(cat latest_thinking_swe2_job_id.txt)"
echo "=========================================="
echo "Job submitted: ${EXP_SUFFIX}"
echo "Job ID: ${JOB_ID}"
echo "Monitor with: squeue -j ${JOB_ID}"
echo "Logs: ${CHECKPOINT_DIR}/"
echo "=========================================="

# ================ 后台监控进程 ================
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
  
  LOG_DIR="${SNAPSHOT_DIR}/${JOB_ID}-logs"
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
) >> "${SNAPSHOT_DIR}/monitor_thinking_swe2_${JOB_ID}.log" 2>&1 &

MONITOR_PID=$!
echo "Started monitoring process (PID: $MONITOR_PID)"

cd - > /dev/null
