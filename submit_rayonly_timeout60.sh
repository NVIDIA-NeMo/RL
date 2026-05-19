#!/usr/bin/env bash
# Re-submit qwen3-235b-swe-perf-rayonly variant (Ray-opt PR #1944 only — NO HybridEP,
# NO FP8, NO skip_logprob, BF16 KV, alltoall dispatcher) for n=19 post-warmup completion.
# Original 11772327 (Iteration 0 baseline) timed out at 4h with n=18 post-warmup steps;
# this rerun gives matching n=19 coverage for the 3-way fair comparison vs:
#   - Track A: 11835558 (FP8 KV + FA3 + HybridEP) — 23 clean steps captured
#   - HybridEP-only: 11847579 (BF16 KV + HybridEP) — IN FLIGHT n=19 rerun
#
# Tokens must be exported in the calling shell (WANDB_API_KEY, HF_TOKEN). No fallback values.
set -euo pipefail

: "${WANDB_API_KEY:?WANDB_API_KEY must be exported in the environment}"
: "${HF_TOKEN:?HF_TOKEN must be exported in the environment}"

VARIANT=ray_only
DRY_RUN="${DRY_RUN:-false}"

REPO_DIR="/lustre/fsw/portfolios/coreai/users/sna/repos/nemo-rl-qwen-swe"
CACHE_BASE="/lustre/fsw/portfolios/coreai/users/sna/.cache/qwen3_235b_swe"
HF_HOME_DIR="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home"
DATA_PATH="/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/nano/dataset/rl/swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl"
CONFIG_PATH="${REPO_DIR}/grpo_qwen3_235b_swe.yaml"
CONTAINER="/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/nvcr.io/nvidian/nemo-rl:7684dc2-45115915.squashfs"

# ray_only variant: empty overrides (matches submit_perf_variant.sh case ray_only)
VARIANT_TAG="rayonly-timeout60"
EXTRA_OVERRIDES=""
EXTRA_ENVS=""

EXP_NAME="qwen3-235b-swe-perf-${VARIANT_TAG}"
CKPT_DIR="${REPO_DIR}/results/${EXP_NAME}"
LOG_DIR="logs/${EXP_NAME}"

cd "${REPO_DIR}"

git checkout sj/super-v3-perf-patch
git pull --ff-only

export OMP_NUM_THREADS=16
export CONTAINER
export MOUNTS="/lustre:/lustre,${REPO_DIR}:${REPO_DIR},${REPO_DIR}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"

unset SANDBOX_CONTAINER SANDBOX_COMMAND SANDBOX_ENV_VARS 2>/dev/null || true

read -r -d '' SETUP_COMMAND <<'SETUP_EOF' || true
echo "[SETUP] Installing apptainer for SWE sandbox..."
apt-get update && apt-get install -y git build-essential gcc wget 2>/dev/null || true
RET=1
RETRIES=3
for attempt in $(seq 1 $RETRIES); do
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
  echo "[SETUP] apptainer install attempt $attempt failed, retrying..."
  sleep 10
done
if [ $RET -ne 0 ]; then
  echo "[SETUP] WARNING: apptainer installation failed after $RETRIES attempts"
fi

echo "[SETUP] Wiring CUDA_HOME and /bin/nvcc for deep_ep JIT..."
if [ -x /usr/local/cuda/bin/nvcc ]; then
  ln -sf /usr/local/cuda/bin/nvcc /bin/nvcc
  echo "[SETUP] /bin/nvcc -> $(readlink -f /bin/nvcc)"
elif [ -x /usr/local/cuda-12.9/bin/nvcc ]; then
  ln -sf /usr/local/cuda-12.9/bin/nvcc /bin/nvcc
  echo "[SETUP] /bin/nvcc -> $(readlink -f /bin/nvcc)"
else
  NVCC_FOUND=$(find /usr/local /opt -maxdepth 5 -name nvcc -type f 2>/dev/null | head -1)
  if [ -n "$NVCC_FOUND" ]; then
    ln -sf "$NVCC_FOUND" /bin/nvcc
    echo "[SETUP] /bin/nvcc -> $NVCC_FOUND"
  else
    echo "[SETUP] WARNING: nvcc not found anywhere on container"
  fi
fi
nvcc --version 2>&1 | head -5 || true

echo "[GYM CLEANUP] Pruning stale swebench_results_* dirs (>1d old) to prevent project inode quota exhaustion..."
GYM_PATH="/lustre/fsw/portfolios/coreai/users/sna/repos/nemo-rl-qwen-swe/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents"
GYM_LOCK="${GYM_PATH}/.cleanup.lock"
if [ -d "$GYM_PATH" ]; then
  # flock ensures only one node prunes; non-blocking, others skip.
  (
    flock -n 9 || { echo "[GYM CLEANUP] another node holds lock, skipping"; exit 0; }
    BEFORE=$(find "$GYM_PATH" -maxdepth 1 -type d -name 'swebench_results_*' 2>/dev/null | wc -l)
    find "$GYM_PATH" -maxdepth 1 -type d -name 'swebench_results_*' -mtime +1 -exec rm -rf {} + 2>/dev/null || true
    AFTER=$(find "$GYM_PATH" -maxdepth 1 -type d -name 'swebench_results_*' 2>/dev/null | wc -l)
    echo "[GYM CLEANUP] pruned $((BEFORE-AFTER)) dirs (before=$BEFORE after=$AFTER)"
  ) 9>"$GYM_LOCK"
fi

echo "[CACHE SEED] Clearing stale /tmp caches and seeding from Lustre..."
rm -rf /tmp/nemo_rl_vllm_cache /tmp/nemo_rl_vllm_cache_*
rm -rf "/tmp/nemo_rl_inductor_cache" "/tmp/nemo_rl_triton_cache"
mkdir -p "/tmp/nemo_rl_inductor_cache" "/tmp/nemo_rl_triton_cache"

VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=https://github.com/vllm-project/vllm/releases/download/v0.13.0/vllm-0.13.0-cp38-abi3-manylinux_2_31_x86_64.whl \
UV_HTTP_TIMEOUT=3600 \
uv sync --frozen
SETUP_EOF
export SETUP_COMMAND

export COMMAND="CUDA_HOME=/usr/local/cuda \
  CUDA_PATH=/usr/local/cuda \
  NRL_VLLM_USE_V1=1 \
  NRL_WG_USE_RAY_REF=1 \
  VLLM_USE_PRECOMPILED=1 \
  VLLM_PRECOMPILED_WHEEL_LOCATION=https://github.com/vllm-project/vllm/releases/download/v0.13.0/vllm-0.13.0-cp38-abi3-manylinux_2_31_x86_64.whl \
  WANDB_API_KEY=${WANDB_API_KEY} \
  HUGGINGFACE_TOKEN=${HF_TOKEN} \
  HF_HOME=${HF_HOME_DIR} \
  HF_DATASETS_CACHE=${HF_HOME_DIR}/cache \
  UV_CACHE_DIR=/lustre/fsw/portfolios/coreai/users/sna/uv_cache \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  VLLM_CACHE_ROOT=${CACHE_BASE}/vllm_compile_cache \
  DG_JIT_CACHE_DIR=${CACHE_BASE}/vllm_compile_cache/deep_gemm \
  VLLM_DEEP_GEMM_WARMUP=skip \
  NRL_FORCE_REBUILD_VENVS=true \
  NRL_IGNORE_VERSION_MISMATCH=1 \
  RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
  UV_HTTP_TIMEOUT=3600 \
  TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST_OVERRIDE:-9.0 10.0}' \
  NEMO_GYM_SKIP_VENV_IF_PRESENT=1 ${EXTRA_ENVS} \
  uv run --frozen ./examples/nemo_gym/run_grpo_nemo_gym.py \
  --config=${CONFIG_PATH} \
  cluster.num_nodes=16 \
  cluster.gpus_per_node=8 \
  ++data.train.data_path=${DATA_PATH} \
  ++data.validation.data_path=${DATA_PATH} \
  grpo.num_prompts_per_step=32 \
  grpo.num_generations_per_prompt=8 \
  grpo.val_at_start=False \
  grpo.normalize_rewards=True \
  grpo.overlong_filtering=True \
  grpo.val_period=1000 \
  grpo.seq_logprob_error_threshold=2 \
  grpo.async_grpo.enabled=True \
  grpo.async_grpo.in_flight_weight_updates=False \
  grpo.async_grpo.recompute_kv_cache_after_weight_updates=False \
  grpo.async_grpo.max_trajectory_age_steps=1 \
  policy.generation.colocated.enabled=False \
  policy.model_name=Qwen/Qwen3-235B-A22B-Thinking-2507 \
  policy.max_total_sequence_length=16384 \
  policy.dynamic_batching.enabled=False \
  policy.train_global_batch_size=256 \
  policy.make_sequence_length_divisible_by=8 \
  policy.sequence_packing.enabled=True \
  policy.megatron_cfg.tensor_model_parallel_size=4 \
  policy.megatron_cfg.expert_model_parallel_size=8 \
  policy.megatron_cfg.context_parallel_size=1 \
  policy.megatron_cfg.pipeline_model_parallel_size=8 \
  policy.megatron_cfg.num_layers_in_first_pipeline_stage=11 \
  policy.megatron_cfg.num_layers_in_last_pipeline_stage=11 \
  policy.megatron_cfg.sequence_parallel=True \
  policy.megatron_cfg.bias_activation_fusion=False \
  policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=False \
  policy.megatron_cfg.moe_permute_fusion=True \
  policy.megatron_cfg.moe_enable_deepep=False \
  policy.megatron_cfg.moe_token_dispatcher_type=alltoall \
  policy.megatron_cfg.moe_aux_loss_coeff=0 \
  policy.megatron_cfg.moe_router_load_balancing_type=none \
  policy.megatron_cfg.moe_router_bias_update_rate=1e-3 \
  policy.megatron_cfg.freeze_moe_router=True \
  policy.megatron_cfg.optimizer.lr=1e-06 \
  policy.megatron_cfg.optimizer.min_lr=1e-06 \
  policy.megatron_cfg.optimizer.weight_decay=0 \
  policy.megatron_cfg.activation_checkpointing=True \
  policy.generation.temperature=1.0 \
  policy.generation.vllm_cfg.tensor_parallel_size=8 \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.8 \
  policy.generation.vllm_cfg.skip_tokenizer_init=False \
  loss_fn.reference_policy_kl_penalty=0 \
  loss_fn.ratio_clip_min=0.2 \
  loss_fn.ratio_clip_max=0.28 \
  loss_fn.use_on_policy_kl_approximation=True \
  loss_fn.use_importance_sampling_correction=True \
  loss_fn.sequence_level_importance_ratios=False \
  loss_fn.token_level_loss=True \
  loss_fn.force_on_policy_ratio=True ${EXTRA_OVERRIDES} \
  checkpointing.checkpoint_dir=${CKPT_DIR} \
  checkpointing.save_period=1000000 \
  checkpointing.keep_top_k=1 \
  ++checkpointing.metric_name=train:total_reward/mean \
  ++checkpointing.checkpoint_must_save_by=99:00:00:00 \
  logger.wandb_enabled=True \
  logger.wandb.name=${EXP_NAME} \
  logger.wandb.project=sna-nemo-rl \
  grpo.max_num_steps=20 \
  policy.generation.colocated.resources.num_nodes=8 \
  policy.generation.colocated.resources.gpus_per_node=8 \
  grpo.advantage_clip_low=-100 \
  grpo.advantage_clip_high=100 \
  loss_fn.truncated_importance_sampling_ratio=5 \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_max_turns=200 \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.swebench_agent_timeout=1800 \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_max_turns=200 \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_agent_timeout=1800 \
  ++env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_tests_timeout=60"

SBATCH_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Async GRPO RL training: training GPUs idle during rollout collection (~30min) and validation each step"}}'

SBATCH_CMD=(
  sbatch
  --nodes=16
  --account=coreai_dlalgo_nemorl
  --job-name="${EXP_NAME}"
  --partition=batch
  --time=04:00:00
  --gres=gpu:8
  --exclusive
  --dependency=singleton
  --comment="${SBATCH_COMMENT}"
  ray.sub
)

echo "========================================"
echo " VARIANT     : ${VARIANT} (tag: ${VARIANT_TAG})"
echo " EXP_NAME    : ${EXP_NAME}"
echo " Partition   : batch  Time: 04:00:00 (batch_long blocked for coreai_dlalgo_nemorl)"
echo " Branch      : $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "========================================"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY-RUN] sbatch:" ; echo "${SBATCH_CMD[*]}"
else
  "${SBATCH_CMD[@]}"
fi
