#!/usr/bin/env bash
# Submit one of: ray_only, mxfp8, hybridep, both
# Usage: VARIANT=ray_only ./submit_perf_variant.sh
set -euo pipefail

: "${VARIANT:?VARIANT is required: ray_only | mxfp8 | hybridep | both}"
DRY_RUN="${DRY_RUN:-false}"

REPO_DIR="/lustre/fsw/portfolios/coreai/users/sna/repos/nemo-rl-qwen-swe"
CACHE_BASE="/lustre/fsw/portfolios/coreai/users/sna/.cache/qwen3_235b_swe"
HF_HOME_DIR="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home"
DATA_PATH="/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/nano/dataset/rl/swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl"
CONFIG_PATH="${REPO_DIR}/grpo_qwen3_235b_swe.yaml"

# Container + branch + vllm-wheel choices are per-variant.
# Default = current super-v3 perf chain (vllm 0.13 container, perf-patch branch).
DEFAULT_CONTAINER="/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/nvcr.io/nvidian/nemo-rl:7684dc2-45115915.squashfs"
DEFAULT_BRANCH="sj/super-v3-perf-patch"
DEFAULT_VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v0.13.0/vllm-0.13.0-cp38-abi3-manylinux_2_31_x86_64.whl"

# May-13 container (vllm 0.17.1+cu130, torch 2.10, py3.13) is required for MXFP8 EMULATION
# bypass — it has ModelOptMxFp8Config + ModelOptMxFp8LinearMethod classes.
MXFP8_CONTAINER="/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/nvcr.io/nvidian/nemo-rl:4641794-51006907.squashfs"
MXFP8_BRANCH="sj/super-v3-mxfp8-bypass"

CONTAINER="${DEFAULT_CONTAINER}"
BRANCH="${DEFAULT_BRANCH}"
VLLM_WHEEL_URL="${DEFAULT_VLLM_WHEEL_URL}"

# Variant-specific overrides and labels
case "$VARIANT" in
  ray_only)
    VARIANT_TAG="rayonly"
    EXTRA_OVERRIDES=""
    EXTRA_ENVS=""
    ;;
  mxfp8)
    VARIANT_TAG="mxfp8"
    CONTAINER="${MXFP8_CONTAINER}"
    BRANCH="${MXFP8_BRANCH}"
    # May-13 container has vllm 0.17.1 prebuilt in /root/.cache/uv. Don't force the 0.13 wheel.
    VLLM_WHEEL_URL=""
    # Trust the container venv — torch/vllm/TE pins in pyproject.toml are for the
    # 0.13-container path and conflict with May-13 torch 2.10.
    export NRL_FORCE_REBUILD_VENVS_OVERRIDE="false"
    EXTRA_OVERRIDES="  policy.generation.vllm_cfg.precision=fp8 \
  ++policy.generation.vllm_cfg.is_mx=true \
  ++policy.generation.vllm_cfg.pow2_weight_scaling_factors=true \
  ++policy.generation.vllm_cfg.pow2_activation_scaling_factors=true"
    # EMULATION backend is pure-torch, no flashinfer dependencies. MNNVL not available on H100.
    EXTRA_ENVS=""
    ;;
  hybridep)
    VARIANT_TAG="hybridep"
    EXTRA_OVERRIDES="  policy.megatron_cfg.moe_token_dispatcher_type=flex \
  ++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep \
  policy.megatron_cfg.moe_shared_expert_overlap=True"
    EXTRA_ENVS="  NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8 \
  USE_MNNVL=False"
    export TORCH_CUDA_ARCH_LIST_OVERRIDE="9.0"
    ;;
  both)
    VARIANT_TAG="mxfp8-hybridep"
    CONTAINER="${MXFP8_CONTAINER}"
    BRANCH="${MXFP8_BRANCH}"
    VLLM_WHEEL_URL=""
    export NRL_FORCE_REBUILD_VENVS_OVERRIDE="false"
    EXTRA_OVERRIDES="  policy.generation.vllm_cfg.precision=fp8 \
  ++policy.generation.vllm_cfg.is_mx=true \
  ++policy.generation.vllm_cfg.pow2_weight_scaling_factors=true \
  ++policy.generation.vllm_cfg.pow2_activation_scaling_factors=true \
  policy.megatron_cfg.moe_token_dispatcher_type=flex \
  ++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep \
  policy.megatron_cfg.moe_shared_expert_overlap=True"
    EXTRA_ENVS="  NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8 \
  USE_MNNVL=False \
  PYTHONPATH=/lustre/fsw/portfolios/coreai/users/sna/hybridep_overlay/site-packages:${PYTHONPATH}"
    export TORCH_CUDA_ARCH_LIST_OVERRIDE="9.0"
    ;;
  *)
    echo "Unknown VARIANT: $VARIANT" >&2
    exit 1
    ;;
esac

EXP_NAME="qwen3-235b-swe-perf-${VARIANT_TAG}"
CKPT_DIR="${REPO_DIR}/results/${EXP_NAME}"
LOG_DIR="logs/${EXP_NAME}"

cd "${REPO_DIR}"

# Ensure correct branch is checked out for this variant.
git checkout "${BRANCH}"
git pull --ff-only

export OMP_NUM_THREADS=16
export CONTAINER
export MOUNTS="/lustre:/lustre,${REPO_DIR}:${REPO_DIR},${REPO_DIR}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"

# Sandbox: baseline 11777073 skipped this. SWE-bench apptainer is installed via SETUP_COMMAND
# instead of via a separate ray.sub sandbox srun.
unset SANDBOX_CONTAINER SANDBOX_COMMAND SANDBOX_ENV_VARS 2>/dev/null || true

# Setup command (apptainer + cache seed). Heredoc stays single-quoted to keep
# loop vars (RET, attempt, NVCC_FOUND, ...) literal. The vllm wheel pin is
# templated via __VLLM_WHEEL_LOC__ placeholder, substituted after the heredoc.
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

echo "[CACHE SEED] Clearing stale /tmp caches and seeding from Lustre..."
rm -rf /tmp/nemo_rl_vllm_cache /tmp/nemo_rl_vllm_cache_*
rm -rf "/tmp/nemo_rl_inductor_cache" "/tmp/nemo_rl_triton_cache"
mkdir -p "/tmp/nemo_rl_inductor_cache" "/tmp/nemo_rl_triton_cache"

__UV_SYNC_BLOCK__
SETUP_EOF

# Substitute __UV_SYNC_BLOCK__: rebuild from pyproject only when VLLM_WHEEL_URL is set
# (the default super-v3 path). When empty (MXFP8 May-13 container), trust the container's
# preinstalled vllm 0.17.1 + torch 2.10 + TE 2.14.1 venv to avoid pyproject pin conflicts.
if [[ -n "${VLLM_WHEEL_URL}" ]]; then
  UV_SYNC_BLOCK="VLLM_USE_PRECOMPILED=1 \\
VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_WHEEL_URL} \\
UV_HTTP_TIMEOUT=3600 \\
uv sync --frozen"
else
  UV_SYNC_BLOCK="echo \"[SETUP] Skipping uv sync — using container preinstalled venv (vllm 0.17.1 + torch 2.10)\""
fi
SETUP_COMMAND="${SETUP_COMMAND//__UV_SYNC_BLOCK__/${UV_SYNC_BLOCK}}"
export SETUP_COMMAND

# Build COMMAND (matches baseline 11777073 with VARIANT overrides and env additions).
# When VLLM_WHEEL_URL is empty (mxfp8/both variants on May-13 container), drop the wheel
# pin so uv uses the container's preinstalled vllm 0.17.1 cache.
if [[ -n "${VLLM_WHEEL_URL}" ]]; then
  VLLM_WHEEL_ENV="VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_WHEEL_URL}"
else
  VLLM_WHEEL_ENV=""
fi

export COMMAND="CUDA_HOME=/usr/local/cuda \
  CUDA_PATH=/usr/local/cuda \
  NRL_VLLM_USE_V1=1 \
  NRL_WG_USE_RAY_REF=1 \
  ${VLLM_WHEEL_ENV} \
  WANDB_API_KEY=${WANDB_API_KEY:-cd4db01aafd025d20369f8eee65e6292c28bfe0d} \
  HUGGINGFACE_TOKEN=${HF_TOKEN:-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} \
  HF_HOME=${HF_HOME_DIR} \
  HF_DATASETS_CACHE=${HF_HOME_DIR}/cache \
  UV_CACHE_DIR=/lustre/fsw/portfolios/coreai/users/sna/uv_cache \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  VLLM_CACHE_ROOT=${CACHE_BASE}/vllm_compile_cache \
  DG_JIT_CACHE_DIR=${CACHE_BASE}/vllm_compile_cache/deep_gemm \
  VLLM_DEEP_GEMM_WARMUP=skip \
  NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS_OVERRIDE:-true} \
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
  grpo.max_num_steps=1000000 \
  policy.generation.colocated.resources.num_nodes=8 \
  policy.generation.colocated.resources.gpus_per_node=8 \
  grpo.advantage_clip_low=-100 \
  grpo.advantage_clip_high=100 \
  loss_fn.truncated_importance_sampling_ratio=5 \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_max_turns=200 \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.swebench_agent_timeout=1800 \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_max_turns=200 \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_agent_timeout=1800"

# OccupiedIdleGPUsJobReaper exemption comment (matches baseline)
SBATCH_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Async GRPO RL training: training GPUs idle during rollout collection (~30min) and validation each step"}}'

SBATCH_CMD=(
  sbatch
  --nodes=16
  --account=coreai_dlalgo_nemorl
  --job-name="${EXP_NAME}"
  --partition=batch
  --time=4:0:0
  --gres=gpu:8
  --exclusive
  --dependency=singleton
  --comment="${SBATCH_COMMENT}"
  ray.sub
)

echo "========================================"
echo " VARIANT     : ${VARIANT} (tag: ${VARIANT_TAG})"
echo " EXP_NAME    : ${EXP_NAME}"
echo " Container   : ${CONTAINER}"
echo " Branch      : $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo " vllm wheel  : ${VLLM_WHEEL_URL:-<container default>}"
echo " Extra ovrd  : ${EXTRA_OVERRIDES:-<none>}"
echo " Extra envs  : ${EXTRA_ENVS:-<none>}"
echo "========================================"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY-RUN] sbatch invocation:"
  echo "${SBATCH_CMD[*]}"
  echo "[DRY-RUN] FULL COMMAND:"
  echo "${COMMAND}"
else
  "${SBATCH_CMD[@]}"
fi
