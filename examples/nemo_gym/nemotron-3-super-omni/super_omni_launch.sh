#!/usr/bin/env bash
set -euo pipefail

# Launcher for Nemotron Super Omni multimodal Gym GRPO.
# Defaults mirror the validated SuperV3 handoff run while remaining
# overridable for checkpoint, data, scheduler, and smoke-test settings.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export EXP_NAME="${EXP_NAME:-grpo-super-v3-omni-textrl-svg-async}"
export MODEL_PATH="${MODEL_PATH:-/path/to/nemotron-super-omni-hf-checkpoint}"
export TRAIN_PATH="${TRAIN_PATH:-/path/to/super-omni-gym-train.jsonl}"
export VAL_PATH="${VAL_PATH:-${TRAIN_PATH}}"
export CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-tp8ep16cp2-async-gym.v1.yaml}"
export ENTRYPOINT="${ENTRYPOINT:-examples/nemo_gym/run_multimodal_grpo_nemo_gym.py}"
export SUPER_OMNI_MODE=true

# MTP off on both paths: the recipe sets megatron_cfg.mtp_num_layers=0 for
# training and refit, and this keeps vLLM speculative decoding off too.
export ENABLE_MTP_INFERENCE="${ENABLE_MTP_INFERENCE:-0}"
# The Omni recipes do not pin cudagraph_capture_sizes (only the text-only Super
# stage configs do), so there is nothing for the MTP path's Hydra delete to
# remove and attempting it aborts config composition.
export MTP_DROP_CUDAGRAPH_CAPTURE_SIZES="${MTP_DROP_CUDAGRAPH_CAPTURE_SIZES:-0}"

export CONTAINER="${CONTAINER:-/path/to/nemo-rl.sqsh}"
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/path/to/nemo-skills-sandbox.sqsh}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-/path/to/cache/nemo-rl-super-omni}"
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-/scratch:/scratch,/lustre:/lustre}"

export SLURM_PARTITION="${SLURM_PARTITION:-batch_long,batch}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_omni_vision}"
export SLURM_TIME_LIMIT="${SLURM_TIME_LIMIT:-4:0:0}"
export SBATCH_NUM_NODES="${SBATCH_NUM_NODES:-16}"

export WANDB_PROJ="${WANDB_PROJ:-grpo-superv3omni}"
export NRL_REFIT_BUFFER_MEMORY_RATIO="${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.006}"
export NRL_REFIT_NUM_BUFFERS="${NRL_REFIT_NUM_BUFFERS:-1}"
# The container's cached Megatron policy-worker environment does not include
# megatron.energon. Rebuild the isolated environments from this checkout unless
# the caller explicitly opts into a known-complete cache.
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-true}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NEMO_RL_LOG_GPU_MEMORY="${NEMO_RL_LOG_GPU_MEMORY:-1}"
export NEMO_RL_VLLM_PRECOMPUTED_IMG_SIZES="${NEMO_RL_VLLM_PRECOMPUTED_IMG_SIZES:-1}"

CHAT_TEMPLATE="${CHAT_TEMPLATE:-${MODEL_PATH}/chat_template.jinja}"
USER_EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
export EXTRA_OVERRIDES="policy.tokenizer.chat_template=${CHAT_TEMPLATE} \
policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template=${CHAT_TEMPLATE} \
${USER_EXTRA_OVERRIDES}"

exec "${SCRIPT_DIR}/../nemotron-3-super/super_launch.sh"
