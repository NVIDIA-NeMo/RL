#!/usr/bin/env bash

source /lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh
export HF_HOME=/lustre/fsw/portfolios/coreai/users/zhiyul/hf

# Disable NVLS to avoid OOM issue (from official test script)
export NCCL_NVLS_ENABLE=0

# Required for Megatron backend - specify CUDA compute capabilities
# Use the same value from CUDA_ARCH_LIST environment variable
export TORCH_CUDA_ARCH_LIST='9.0 10.0 12.0'

# Increase timeouts for large model checkpoint loading (DeepSeek-V3 671B)
# Default timeout is 600s (10min), increase to 3600s (1 hour)
export TORCH_DISTRIBUTED_INIT_TIMEOUT=3600
export NCCL_TIMEOUT=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1

# Model path
export MODEL_NAME=${NRL_DEEPSEEK_V3_BF16_CKPT:-"/lustre/fsw/portfolios/coreai/users/yifuw/hf_checkpoints/dsv3/DeepSeek-V3-BF16"}

# Ensure NRL_FORCE_REBUILD_VENVS is set to true only on first run
FLAG_FILE="/tmp/nrl_venv_built_$(whoami).flag"
if [ ! -f "$FLAG_FILE" ]; then
    export NRL_FORCE_REBUILD_VENVS=true
    touch "$FLAG_FILE"
    echo "First run detected: NRL_FORCE_REBUILD_VENVS=true"
else
    export NRL_FORCE_REBUILD_VENVS=false
    echo "Subsequent run: NRL_FORCE_REBUILD_VENVS=false (flag file exists)"
fi

uv run examples/run_grpo_math.py \
    --config examples/configs/recipes/llm/performance/dapo-deepseek-v3-64n8g.yaml \
    grpo.num_prompts_per_step=64 \
    grpo.num_generations_per_prompt=8 \
    grpo.max_num_steps=10 \
    policy.model_name=$MODEL_NAME \
    policy.tokenizer.name=$MODEL_NAME \
    logger.wandb_enabled=True \
    logger.wandb.project='grpo-dev-zhiyul' \
    logger.wandb.name=dapo-deepseek-v3-671b-megatron \
    cluster.num_nodes=64