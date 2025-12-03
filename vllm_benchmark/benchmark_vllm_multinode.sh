#!/bin/bash
# Run from the root of NeMo RL repo
# This script uses the same multi-node vLLM infrastructure as grpo.py

# Cluster Configuration
NUM_NODES=4
GPUS_PER_NODE=4
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# Parallelism Configuration
# Example: TP=4 (one model shard per node), DP=4 (4 model replicas) -> 4 * 4 = 16 GPUs
# DP is automatically calculated as: TOTAL_GPUS / (TP * PP)
TP_SIZE=4
PP_SIZE=1
EP_SIZE=1  # Expert parallel for MoE models (set to 1 for non-MoE)

account=coreai_dlalgo_nemorl

# Define the command (uses same VllmGeneration as grpo.py)
COMMAND="NRL_FORCE_REBUILD_VENVS=true uv run benchmark_vllm_standalone.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --num-nodes $NUM_NODES \
    --gpus-per-node $GPUS_PER_NODE \
    --tp $TP_SIZE \
    --pp $PP_SIZE \
    --ep $EP_SIZE \
    --num-prompts 128 \
    --n 32 \
    --max-model-len 4096 \
    --max-tokens 2048 \
    --temperature 1.0 \
    --gpu-utilization 0.7"

# Environment variables for ray.sub
export CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/nemo_rl.sqsh
export HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home
export HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/cache
export MOUNTS="/lustre:/lustre"
export COMMAND="$COMMAND"

echo "Submitting benchmark job on $NUM_NODES nodes ($TOTAL_GPUS GPUs)..."
echo "Command: $COMMAND"

sbatch \
    --nodes=${NUM_NODES} \
    --account=${account} \
    --job-name=vllm-bench-16gpus \
    --partition=batch \
    --time=01:00:00 \
    --gres=gpu:${GPUS_PER_NODE} \
    --segment ${NUM_NODES} \
    ray.sub

