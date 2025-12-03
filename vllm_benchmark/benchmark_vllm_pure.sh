#!/bin/bash
# Pure vLLM Offline Throughput Benchmark
# 
# This script measures "ideal" vLLM inference performance without NeMo-RL overhead.
# Uses the same dataset and GRPO-style batching for fair comparison.
#
# Usage:
#   ./benchmark_vllm_pure.sh

# Cluster Configuration
NUM_NODES=4
GPUS_PER_NODE=4
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# Parallelism Configuration
# TP=4: model sharded across 4 GPUs (one node)
# DP=4: 4 model replicas running in parallel
# Total GPUs = TP * PP * DP = 4 * 1 * 4 = 16
TP_SIZE=4
PP_SIZE=1
DP_SIZE=4  # Data parallel: runs 4 vLLM instances in parallel

# GRPO-style batch configuration (same as training)
NUM_PROMPTS=64
NUM_GENERATIONS=32

account=coreai_dlalgo_nemorl

# Define the command
COMMAND="uv run benchmark_vllm_pure.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --tp $TP_SIZE \
    --pp $PP_SIZE \
    --dp $DP_SIZE \
    --num-prompts $NUM_PROMPTS \
    --num-generations $NUM_GENERATIONS \
    --max-model-len 4096 \
    --max-tokens 2048 \
    --temperature 1.0 \
    --gpu-utilization 0.9"

# Environment variables for ray.sub
export CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/nemo_rl.sqsh
export HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home
export HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/cache
export MOUNTS="/lustre:/lustre"
export COMMAND="$COMMAND"

echo "============================================================"
echo "Pure vLLM Offline Throughput Benchmark"
echo "============================================================"
echo "Nodes: $NUM_NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo "TP: $TP_SIZE, PP: $PP_SIZE, DP: $DP_SIZE"
echo "GRPO-style: $NUM_PROMPTS prompts × $NUM_GENERATIONS generations"
echo "============================================================"
echo "Command: $COMMAND"
echo "============================================================"

sbatch \
    --nodes=${NUM_NODES} \
    --account=${account} \
    --job-name=vllm-pure-bench \
    --partition=batch \
    --time=01:00:00 \
    --gres=gpu:${GPUS_PER_NODE} \
    --segment ${NUM_NODES} \
    ray.sub

