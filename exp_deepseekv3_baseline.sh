#!/bin/bash
# DeepSeek-V3 Baseline (AllToAll, no HybridEP) for sweep comparison
# GB200: 32 nodes x 4 GPUs = 128 GPUs
# Training: TP=1, PP=8, EP=16, Gen: vLLM TP=16
# Usage: bash exp_deepseekv3_baseline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "${SCRIPT_DIR}/cluster_config.sh"
setup_cluster_config "batch"
export_cluster_config

CONFIG_FILE="examples/configs/recipes/llm/performance/grpo-deepseek-v3-32n4g.yaml"
NUM_NODES=32
EP_SIZE=16
RUN_NAME="deepseekv3_baseline_ep${EP_SIZE}"

# No HybridEP
unset USE_MNNVL
unset NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN

echo "=================================================================="
echo "  DeepSeek-V3: Baseline AllToAll, EP=${EP_SIZE}"
echo "  Nodes: ${NUM_NODES} x ${GPUS_PER_NODE} GPUs"
echo "  Training: TP=1, PP=8, EP=${EP_SIZE}"
echo "  Generation: vLLM TP=16 (BF16, NOT TP=32 which hits FP8 block error)"
echo "  Steps: 5"
echo "=================================================================="

WANDB_PROJECT="RL_GB200_deepseekv3_sweep"

COMMAND="NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo_math.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.megatron_cfg.tensor_model_parallel_size=1 \
policy.megatron_cfg.expert_model_parallel_size=${EP_SIZE} \
policy.megatron_cfg.expert_tensor_parallel_size=1 \
policy.megatron_cfg.pipeline_model_parallel_size=8 \
policy.megatron_cfg.context_parallel_size=1 \
policy.megatron_cfg.sequence_parallel=False \
policy.train_micro_batch_size=1 \
policy.max_total_sequence_length=1536 \
policy.generation.vllm_cfg.tensor_parallel_size=16 \
policy.megatron_cfg.activation_checkpointing=True \
++policy.megatron_cfg.recompute_granularity=selective \
++policy.megatron_cfg.recompute_num_layers=null \
'++policy.megatron_cfg.recompute_modules=[moe_act]' \
policy.sequence_packing.enabled=False \
policy.train_global_batch_size=512 \
grpo.async_grpo.enabled=false \
grpo.num_prompts_per_step=16 \
grpo.num_generations_per_prompt=32 \
grpo.max_num_steps=5 \
grpo.val_period=1000 \
checkpointing.enabled=false \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${RUN_NAME}' \
++policy.megatron_cfg.moe_token_dispatcher_type=alltoall" \

LOG_SUBDIR="${SCRIPT_DIR}/exp_logs/deepseekv3_sweep/sync_baseline_ep${EP_SIZE}"
mkdir -p "$LOG_SUBDIR"

COMMAND="$COMMAND" \
CONTAINER=$CONTAINER \
HF_HOME=$HF_HOME \
HF_DATASETS_CACHE=$HF_DATASETS_CACHE \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="$MOUNTS" \
BASE_LOG_DIR="$LOG_SUBDIR" \
sbatch \
    --nodes=${NUM_NODES} \
    --account=${ACCOUNT} \
    --job-name=${RUN_NAME} \
    --partition=${PARTITION} \
    --time=04:00:00 \
    ${GRES_FLAG} \
    --segment 16 \
    ray.sub

echo "[INFO] ${RUN_NAME} submitted!"
