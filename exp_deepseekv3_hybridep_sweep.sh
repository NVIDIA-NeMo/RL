#!/bin/bash
# DeepSeek-V3 HybridEP Parameter Sweep
# GB200: 32 nodes x 4 GPUs = 128 GPUs
# Training: TP=1, PP=8, EP=16, Gen: vLLM TP=16
# Usage: bash exp_deepseekv3_hybridep_sweep.sh <num_sms> [cg]
#   num_sms: moe_hybridep_num_sms (8, 16, 24, 32)
#   cg: "cg" to enable CUDA graphs [attn,moe_router]

set -euo pipefail

NUM_SMS=${1:?Usage: $0 <num_sms> [cg]}
ENABLE_CG=${2:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "${SCRIPT_DIR}/cluster_config.sh"
setup_cluster_config "batch"
export_cluster_config

CONFIG_FILE="examples/configs/recipes/llm/performance/grpo-deepseek-v3-32n4g.yaml"
NUM_NODES=32
EP_SIZE=16
T_TP=1; T_PP=8

# HybridEP env
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$(( EP_SIZE < 16 ? EP_SIZE : 16 ))
export NVLINK_DOMAIN_SIZE=72
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export NCCL_P2P_NET_CHUNKSIZE=2097152
# TP=1, EP=16, PP=8: nodes/stage = (1*16)/4 = 4 → fits in NVLink domain
# No HYBRID_EP_MULTINODE needed
unset HYBRID_EP_MULTINODE

# Build run name
if [ "$ENABLE_CG" = "cg" ]; then
    RUN_NAME="deepseekv3_hybridep_sm${NUM_SMS}_cg"
    CG_ARGS="++policy.megatron_cfg.cuda_graph_impl=transformer_engine \
'++policy.megatron_cfg.cuda_graph_scope=[attn,moe_router]' \
++policy.megatron_cfg.cuda_graph_warmup_steps=3"
    CG_LABEL="ENABLED [attn,moe_router]"
else
    RUN_NAME="deepseekv3_hybridep_sm${NUM_SMS}"
    CG_ARGS=""
    CG_LABEL="DISABLED"
fi

echo "=================================================================="
echo "  DeepSeek-V3: HybridEP SM=${NUM_SMS}, EP=${EP_SIZE}, CG=${CG_LABEL}"
echo "  Nodes: ${NUM_NODES} x ${GPUS_PER_NODE} GPUs"
echo "  Training: TP=${T_TP}, PP=${T_PP}, EP=${EP_SIZE}"
echo "  Generation: vLLM TP=16"
echo "  Steps: 5"
echo "=================================================================="

WANDB_PROJECT="RL_GB200_deepseekv3_sweep"

COMMAND="NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo_math.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${EP_SIZE} \
policy.megatron_cfg.expert_tensor_parallel_size=1 \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
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
++policy.megatron_cfg.moe_token_dispatcher_type=flex \
++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep \
++policy.megatron_cfg.moe_hybridep_num_sms=${NUM_SMS} \
${CG_ARGS}" \

LOG_SUBDIR="${SCRIPT_DIR}/exp_logs/deepseekv3_sweep/sync_hybridep"
mkdir -p "$LOG_SUBDIR"

COMMAND="$COMMAND" \
CONTAINER=$CONTAINER \
HF_HOME=$HF_HOME \
HF_DATASETS_CACHE=$HF_DATASETS_CACHE \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="$MOUNTS" \
USE_MNNVL=$USE_MNNVL \
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
