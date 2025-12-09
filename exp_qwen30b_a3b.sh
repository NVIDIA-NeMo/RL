#!/bin/bash
# Qwen/Qwen3-30B-A3B (MoE)
# Auto-detects cluster type and loads parallelism from model_configs.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source cluster configuration (auto-detect H100/GB200)
source "${SCRIPT_DIR}/cluster_config.sh"
setup_cluster_config "batch"
export_cluster_config

echo "============================================"
echo "Launching Qwen3-30B-A3B GRPO Training"
echo "Cluster: ${CLUSTER_TYPE}, GPUs/Node: ${GPUS_PER_NODE}"
echo "============================================"

# Load model-specific config from YAML
eval $(python3 "${SCRIPT_DIR}/get_model_config.py" qwen30b ${CLUSTER_TYPE,,})

# Calculate number of nodes
NUM_NODES=$((NUM_GPUS / GPUS_PER_NODE))

# Print configuration
echo "[INFO] Model: ${MODEL_NAME}"
echo "[INFO] Nodes: ${NUM_NODES}, Total GPUs: ${NUM_GPUS}"
echo "[INFO] Generation: TP=${G_TP}, PP=${G_PP}"
echo "[INFO] Training: TP=${T_TP}, CP=${T_CP}, EP=${T_EP}, PP=${T_PP}"
echo "[INFO] Batch: Rollout=${ROLLOUT_GBS}, Train=${TRAIN_GBS}"

account=coreai_dlalgo_nemorl
WANDB_PROJECT="sync-grpo-${CLUSTER_TYPE,,}-benchmark"

COMMAND="NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo_math.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.expert_parallel_size=${G_EP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=${T_SP} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.num_prompts_per_step=${NUM_PROMPTS} \
grpo.num_generations_per_prompt=${NUM_GENERATIONS} \
policy.sequence_packing.enabled=True \
policy.train_global_batch_size=${TRAIN_GBS} \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='Qwen30B_A3B_N${NUM_NODES}xG${GPUS_PER_NODE}_Ttp${T_TP}pp${T_PP}ep${T_EP}cp${T_CP}_Gtp${G_TP}pp${G_PP}ep${G_EP}'" \
CONTAINER=$CONTAINER \
HF_HOME=$HF_HOME \
HF_DATASETS_CACHE=$HF_DATASETS_CACHE \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="$MOUNTS" \
sbatch \
    --nodes=${NUM_NODES} \
    --account=${account} \
    --job-name=qwen30b-a3b-N${NUM_NODES}xG${GPUS_PER_NODE}-T.tp${T_TP}.pp${T_PP}.ep${T_EP}-G.tp${G_TP}.pp${G_PP} \
    --partition=${PARTITION} \
    --time=04:00:00 \
    ${GRES_FLAG} \
    --segment ${NUM_NODES} \
    ray.sub
