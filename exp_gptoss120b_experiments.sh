#!/bin/bash
# ============================================
# GPT-OSS-120B Experiments
# ============================================
# Configurations for 120B MoE model on reduced node counts.
# 120B is ~6x larger than 20B - requires aggressive parallelism.
#
# Model: openai/gpt-oss-120b (8 experts, ~120B params)
# BF16 model weights: ~240GB
# Generation requires TP=8 minimum (30GB/GPU)
#
# Experiments:
#   1-4. optimized_2node*: EP=8, TP=2, PP=1, DP=1 — 2 nodes (likely OOM)
#   5-6. optimized_4node*: EP=8, TP=2, PP=1, DP=2 — 4 nodes (recommended min)
#   7-8. optimized_8node*: EP=8, TP=4, PP=2, DP=1 — 8 nodes (safe)
#
# Usage: ./exp_gptoss120b_experiments.sh [experiment_name]
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Deactivate conda to avoid Ray version conflicts
if echo "$PATH" | grep -qE "conda|miniconda"; then
    echo "[INFO] Removing conda/miniconda from PATH to avoid Ray version conflict..."
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -vE "conda|miniconda" | tr '\n' ':' | sed 's/:$//')
fi
unset CONDA_PREFIX CONDA_DEFAULT_ENV 2>/dev/null || true

# Source cluster configuration
source "${SCRIPT_DIR}/cluster_config.sh"
setup_cluster_config "batch"
export_cluster_config

EXPERIMENT="${1:-optimized_2node_debug}"
CONFIG_FILE="examples/configs/recipes/llm/grpo-gptoss-120b-8n8g-megatron.yaml"
account=coreai_dlalgo_nemorl
WANDB_PROJECT="sync-grpo-${CLUSTER_TYPE,,}-gptoss120b-exp"

# ========================================
# cuDNN Setup for FusedAttention
# ========================================
# ray.sub pip-installs nvidia-cudnn-cu12==9.19.0.56 into /opt/nemo_rl_venv on ALL compute nodes.
# PYTHONPATH fix: prepend SLURM_SUBMIT_DIR so the Lustre nemo_rl (with _worker_cudnn_lib)
# is imported before container's /opt/nemo_rl_venv version. PYTHONPATH takes precedence.
CUDNN_SETUP="export PYTHONPATH=\${SLURM_SUBMIT_DIR}:\${PYTHONPATH:-} && export LD_LIBRARY_PATH=/opt/nemo_rl_venv/lib/python3.12/site-packages/nvidia/cudnn/lib:\${LD_LIBRARY_PATH:-} && export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1 && export HF_DATASETS_OFFLINE=1 && export NCCL_CUMEM_ENABLE=0 && export PYTORCH_ALLOC_CONF=expandable_segments:True && "

# Organize logs
export BASE_LOG_DIR="${SCRIPT_DIR}/exp_logs/gpt-oss-120b"
mkdir -p "$BASE_LOG_DIR"

echo ""
echo "============================================"
echo "GPT-OSS-120B Experiments"
echo "============================================"
echo "  Experiment: ${EXPERIMENT}"
echo "  Config: ${CONFIG_FILE}"
echo "  Cluster: ${CLUSTER_TYPE}"
echo "============================================"
echo ""

case "$EXPERIMENT" in

    # ========================================
    # 1. Optimized 2-Node (16 GPUs)
    # EP=8, TP=2, PP=1 — maximum expert parallelism
    # Reduced seq_len and batch for memory
    # ========================================
    optimized_2node)
        NUM_NODES=2
        G_TP=8   # 120B needs TP=8 for vLLM (240GB / 8 = 30GB/GPU)
        G_PP=1
        T_EP=8
        T_TP=2
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS120B_optimized_2node"
        
        echo "[INFO] Running optimized 2-node configuration"
        echo "  - Nodes: ${NUM_NODES} (16 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP} (DP=1)"
        echo "  - Generation: TP=${G_TP}"
        echo "  - max_total_sequence_length: 2048 (reduced from 4096)"
        echo "  - defer_fp32_logits: true"
        echo "  - activation_checkpointing: true"
        
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.70 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.max_total_sequence_length=2048 \
policy.train_global_batch_size=128 \
grpo.num_prompts_per_step=32 \
grpo.num_generations_per_prompt=16 \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 2. Optimized 2-Node DEBUG (16 GPUs)
    # Same as above + NVTE_DEBUG to show attention backend
    # ========================================
    optimized_2node_debug)
        NUM_NODES=2
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=2
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS120B_optimized_2node_debug"
        
        echo "[INFO] Running optimized 2-node configuration with NVTE DEBUG"
        echo "  - Nodes: ${NUM_NODES} (16 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP} (DP=1)"
        echo "  - Generation: TP=${G_TP}"
        echo "  - NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2"
        echo "  - max_total_sequence_length: 2048"
        echo "  - defer_fp32_logits: true"
        echo "  - activation_checkpointing: true"
        
        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.70 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.max_total_sequence_length=2048 \
policy.train_global_batch_size=128 \
grpo.num_prompts_per_step=32 \
grpo.num_generations_per_prompt=16 \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 3. Optimized 2-Node + Sequence Packing (16 GPUs)
    # Same as debug + sequence packing enabled
    # ========================================
    optimized_2node_seqpack)
        NUM_NODES=2
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=2
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS120B_optimized_2node_seqpack"
        
        echo "[INFO] Running optimized 2-node with NVTE DEBUG + Sequence Packing"
        echo "  - Nodes: ${NUM_NODES} (16 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP} (DP=1)"
        echo "  - Generation: TP=${G_TP}"
        echo "  - sequence_packing: enabled"
        echo "  - max_total_sequence_length: 2048"
        echo "  - defer_fp32_logits: true"
        echo "  - activation_checkpointing: true"
        
        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.70 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.sequence_packing.enabled=true \
policy.max_total_sequence_length=2048 \
policy.train_global_batch_size=128 \
grpo.num_prompts_per_step=32 \
grpo.num_generations_per_prompt=16 \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 4. Optimized 2-Node Unfused Attention (16 GPUs)
    # Forces UnfusedDotProductAttention for comparison
    # ========================================
    optimized_2node_unfused)
        NUM_NODES=2
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=2
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS120B_optimized_2node_unfused"
        
        echo "[INFO] Running optimized 2-node with UNFUSED attention"
        echo "  - Nodes: ${NUM_NODES} (16 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP} (DP=1)"
        echo "  - Generation: TP=${G_TP}"
        echo "  - attention_backend=unfused"
        echo "  - max_total_sequence_length: 2048"
        
        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.70 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.attention_backend=unfused \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.max_total_sequence_length=2048 \
policy.train_global_batch_size=128 \
grpo.num_prompts_per_step=32 \
grpo.num_generations_per_prompt=16 \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 5. 4-Node (32 GPUs) — Recommended minimum
    # EP=8, TP=2, PP=1, DP=2
    # DP=2 enables distributed optimizer (halves optimizer memory)
    # ========================================
    optimized_4node)
        NUM_NODES=4
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=2
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS120B_optimized_4node"
        
        echo "[INFO] Running 4-node configuration (recommended minimum for 120B)"
        echo "  - Nodes: ${NUM_NODES} (32 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}, DP=2"
        echo "  - Generation: TP=${G_TP}"
        echo "  - Distributed optimizer enabled (DP=2)"
        echo "  - max_total_sequence_length: 2048"
        echo "  - defer_fp32_logits: true"
        echo "  - activation_checkpointing: true"
        
        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.70 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.max_total_sequence_length=2048 \
policy.train_global_batch_size=128 \
grpo.num_prompts_per_step=32 \
grpo.num_generations_per_prompt=16 \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 5b. 4-Node + moe_permute_fusion + seqpack (32 GPUs)
    # EP=8, TP=2, PP=1, DP=2 with both optimizations
    # ========================================
    moe_seqpack_4node)
        NUM_NODES=4
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=2
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS120B_moe_seqpack_4node"
        
        echo "[INFO] Running 4-node with moe_permute_fusion + sequence_packing"
        echo "  - Nodes: ${NUM_NODES} (32 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}, DP=2"
        echo "  - Generation: TP=${G_TP}"
        echo "  - moe_permute_fusion: true"
        echo "  - sequence_packing: true"
        echo "  - defer_fp32_logits: true"
        echo "  - activation_checkpointing: true"
        
        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.70 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.moe_permute_fusion=true \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.sequence_packing.enabled=true \
policy.max_total_sequence_length=2048 \
policy.train_global_batch_size=128 \
grpo.num_prompts_per_step=32 \
grpo.num_generations_per_prompt=16 \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # NCCL DIAG: alltoall + seqpack + 4node (32 GPUs)
    # moe_permute_fusion=true, dispatcher=alltoall, SP=true
    # ========================================
    alltoall_seqpack_4node)
        NUM_NODES=4
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=4
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS120B_alltoall_seqpack_4node"

        echo "[INFO] NCCL DIAG: alltoall dispatcher + seqpack=TRUE + fusion=TRUE (4 nodes)"

        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.70 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.moe_permute_fusion=true \
policy.megatron_cfg.moe_token_dispatcher_type=alltoall \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.sequence_packing.enabled=true \
policy.max_total_sequence_length=2048 \
policy.train_global_batch_size=128 \
grpo.num_prompts_per_step=32 \
grpo.num_generations_per_prompt=16 \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # NCCL DIAG: alltoall + nopack + 4node (32 GPUs)
    # moe_permute_fusion=true, dispatcher=alltoall, SP=false
    # ========================================
    alltoall_nopack_4node)
        NUM_NODES=4
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=4
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS120B_alltoall_nopack_4node"

        echo "[INFO] NCCL DIAG: alltoall dispatcher + seqpack=FALSE + fusion=TRUE (4 nodes)"

        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.70 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.moe_permute_fusion=true \
policy.megatron_cfg.moe_token_dispatcher_type=alltoall \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.sequence_packing.enabled=false \
policy.max_total_sequence_length=2048 \
policy.train_global_batch_size=128 \
grpo.num_prompts_per_step=32 \
grpo.num_generations_per_prompt=16 \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # NCCL DIAG: allgather + nopack + 4node (32 GPUs)
    # moe_permute_fusion=true, dispatcher=allgather (default), SP=false
    # ========================================
    allgather_nopack_4node)
        NUM_NODES=4
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=4
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS120B_allgather_nopack_4node"

        echo "[INFO] NCCL DIAG: allgather dispatcher + seqpack=FALSE + fusion=TRUE (4 nodes)"

        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.70 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.moe_permute_fusion=true \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.sequence_packing.enabled=false \
policy.max_total_sequence_length=2048 \
policy.train_global_batch_size=128 \
grpo.num_prompts_per_step=32 \
grpo.num_generations_per_prompt=16 \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # 7b. 8-Node + moe_permute_fusion + seqpack (64 GPUs)
    # EP=8, TP=4, PP=2 with both optimizations (explicit)
    # ========================================
    moe_seqpack_8node)
        NUM_NODES=8
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=4
        T_PP=2
        T_CP=1
        WANDB_NAME="GPTOSS120B_moe_seqpack_8node"
        
        echo "[INFO] Running 8-node with moe_permute_fusion + sequence_packing"
        echo "  - Nodes: ${NUM_NODES} (64 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}, DP=1"
        echo "  - Generation: TP=${G_TP}"
        echo "  - moe_permute_fusion: true"
        echo "  - sequence_packing: true"
        echo "  - defer_fp32_logits: true"
        echo "  - activation_checkpointing: true"
        
        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.47 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.moe_permute_fusion=true \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.sequence_packing.enabled=true \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # NCCL DIAG: allgather + nopack + 8node (64 GPUs)
    # moe_permute_fusion=true (fixed), dispatcher=allgather, SP=false
    # ========================================
    allgather_nopack_8node)
        NUM_NODES=8
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=4
        T_PP=2
        T_CP=1
        WANDB_NAME="GPTOSS120B_allgather_nopack_8node"

        echo "[INFO] NCCL DIAG: allgather + seqpack=FALSE + fusion=TRUE (8 nodes)"
        echo "  - Nodes: ${NUM_NODES} (64 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}"
        echo "  - Generation: TP=${G_TP}"

        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.50 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.moe_permute_fusion=true \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.sequence_packing.enabled=false \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # NCCL DIAG: alltoall + seqpack + 8node (64 GPUs)
    # moe_permute_fusion=true (fixed), dispatcher=alltoall, SP=true
    # ========================================
    alltoall_seqpack_8node)
        NUM_NODES=8
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=4
        T_PP=2
        T_CP=1
        WANDB_NAME="GPTOSS120B_alltoall_seqpack_8node"

        echo "[INFO] NCCL DIAG: alltoall + seqpack=TRUE + fusion=TRUE (8 nodes)"
        echo "  - Nodes: ${NUM_NODES} (64 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}"
        echo "  - Generation: TP=${G_TP}"

        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.47 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.moe_permute_fusion=true \
policy.megatron_cfg.moe_token_dispatcher_type=alltoall \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.sequence_packing.enabled=true \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # NCCL DIAG: alltoall + nopack + 8node (64 GPUs)
    # moe_permute_fusion=true (fixed), dispatcher=alltoall, SP=false
    # ========================================
    alltoall_nopack_8node)
        NUM_NODES=8
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=4
        T_PP=2
        T_CP=1
        WANDB_NAME="GPTOSS120B_alltoall_nopack_8node"

        echo "[INFO] NCCL DIAG: alltoall + seqpack=FALSE + fusion=TRUE (8 nodes)"
        echo "  - Nodes: ${NUM_NODES} (64 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}"
        echo "  - Generation: TP=${G_TP}"

        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.50 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.moe_permute_fusion=true \
policy.megatron_cfg.moe_token_dispatcher_type=alltoall \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.sequence_packing.enabled=false \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # 6. 4-Node + Unfused Attention (32 GPUs)
    # Same as 4-node + attention_backend=unfused
    # ========================================
    optimized_4node_unfused)
        NUM_NODES=4
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=2
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS120B_optimized_4node_unfused"
        
        echo "[INFO] Running 4-node with UNFUSED attention"
        echo "  - Nodes: ${NUM_NODES} (32 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}, DP=2"
        echo "  - Generation: TP=${G_TP}"
        echo "  - attention_backend=unfused"
        echo "  - Distributed optimizer enabled (DP=2)"
        
        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.70 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.attention_backend=unfused \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.max_total_sequence_length=2048 \
policy.train_global_batch_size=128 \
grpo.num_prompts_per_step=32 \
grpo.num_generations_per_prompt=16 \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 7. 8-Node (64 GPUs) — Safe / Original recipe scale
    # EP=8, TP=4, PP=2, DP=1
    # ========================================
    optimized_8node)
        NUM_NODES=8
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=4
        T_PP=2
        T_CP=1
        WANDB_NAME="GPTOSS120B_optimized_8node"
        
        echo "[INFO] Running 8-node configuration (safe for 120B)"
        echo "  - Nodes: ${NUM_NODES} (64 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}, DP=1"
        echo "  - Generation: TP=${G_TP}"
        echo "  - max_total_sequence_length: 4096"
        echo "  - defer_fp32_logits: true"
        echo "  - activation_checkpointing: true"
        
        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.70 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 8. 8-Node + Unfused Attention (64 GPUs)
    # ========================================
    optimized_8node_unfused)
        NUM_NODES=8
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=4
        T_PP=2
        T_CP=1
        WANDB_NAME="GPTOSS120B_optimized_8node_unfused"
        
        echo "[INFO] Running 8-node with UNFUSED attention"
        echo "  - Nodes: ${NUM_NODES} (64 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}, DP=1"
        echo "  - Generation: TP=${G_TP}"
        echo "  - attention_backend=unfused"
        
        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.70 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.attention_backend=unfused \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # INTERACTIVE MODE - 2 Nodes
    # ========================================
    interactive)
        NUM_NODES=2
        WANDB_NAME="GPTOSS120B_interactive"
        
        echo "[INFO] Launching INTERACTIVE 2-node Ray cluster for 120B"
        echo "  - Nodes: ${NUM_NODES} (16 GPUs)"
        echo ""
        echo "  After job starts, run: bash JOBID-attach.sh"
        echo ""
        
        COMMAND=""
        ;;
    
    # ========================================
    # INTERACTIVE MODE - 8 Nodes (batch partition)
    # ========================================
    interactive_8node)
        NUM_NODES=8
        USE_BATCH_PARTITION=1
        WANDB_NAME="GPTOSS120B_interactive_8node"
        
        echo "[INFO] Launching INTERACTIVE 8-node Ray cluster for 120B (batch partition)"
        echo "  - Nodes: ${NUM_NODES} (64 GPUs)"
        echo "  - Partition: batch"
        echo ""
        echo "  After job starts, run: bash JOBID-attach.sh"
        echo "  Then set cuDNN and run your command (CUDA_DEVICE_MAX_CONNECTIONS=1 avoids logprob-stage hang with sequence parallel):"
        echo ""
        echo "    export CUDA_DEVICE_MAX_CONNECTIONS=1"
        echo "    export PIP_CUDNN_LIB=\$(uv run python3 -c \"import nvidia.cudnn, pathlib; print(pathlib.Path(list(nvidia.cudnn.__path__)[0]) / 'lib')\")"
        echo "    export LD_LIBRARY_PATH=\${PIP_CUDNN_LIB}:\${LD_LIBRARY_PATH}"
        echo "    ln -sf libcudnn.so.9 \${PIP_CUDNN_LIB}/libcudnn.so"
        echo ""
        echo "    NVTE_DEBUG=1 CUDNN_INSTALL=0 NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \\"
        echo "      --config examples/configs/recipes/llm/grpo-gptoss-120b-8n8g-megatron.yaml \\"
        echo "      cluster.num_nodes=8 cluster.gpus_per_node=8 \\"
        echo "      policy.generation.vllm_cfg.tensor_parallel_size=8 \\"
        echo "      policy.megatron_cfg.tensor_model_parallel_size=4 \\"
        echo "      policy.megatron_cfg.expert_model_parallel_size=8 \\"
        echo "      policy.megatron_cfg.pipeline_model_parallel_size=2 \\"
        echo "      policy.megatron_cfg.moe_permute_fusion=true \\"
        echo "      policy.sequence_packing.enabled=true \\"
        echo "      grpo.max_num_steps=3 checkpointing.enabled=false \\"
        echo "      logger.wandb_enabled=True logger.wandb.project=sync-grpo-h100-gptoss120b-exp \\"
        echo "      logger.wandb.name=GPTOSS120B_pip_cudnn919_8node"
        echo ""
        
        COMMAND=""
        ;;
    
    # ========================================
    # TP=8 8-Node: generation TP=8, training TP=8, PP=1, EP=8 (64 GPUs)
    # Reduces per-GPU activation memory (no PP pipeline buffers)
    # Aligns vLLM TP=8 with Megatron TP=8 → simpler weight refit
    # Reduced samples (16 prompts × 8 gens = 128 total) to avoid OOM
    # NOTE: gpu_memory_utilization=0.40 was too low → KV cache = -3.14 GiB → OOM at vLLM init
    # Use tp8_8node instead (gpu_memory_utilization=0.47)
    # ========================================
    tp8_seqpack_8node)
        NUM_NODES=8
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=8
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS120B_tp8_seqpack_8node"

        echo "[INFO] 120B 8-node: G_TP=8, T_TP=8, T_PP=1, EP=8 — reduced samples (OOM fix)"
        echo "  - Nodes: ${NUM_NODES} (64 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}, DP=1"
        echo "  - Generation: TP=${G_TP}"
        echo "  - Reduced: num_prompts=16, num_generations=8, GBS=64"
        echo "  - gpu_memory_utilization=0.40 (conservative)"
        echo "  - moe_permute_fusion: true, sequence_packing: true"

        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.40 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.moe_permute_fusion=true \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.sequence_packing.enabled=true \
policy.train_global_batch_size=64 \
grpo.num_prompts_per_step=16 \
grpo.num_generations_per_prompt=8 \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # TP=8 PP=1 8-Node FIXED: G_TP=8, T_TP=8, T_PP=1, EP=8 (64 GPUs)
    # Root cause of prior OOM (moe_seqpack_8node, T_PP=2): PP=2 pipeline buffers
    # used ~11 GiB (128 micro-batches × activation memory), leaving only 599 MiB free.
    # Fix: T_PP=1 eliminates pipeline buffers → saves ~11 GiB.
    # Prior tp8_seqpack_8node used gpu_memory_utilization=0.40 → KV cache=-3.14 GiB → OOM.
    # Fix: gpu_memory_utilization=0.47 → budget=37 GiB, model=30 GiB, KV=7 GiB ✓
    # ========================================
    tp8_8node)
        NUM_NODES=8
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=8
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS120B_tp8_8node"

        echo "[INFO] 120B 8-node FIXED: G_TP=8, T_TP=8, T_PP=1, EP=8 (no PP buffers)"
        echo "  - Nodes: ${NUM_NODES} (64 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}, DP=1"
        echo "  - Generation: TP=${G_TP}"
        echo "  - gpu_memory_utilization=0.47 (120B model=30 GiB, KV=7 GiB headroom)"
        echo "  - T_PP=1 removes pipeline buffers (~11 GiB savings vs T_PP=2)"
        echo "  - Reduced batch: num_prompts=16, num_generations=8, GBS=64"
        echo "  - moe_permute_fusion: true, sequence_packing: true"

        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.47 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.moe_permute_fusion=true \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.sequence_packing.enabled=true \
policy.train_global_batch_size=64 \
grpo.num_prompts_per_step=16 \
grpo.num_generations_per_prompt=8 \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # EP=4 8-node: T_TP=4, T_PP=2, T_EP=4, DP=2 (64 GPUs)
    # Fix for weight refit OOM (Bug C):
    #   - EP=8 gather_from_ep_ranks uses ~13.85 GiB temp → only 0.6 GiB left for cat(3.96 GiB)
    #   - EP=4 gathers from 4 ranks instead of 8 → ~6.9 GiB temp → ~7.5 GiB left → cat fits!
    #   - DP=2 (64 / (4×2×4) = 2) → distributed optimizer shards fp32 states across 2 ranks
    #   EP=4 is valid: 8 experts / EP=4 = 2 experts per EP group
    # ========================================
    ep4_seqpack_8node)
        NUM_NODES=8
        G_TP=8
        G_PP=1
        T_EP=4
        T_TP=4
        T_PP=2
        T_CP=1
        WANDB_NAME="GPTOSS120B_ep4_seqpack_8node"

        echo "[INFO] 120B 8-node: T_EP=4, T_TP=4, T_PP=2, DP=2 — weight refit OOM fix"
        echo "  - Nodes: ${NUM_NODES} (64 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}, DP=2"
        echo "  - Generation: TP=${G_TP}"
        echo "  - EP=4 reduces gather_from_ep_ranks from 8x to 4x (~7 GiB vs ~14 GiB)"
        echo "  - DP=2 halves optimizer fp32 states via distributed optimizer"
        echo "  - gpu_memory_utilization=0.47"
        echo "  - moe_permute_fusion: true, sequence_packing: true"

        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.47 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.moe_permute_fusion=true \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.sequence_packing.enabled=true \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # opt_offload_8node: 8-node + optimizer_cpu_offload=true
    # Same as moe_seqpack_8node (T_EP=8, T_TP=4, T_PP=2, DP=1) but with optimizer
    # fp32 states offloaded to CPU → frees ~22+ GiB GPU memory at weight refit time.
    # Targets Bug C Stage 2: gather_from_ep_ranks uses ~13.85 GiB temp leaving only
    # 0.60 GiB for cat(3.96 GiB). optimizer_cpu_offload moves fp32 states to CPU
    # BEFORE weight refit, providing the needed headroom.
    # Constraint: optimizer_offload_fraction must be 1.0 when cpu_offload=true
    # ========================================
    opt_offload_8node)
        NUM_NODES=8
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=4
        T_PP=2
        T_CP=1
        WANDB_NAME="GPTOSS120B_opt_offload_8node"

        echo "[INFO] 120B 8-node: optimizer_cpu_offload=true — Bug C weight refit fix"
        echo "  - Nodes: ${NUM_NODES} (64 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}, DP=1"
        echo "  - Generation: TP=${G_TP}, gpu_memory_utilization=0.47"
        echo "  - optimizer_cpu_offload=true, optimizer_offload_fraction=1.0"
        echo "  - moe_permute_fusion: true, sequence_packing: true"

        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.47 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.moe_permute_fusion=true \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.megatron_cfg.optimizer.optimizer_cpu_offload=true \
policy.megatron_cfg.optimizer.optimizer_offload_fraction=1.0 \
policy.sequence_packing.enabled=true \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # 16-Node (128 GPUs) — TP=8 PP=2 EP=8
    # VALIDATED: job 9867278 — FusedAttention sub-backend 1, is_training=True ✅
    # Fix for Bug 13: 8-node OOM at grad_data alloc (~90GiB). 16-node halves per-GPU load.
    # Megatron TP=8 == vLLM TP=8 → no Bug 11 OOM at prepare_refit_info.
    # ========================================
    tp8_seqpack_16node)
        NUM_NODES=16
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=8
        T_PP=2
        T_CP=1
        WANDB_NAME="GPTOSS120B_tp8_seqpack_16node"

        echo "[INFO] 120B 16-node: T_TP=8, T_PP=2, T_EP=8, vLLM TP=8 — VALIDATED config"
        echo "  - Nodes: ${NUM_NODES} (128 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}"
        echo "  - Generation: TP=${G_TP}, gpu_memory_utilization=0.47, NRL_REFIT_BUFFER_MEMORY_RATIO=0.40"
        echo "  - Megatron TP=vLLM TP=8 → 1:1 weight copy at prepare_refit_info (no Bug 11)"
        echo "  - 16-node fixes Bug 13: ~27 GiB/GPU model vs ~54 GiB on 8-node"
        echo "  - moe_permute_fusion: true, sequence_packing: true, alltoall dispatcher"

        # Bug 14 fix: NRL_REFIT_BUFFER_MEMORY_RATIO must give half-buffer ≥ gate_up_proj_size (3.96 GiB)
        # gate_up_proj aligned = 4246732800 bytes = 3.96 GiB per EP rank.
        # Default ratio=0.3: half = 27.86*0.3/2 = 4.18 GiB > 3.96 GiB ✓ (just barely)
        # Actually default ratio=0.3: half = free(~25GiB)*0.3/2 = 3.75 GiB < 3.96 GiB → AssertionError!
        # Bug 15 fix: Step 2 refit OOM at gather_from_ep_ranks (need 3.96 GiB for torch.cat).
        # Memory forensics (from jobs 9870652, 9870789 with RATIO=0.5):
        #   vLLM sleeping: ~25 GiB | Megatron model+NCCL: ~26.26 GiB → free_at_alloc ≈ 27.86 GiB
        #   buffers(RATIO=0.5) = 27.86*0.5 = 13.93 GiB → free_for_cat = 16.06-13.93 = 2.13 GiB < 3.96 GiB ✗
        # Fix: RATIO=0.40 → buffers = 27.86*0.40 = 11.14 GiB
        #   half = 5.57 GiB > 3.96 GiB ✓ (Bug 14 OK)
        #   free_for_cat = 16.06-11.14 = 4.92 GiB > 3.96 GiB ✓ (Bug 15 fixed, 0.96 GiB margin)
        # gpu_memory_utilization=0.47: (0.43 too low → KV=-0.77GiB; 0.5 → buffers too large)
        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && export NRL_REFIT_BUFFER_MEMORY_RATIO=0.40 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=false uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.47 \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.moe_token_dispatcher_type=alltoall \
policy.megatron_cfg.moe_permute_fusion=true \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.sequence_packing.enabled=true \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=3 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo ""
        echo "Available experiments:"
        echo ""
        echo "  === 2-Node (16 GPUs) - EP=8, TP=2, PP=1, DP=1 ==="
        echo "  optimized_2node        - Basic 2-node (seq_len=2048, reduced batch)"
        echo "  optimized_2node_debug  - 2-node + NVTE_DEBUG (default)"
        echo "  optimized_2node_seqpack- 2-node + sequence packing"
        echo "  optimized_2node_unfused- 2-node + unfused attention"
        echo "  WARNING: 2-node likely OOM (DP=1 → optimizer not distributed)"
        echo ""
        echo "  === 4-Node (32 GPUs) - EP=8, TP=2, PP=1, DP=2 (recommended) ==="
        echo "  optimized_4node        - 4-node with distributed optimizer"
        echo "  optimized_4node_unfused- 4-node + unfused attention"
        echo ""
        echo "  === 8-Node (64 GPUs) - EP=8, TP=4, PP=2 (safe) ==="
        echo "  optimized_8node        - 8-node full config"
        echo "  optimized_8node_unfused- 8-node + unfused attention"
        echo ""
        echo "  === 16-Node (128 GPUs) - TP=8 PP=2 EP=8 (VALIDATED) ==="
        echo "  tp8_seqpack_16node     - 16-node, FusedAttention sub-backend 1 validated"
        echo ""
        echo "  === Interactive ==="
        echo "  interactive            - 2-node Ray cluster only"
        echo ""
        echo "Notes:"
        echo "  - 120B requires TP=8 for vLLM generation (240GB model / 8 = 30GB/GPU)"
        echo "  - 2-node OOMs because DP=1 → Adam optimizer states (~3x model) not distributed"
        echo "  - 4-node (DP=2) is minimum safe config: optimizer sharded across 2 ranks"
        echo "  - 8-node is most comfortable with full seq_len=4096"
        exit 1
        ;;
esac

echo ""

if [[ -z "$COMMAND" ]]; then
    echo "[INFO] INTERACTIVE MODE - No command will be executed"
    echo "[INFO] Submitting Ray cluster only..."
    echo ""
    
    INTERACTIVE_PARTITION="${USE_BATCH_PARTITION:+batch}"
    INTERACTIVE_PARTITION="${INTERACTIVE_PARTITION:-interactive}"
    
    GPU_IDLE_EXEMPTION='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"other","description":"120B model: HF->Megatron weight conversion + vLLM sleep causes long GPU idle periods"}}'
    
    CONTAINER=$CONTAINER \
    CUDNN_INSTALL=${CUDNN_INSTALL:-1} \
    CUDNN_VERSION=${CUDNN_VERSION:-9.18.1} \
    HF_HOME=$HF_HOME \
    HF_DATASETS_CACHE=$HF_DATASETS_CACHE \
    WANDB_API_KEY=$WANDB_API_KEY \
    UV_HTTP_TIMEOUT=300 \
    MOUNTS="$MOUNTS" \
    sbatch \
        --nodes=${NUM_NODES} \
        --account=${account} \
        --job-name=p1962-120b-${EXPERIMENT} \
        --partition=${INTERACTIVE_PARTITION} \
        --time=04:00:00 \
        --comment="${GPU_IDLE_EXEMPTION}" \
        ${GRES_FLAG} \
        ray.sub
    
    echo ""
    echo "[INFO] After job starts, run: bash JOBID-attach.sh"
else
    echo "[INFO] Command:"
    echo "$COMMAND"
    echo ""
    
    GPU_IDLE_EXEMPTION='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"90","reason":"other","description":"Hybrid vLLM-Megatron RL training: vLLM sleep mode causes expected GPU idle during phase transitions"}}'
    
    COMMAND="$COMMAND" \
    CONTAINER=$CONTAINER \
    CUDNN_INSTALL=${CUDNN_INSTALL:-1} \
    CUDNN_VERSION=${CUDNN_VERSION:-9.18.1} \
    HF_HOME=$HF_HOME \
    HF_DATASETS_CACHE=$HF_DATASETS_CACHE \
    WANDB_API_KEY=$WANDB_API_KEY \
    UV_HTTP_TIMEOUT=300 \
    MOUNTS="$MOUNTS" \
    sbatch \
        --nodes=${NUM_NODES} \
        --account=${account} \
        --job-name=p1962-120b-${EXPERIMENT} \
        --partition=batch \
        --time=04:00:00 \
        --comment="${GPU_IDLE_EXEMPTION}" \
        ${GRES_FLAG} \
        ray.sub
fi
