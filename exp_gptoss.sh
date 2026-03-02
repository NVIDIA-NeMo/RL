#!/bin/bash
# GPT-OSS-20B Experiments
# Configurations for comparison:
#   1. yaml_default: YAML settings only (8 nodes)
#   2. benchmark: YAML + benchmark overrides (8 nodes, short run)
#   3. yaml_2node: YAML defaults + 2 nodes ONLY (no optimizations) - baseline test
#   4. yaml_1node: YAML defaults + 1 node ONLY (no optimizations) - baseline test
#   5. optimized_2node: 2 nodes with optimizations (defer_fp32_logits, activation_ckpt)
#   6. optimized_1node: 1 node attempt (aggressive optimizations)
#   7. alt_1node_pp: Alternative 1 node with Pipeline Parallelism
#
# Usage: ./exp_gptoss20b_experiments.sh [experiment_name]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Deactivate conda to avoid Ray version conflicts between host and container
# Container uses Ray 2.49.2, but conda might have a different version
# Remove conda/miniconda paths from PATH unconditionally (even if CONDA_DEFAULT_ENV is unset)
if echo "$PATH" | grep -qE "conda|miniconda"; then
    echo "[INFO] Removing conda/miniconda from PATH to avoid Ray version conflict..."
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -vE "conda|miniconda" | tr '\n' ':' | sed 's/:$//')
fi
# Also unset CONDA_PREFIX to prevent any conda-related behavior
unset CONDA_PREFIX CONDA_DEFAULT_ENV 2>/dev/null || true

# Source cluster configuration
source "${SCRIPT_DIR}/cluster_config.sh"
setup_cluster_config "batch"
export_cluster_config


EXPERIMENT="${1:-benchmark}"
CONFIG_FILE="examples/configs/recipes/llm/grpo-gptoss-20b-8n8g-megatron.yaml"
account=coreai_dlalgo_nemorl
# Force B200 for WandB project name (override auto-detected H100)
WANDB_PROJECT="sync-grpo-b200-gptoss-exp"

# ========================================
# cuDNN Setup for FusedAttention
# ========================================
# This is required because:
# 1. pyproject.toml specifies TE 2.10.0 which is built with cuDNN 9.18.0+
# 2. Runtime needs LD_LIBRARY_PATH to include cuDNN libraries
# 3. CUDNN_HOME is needed for any TE recompilation
CUDNN_VERSION=${CUDNN_VERSION:-9.18.1}
CUDNN_BUILD=${CUDNN_BUILD:-3}
# MAX_JOBS controls parallelism for CUDA kernel compilation (flash-attn, mamba-ssm, etc.)
# Using $(nproc) to use all available CPU cores for faster builds
CUDNN_SETUP="export MAX_JOBS=\$(nproc) && export CUDNN_HOME=/tmp/cudnn-linux-x86_64-${CUDNN_VERSION}.${CUDNN_BUILD}_cuda12-archive && export LD_LIBRARY_PATH=\${CUDNN_HOME}/lib:\${LD_LIBRARY_PATH} && "

# Organize logs by model type
if [[ "$EXPERIMENT" == gptoss120b_* ]]; then
    export BASE_LOG_DIR="${SCRIPT_DIR}/exp_logs/gpt-oss-120b"
else
    export BASE_LOG_DIR="${SCRIPT_DIR}/exp_logs/gpt-oss-20b"
fi
mkdir -p "$BASE_LOG_DIR"

# Function to clear checkpoint to avoid parallelism conflicts
# IMPORTANT: GPT-OSS-20B checkpoint includes parallelism-dependent optimizer state
# in distributed checkpoint files, so we need to delete the entire checkpoint
# clear_checkpoint_for_parallelism() {
#     local CKPT_BASE="${HF_HOME:-$HOME/.cache/huggingface}/nemo_rl"
#     local CKPT_PATH="${CKPT_BASE}/openai/gpt-oss-20b"
#     if [ -d "${CKPT_PATH}" ]; then
#         echo "[INFO] Clearing existing checkpoint to avoid parallelism conflict..."
#         local BACKUP_PATH="${CKPT_PATH}_backup_$(date +%Y%m%d_%H%M%S)"
#         mv "${CKPT_PATH}" "${BACKUP_PATH}"
#         echo "[INFO] Checkpoint backed up to: ${BACKUP_PATH}"
#         echo "[INFO] New checkpoint will be created with current parallelism settings."
#     fi
# }

echo "============================================"
echo "GPT-OSS-20B Experiment: ${EXPERIMENT}"
echo "Cluster: ${CLUSTER_TYPE}, GPUs/Node: ${GPUS_PER_NODE}"
echo "============================================"

case "$EXPERIMENT" in
    # ========================================
    # 1. YAML Default Only (8 nodes, 64 GPUs)
    # Uses YAML settings exactly as defined
    # ========================================
    yaml_default)
        NUM_NODES=8
        G_TP=4  # Override for better H100 performance
        G_PP=1
        WANDB_NAME="GPTOSS20B_yaml_default"
        
        # Clear checkpoint if parallelism changed
        # clear_checkpoint_for_parallelism
        
        echo "[INFO] Running YAML default configuration"
        echo "  - Nodes: ${NUM_NODES}"
        echo "  - YAML defaults: EP=8, TP=4, moe_permute_fusion=true"
        echo "  - defer_fp32_logits: False (default)"
        echo "  - activation_checkpointing: false (default)"
        
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 2. Benchmark Settings (8 nodes, short run)
    # YAML + benchmark overrides for performance testing
    # ========================================
    benchmark)
        NUM_NODES=8
        G_TP=4
        G_PP=1
        WANDB_NAME="GPTOSS20B_benchmark"
        
        # Clear checkpoint if parallelism changed
        # clear_checkpoint_for_parallelism
        
        echo "[INFO] Running benchmark configuration"
        echo "  - Nodes: ${NUM_NODES}"
        echo "  - Short run: max_num_steps=20"
        echo "  - Checkpointing disabled"
        
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 3. YAML Default + 2 Nodes ONLY (Baseline Test)
    # No optimizations - tests if YAML defaults work on 2 nodes
    # YAML: EP=8, TP=4 -> requires 32 GPUs minimum!
    # This will likely FAIL due to parallelism constraints
    # ========================================
    yaml_2node)
        NUM_NODES=2
        G_TP=4  # YAML default
        G_PP=1
        WANDB_NAME="GPTOSS20B_yaml_2node_baseline"
        
        # Clear checkpoint if parallelism changed
        # clear_checkpoint_for_parallelism
        
        echo "[INFO] Running YAML defaults on 2 nodes (BASELINE TEST)"
        echo "  - Nodes: ${NUM_NODES} (16 GPUs)"
        echo "  - Using YAML defaults: EP=8, TP=4"
        echo "  - WARNING: EP*TP = 32 > 16 GPUs, may fail!"
        echo "  - No optimizations applied"
        
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 3b. YAML Default + 2 Nodes + G_TP=2
    # IDENTICAL to exp_gptoss20b.sh
    # Uses YAML defaults for training (EP=8, TP=4)
    # Only overrides: cluster, generation TP=2
    # ========================================
    yaml_2node_gtp2)
        NUM_NODES=2
        G_TP=2  # Same as exp_gptoss20b.sh (YAML default is also 2)
        G_PP=1
        WANDB_NAME="GPTOSS20B_yaml_2node_gtp2"
        
        # Clear checkpoint if parallelism changed
        # clear_checkpoint_for_parallelism
        
        echo "[INFO] Running YAML defaults on 2 nodes with G_TP=2"
        echo "  - Nodes: ${NUM_NODES} (16 GPUs)"
        echo "  - Generation: TP=${G_TP}, PP=${G_PP}"
        echo "  - Training: YAML defaults (EP=8, TP=4)"
        echo "  - IDENTICAL to exp_gptoss20b.sh"
        
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 4. YAML Default + 1 Node ONLY (Baseline Test)
    # No optimizations - tests if YAML defaults work on 1 node
    # YAML: EP=8, TP=4 -> requires 32 GPUs minimum!
    # This will likely FAIL due to parallelism constraints
    # ========================================
    yaml_1node)
        NUM_NODES=1
        G_TP=8  # Max TP for 1 node generation
        G_PP=1
        WANDB_NAME="GPTOSS20B_yaml_1node_baseline"
        
        # Clear checkpoint if parallelism changed
        # clear_checkpoint_for_parallelism
        
        echo "[INFO] Running YAML defaults on 1 node (BASELINE TEST)"
        echo "  - Nodes: ${NUM_NODES} (8 GPUs)"
        echo "  - Using YAML defaults: EP=8, TP=4"
        echo "  - WARNING: EP*TP = 32 > 8 GPUs, will definitely fail!"
        echo "  - No optimizations applied"
        
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 5. Optimized 2-Node (16 GPUs)
    # With memory optimizations for fewer nodes
    # ========================================
    optimized_2node)
        NUM_NODES=2
        G_TP=4
        G_PP=1
        # Training parallelism for 16 GPUs (need to fit EP=8 with TP*PP*EP*CP = world_size/DP)
        # With EP=8, TP=4: 8*4 = 32 > 16, need to adjust
        # Option: EP=4, TP=4 -> 16 GPUs
        T_EP=4
        T_TP=4
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS20B_optimized_2node"
        
        # Clear checkpoint if parallelism changed
        # clear_checkpoint_for_parallelism
        
        echo "[INFO] Running optimized 2-node configuration"
        echo "  - Nodes: ${NUM_NODES} (16 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}"
        echo "  - defer_fp32_logits: true"
        echo "  - activation_checkpointing: true"
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 5b. Optimized 2-Node DEBUG (16 GPUs)
    # Same as optimized_2node but with NVTE_DEBUG enabled
    # Shows FusedAttention backend selection details
    # ========================================
    optimized_2node_debug)
        NUM_NODES=2
        G_TP=4
        G_PP=1
        T_EP=4
        T_TP=4
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS20B_optimized_2node_debug"
        
        echo "[INFO] Running optimized 2-node configuration with NVTE DEBUG"
        echo "  - Nodes: ${NUM_NODES} (16 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}"
        echo "  - Generation: TP=${G_TP}"
        echo "  - NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 (shows FusedAttention backend)"
        echo "  - defer_fp32_logits: true"
        echo "  - activation_checkpointing: true"
        
        # NVTE_DEBUG shows Transformer Engine attention backend selection
        # Note: export is required for Ray workers to inherit these env vars
        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 5c. Optimized 2-Node UNFUSED Attention (16 GPUs)
    # Uses attention_backend=unfused via Megatron config
    # moe_permute_fusion: uses YAML default (true). If Triton "cpu tensor" error appears, set policy.megatron_cfg.moe_permute_fusion=false
    # ========================================
    optimized_2node_unfused)
        NUM_NODES=2
        G_TP=4
        G_PP=1
        T_EP=4
        T_TP=4
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS20B_2node_unfused_attn_kernel"
        
        echo "[INFO] Running optimized 2-node with UNFUSED attention (for comparison)"
        echo "  - Nodes: ${NUM_NODES} (16 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}"
        echo "  - Generation: TP=${G_TP}"
        echo "  - attention_backend=unfused (Megatron config)"
        echo "  - NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 (shows UnfusedAttention)"
        echo "  - defer_fp32_logits: true"
        echo "  - activation_checkpointing: true"
        
        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.attention_backend=unfused \
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
    # 5d. Optimized 2-Node UNFUSED Attention DEBUG (16 GPUs)
    # Same as optimized_2node_unfused but with NVTE_DEBUG enabled
    # Shows unfused attention backend selection details
    # ========================================
    optimized_2node_unfused_debug)
        NUM_NODES=2
        G_TP=4
        G_PP=1
        T_EP=4
        T_TP=4
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS20B_2node_unfused_attn_kernel_debug"
        
        echo "[INFO] Running optimized 2-node with UNFUSED attention + DEBUG (for comparison)"
        echo "  - Nodes: ${NUM_NODES} (16 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}"
        echo "  - Generation: TP=${G_TP}"
        echo "  - attention_backend=unfused (Megatron config)"
        echo "  - NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 (shows UnfusedAttention)"
        echo "  - defer_fp32_logits: true"
        echo "  - activation_checkpointing: true"
        
        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.attention_backend=unfused \
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
    # 4. Optimized 1-Node Attempt (8 GPUs)
    # Aggressive memory optimizations
    # ========================================
    optimized_1node)
        NUM_NODES=1
        G_TP=8  # Max TP for generation
        G_PP=1
        # Training parallelism for 8 GPUs
        # Need to fit MoE (32 experts, 4 active) in 8 GPUs
        # Option: EP=8, TP=1, PP=1 OR EP=4, TP=2, PP=1
        T_EP=8
        T_TP=1
        T_PP=1
        T_CP=1
        TRAIN_GBS=512  # YAML default
        WANDB_NAME="GPTOSS20B_optimized_1node"
        
        # Clear checkpoint if parallelism changed
        # clear_checkpoint_for_parallelism
        
        echo "[INFO] Running optimized 1-node configuration (8 GPUs)"
        echo "  - Nodes: ${NUM_NODES} (8 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}"
        echo "  - Generation: TP=${G_TP}"
        echo "  - defer_fp32_logits: true"
        echo "  - activation_checkpointing: true"
        echo "  - Reduced train_global_batch_size: ${TRAIN_GBS}"
        
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.train_global_batch_size=${TRAIN_GBS} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 4b. Optimized 1-Node DEBUG (8 GPUs)
    # Same as optimized_1node but with NVTE_DEBUG enabled
    # Shows FusedAttention backend selection details
    # ========================================
    optimized_1node_debug)
        NUM_NODES=1
        G_TP=8  # Max TP for generation
        G_PP=1
        T_EP=8
        T_TP=1
        T_PP=1
        T_CP=1
        TRAIN_GBS=512
        WANDB_NAME="GPTOSS20B_optimized_1node_debug"
        
        echo "[INFO] Running optimized 1-node configuration with NVTE DEBUG"
        echo "  - Nodes: ${NUM_NODES} (8 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}"
        echo "  - Generation: TP=${G_TP}"
        echo "  - NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 (shows FusedAttention backend)"
        echo "  - defer_fp32_logits: true"
        echo "  - activation_checkpointing: true"
        
        # NVTE_DEBUG shows Transformer Engine attention backend selection
        # Note: export is required for Ray workers to inherit these env vars
        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=false \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.train_global_batch_size=${TRAIN_GBS} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 5. Alternative 1-Node with PP (8 GPUs)
    # Using Pipeline Parallelism instead of EP
    # ========================================
    alt_1node_pp)
        NUM_NODES=1
        G_TP=4
        G_PP=2  # Use PP for generation too
        # Training: TP=4, PP=2 = 8 GPUs
        T_EP=1  # Disable EP
        T_TP=4
        T_PP=2
        T_CP=1
        TRAIN_GBS=512  # YAML default
        WANDB_NAME="GPTOSS20B_alt_1node_pp"
        
        # Clear checkpoint if parallelism changed
        # clear_checkpoint_for_parallelism
        
        echo "[INFO] Running alternative 1-node with PP (8 GPUs)"
        echo "  - Nodes: ${NUM_NODES} (8 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}"
        echo "  - Generation: TP=${G_TP}, PP=${G_PP}"
        echo "  - MoE routing on single rank (EP=1)"
        
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.megatron_cfg.moe_permute_fusion=false \
policy.train_global_batch_size=${TRAIN_GBS} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 6. Low Memory 1-Node (vLLM gpu_memory_utilization만 감소)
    # 기존 optimized_1node에서 vLLM 메모리만 줄임 (GBS, num_prompts 유지)
    # ========================================
    lowmem_1node)
        NUM_NODES=1
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=1
        T_PP=1
        T_CP=1
        TRAIN_GBS=512  # YAML default
        GPU_MEM_UTIL=0.4  # 0.6 → 0.4로 감소 (이것만 변경!)
        WANDB_NAME="GPTOSS20B_lowmem_1node"
        
        # Clear checkpoint if parallelism changed
        # clear_checkpoint_for_parallelism
        
        echo "[INFO] Running low memory 1-node configuration"
        echo "  - Nodes: ${NUM_NODES} (8 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}"
        echo "  - Generation: TP=${G_TP}"
        echo "  - vLLM gpu_memory_utilization: ${GPU_MEM_UTIL} (changed from 0.6)"
        echo "  - train_global_batch_size: ${TRAIN_GBS} (unchanged)"
        echo "  - num_prompts_per_step: 64 (YAML default, unchanged)"
        
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=${GPU_MEM_UTIL} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=false \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.train_global_batch_size=${TRAIN_GBS} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 7. Questioner's 2-Node Config
    # 질문자가 제시한 설정으로 2노드 실험
    # defer_fp32_logits=true, activation_checkpointing=true
    # precision_aware_optimizer, Generation TP=4
    # ========================================
    questioner_2node)
        NUM_NODES=2
        G_TP=4  # 질문자 설정
        G_PP=1
        # YAML 기본값 사용 (EP=8, TP=4는 16 GPU에 맞지 않으므로 조정)
        # EP*TP <= 16 이어야 함. EP=4, TP=4 = 16 ✓
        T_EP=4
        T_TP=4
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS20B_questioner_2node"
        
        # Clear checkpoint if parallelism changed
        # clear_checkpoint_for_parallelism
        
        echo "[INFO] Running questioner's 2-node configuration"
        echo "  - Nodes: ${NUM_NODES} (16 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}"
        echo "  - Generation: TP=${G_TP}"
        echo "  - defer_fp32_logits: true"
        echo "  - activation_checkpointing: true"
        echo "  - precision_aware_optimizer: true"
        
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.precision=bfloat16 \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.megatron_cfg.optimizer.use_precision_aware_optimizer=true \
policy.megatron_cfg.optimizer.bf16=true \
policy.megatron_cfg.optimizer.fp16=false \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 8. Questioner's 1-Node Config (Aggressive)
    # 질문자 설정 + 1노드에 맞게 parallelism 조정
    # ========================================
    questioner_1node)
        NUM_NODES=1
        G_TP=8  # 1노드에서는 TP=8 최대
        G_PP=1
        # 1노드(8 GPU)에 맞게: EP=8, TP=1 = 8 ✓
        T_EP=8
        T_TP=1
        T_PP=1
        T_CP=1
        TRAIN_GBS=512  # YAML default
        GPU_MEM_UTIL=0.5  # vLLM 메모리 감소
        WANDB_NAME="GPTOSS20B_questioner_1node"
        
        # Clear checkpoint if parallelism changed
        # clear_checkpoint_for_parallelism
        
        echo "[INFO] Running questioner's 1-node configuration"
        echo "  - Nodes: ${NUM_NODES} (8 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}"
        echo "  - Generation: TP=${G_TP}"
        echo "  - defer_fp32_logits: true"
        echo "  - activation_checkpointing: true"
        echo "  - precision_aware_optimizer: true"
        echo "  - train_global_batch_size: ${TRAIN_GBS}"
        echo "  - Rollout GBS: 2048 (64 prompts × 32 generations)"
        echo "  - vLLM gpu_memory_utilization: ${GPU_MEM_UTIL}"
        
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.precision=bfloat16 \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.generation.vllm_cfg.gpu_memory_utilization=${GPU_MEM_UTIL} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.megatron_cfg.optimizer.use_precision_aware_optimizer=true \
policy.megatron_cfg.optimizer.bf16=true \
policy.megatron_cfg.optimizer.fp16=false \
policy.train_global_batch_size=${TRAIN_GBS} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # 9. Qwen30B Comparable (2 Nodes)
    # Matches Qwen30B 2node parallelisms exactly
    # Training: TP=4, PP=2, EP=2
    # Generation: TP=4 (to match typical Qwen setups if needed, or 2 for H100 optimal)
    # ========================================
    qwen_comp_2node)
        NUM_NODES=2
        G_TP=4
        G_PP=1
        
        # Qwen30B 2node settings: TP=4, PP=2, EP=2
        T_EP=2
        T_TP=4
        T_PP=2
        T_CP=1
        
        WANDB_NAME="GPTOSS20B_qwen_comp_2node"
        
        # Clear checkpoint if parallelism changed
        # clear_checkpoint_for_parallelism
        
        echo "[INFO] Running Qwen30B Comparable 2-node configuration"
        echo "  - Nodes: ${NUM_NODES} (16 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}"
        echo "  - Generation: TP=${G_TP}, PP=${G_PP}"
        
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # INTERACTIVE MODE - 2 Nodes
    # Launches Ray cluster only, no command
    # Use: ./exp_gptoss20b_experiments.sh interactive_2node
    # Then: bash JOBID-attach.sh
    # Then: uv run ./examples/run_grpo.py --config ...
    # ========================================
    interactive_2node)
        NUM_NODES=2
        WANDB_NAME="GPTOSS20B_interactive_2node"
        
        echo "[INFO] Launching INTERACTIVE 2-node Ray cluster"
        echo "  - Nodes: ${NUM_NODES} (16 GPUs)"
        echo "  - No command will be executed automatically"
        echo ""
        echo "  After job starts:"
        echo "    1. Run: bash JOBID-attach.sh"
        echo "    2. On head node, set cuDNN environment first:"
        echo ""
        echo "    export CUDNN_HOME=/tmp/cudnn-linux-x86_64-9.18.0.65"
        echo "    export LD_LIBRARY_PATH=\${CUDNN_HOME}/lib:\${LD_LIBRARY_PATH}"
        echo ""
        echo "    3. Then run your command:"
        echo ""
        echo "    uv run ./examples/run_grpo.py \\"
        echo "      --config examples/configs/recipes/llm/grpo-gptoss-20b-8n8g-megatron.yaml \\"
        echo "      cluster.num_nodes=2 cluster.gpus_per_node=8 \\"
        echo "      policy.generation.vllm_cfg.tensor_parallel_size=4 \\"
        echo "      policy.megatron_cfg.tensor_model_parallel_size=4 \\"
        echo "      policy.megatron_cfg.expert_model_parallel_size=4 \\"
        echo "      policy.megatron_cfg.defer_fp32_logits=true \\"
        echo "      policy.megatron_cfg.activation_checkpointing=true \\"
        echo "      grpo.max_num_steps=20"
        echo ""
        
        # No COMMAND - this triggers interactive mode
        COMMAND=""
        ;;
    
    # ========================================
    # INTERACTIVE MODE - 1 Node
    # Launches Ray cluster only, no command
    # ========================================
    interactive_1node)
        NUM_NODES=1
        WANDB_NAME="GPTOSS20B_interactive_1node"
        
        echo "[INFO] Launching INTERACTIVE 1-node Ray cluster"
        echo "  - Nodes: ${NUM_NODES} (8 GPUs)"
        echo "  - No command will be executed automatically"
        echo ""
        echo "  After job starts:"
        echo "    1. Run: bash JOBID-attach.sh"
        echo "    2. On head node, set cuDNN environment first:"
        echo ""
        echo "    export CUDNN_HOME=/tmp/cudnn-linux-x86_64-9.18.0.65"
        echo "    export LD_LIBRARY_PATH=\${CUDNN_HOME}/lib:\${LD_LIBRARY_PATH}"
        echo ""
        echo "    3. Then run your command:"
        echo ""
        echo "    uv run ./examples/run_grpo.py \\"
        echo "      --config examples/configs/recipes/llm/grpo-gptoss-20b-8n8g-megatron.yaml \\"
        echo "      cluster.num_nodes=1 cluster.gpus_per_node=8 \\"
        echo "      policy.generation.vllm_cfg.tensor_parallel_size=8 \\"
        echo "      policy.megatron_cfg.tensor_model_parallel_size=1 \\"
        echo "      policy.megatron_cfg.expert_model_parallel_size=8 \\"
        echo "      policy.megatron_cfg.sequence_parallel=true \\"
        echo "      policy.megatron_cfg.defer_fp32_logits=true \\"
        echo "      policy.megatron_cfg.activation_checkpointing=true \\"
        echo "      policy.train_global_batch_size=512 \\"
        echo "      grpo.max_num_steps=20"
        echo ""
        
        # No COMMAND - this triggers interactive mode
        COMMAND=""
        ;;
    
    # ========================================
    # INTERACTIVE MODE - 8 Nodes
    # Launches Ray cluster only, no command
    # ========================================
    interactive_8node)
        NUM_NODES=8
        WANDB_NAME="GPTOSS20B_interactive_8node"
        
        echo "[INFO] Launching INTERACTIVE 8-node Ray cluster"
        echo "  - Nodes: ${NUM_NODES} (64 GPUs)"
        echo "  - No command will be executed automatically"
        echo ""
        echo "  After job starts:"
        echo "    1. Run: bash JOBID-attach.sh"
        echo "    2. On head node, set cuDNN environment first:"
        echo ""
        echo "    export CUDNN_HOME=/tmp/cudnn-linux-x86_64-9.18.0.65"
        echo "    export LD_LIBRARY_PATH=\${CUDNN_HOME}/lib:\${LD_LIBRARY_PATH}"
        echo ""
        echo "    3. Then run your uv run command"
        echo ""
        
        # No COMMAND - this triggers interactive mode
        COMMAND=""
        ;;
    
    # ========================================
    # 10. Qwen30B Comparable (8 Nodes)
    # Matches Qwen30B 8node parallelisms exactly
    # Training: TP=4, PP=4, EP=4
    # Generation: TP=4
    # ========================================
    qwen_comp_8node)
        NUM_NODES=8
        G_TP=4
        G_PP=1
        
        # Qwen30B 8node settings: TP=4, PP=4, EP=4
        T_EP=4
        T_TP=4
        T_PP=4
        T_CP=1
        
        WANDB_NAME="GPTOSS20B_qwen_comp_8node"
        
        # Clear checkpoint if parallelism changed
        # clear_checkpoint_for_parallelism
        
        echo "[INFO] Running Qwen30B Comparable 8-node configuration"
        echo "  - Nodes: ${NUM_NODES} (64 GPUs)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}"
        echo "  - Generation: TP=${G_TP}, PP=${G_PP}"
        
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;
    
    # ========================================
    # GPT-OSS-120B: 2 Nodes (16 B200 GPUs, 192GB each)
    # 117B total params, 128 experts, 4 active, 36 layers
    # Reduced batch/seq for memory: prompts=16, gens=8, seq=2048
    # Training: EP=8, TP=2 (8×2=16, no DP)
    # Generation: vLLM TP=8
    # ========================================
    gptoss120b_2node)
        NUM_NODES=2
        CONFIG_FILE="examples/configs/recipes/llm/grpo-gptoss-120b-2n8g-megatron.yaml"
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=2
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS120B_2node"
        
        echo "[INFO] Running GPT-OSS-120B on 2 nodes (16 B200 GPUs)"
        echo "  - Model: openai/gpt-oss-120b (117B params, 128 experts)"
        echo "  - Nodes: ${NUM_NODES} (16 GPUs × 192GB = 3072GB total)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP} (no DP)"
        echo "  - Generation: vLLM TP=${G_TP}"
        echo "  - Reduced size: prompts=16, gens=8, seq=2048, GBS=128"
        echo "  - Memory opts: activation_ckpt, defer_fp32_logits"
        
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # GPT-OSS-120B: 2 Nodes DEBUG
    # Same as gptoss120b_2node + NVTE_DEBUG
    # ========================================
    gptoss120b_2node_debug)
        NUM_NODES=2
        CONFIG_FILE="examples/configs/recipes/llm/grpo-gptoss-120b-2n8g-megatron.yaml"
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=2
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS120B_2node_debug"
        
        echo "[INFO] Running GPT-OSS-120B on 2 nodes with NVTE DEBUG"
        echo "  - Model: openai/gpt-oss-120b (117B params, 128 experts)"
        echo "  - Nodes: ${NUM_NODES} (16 GPUs × 192GB)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}"
        echo "  - Generation: vLLM TP=${G_TP}"
        echo "  - NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2"
        
        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # GPT-OSS-120B: FP8 2 Nodes (more memory efficient)
    # Uses FP8 training for ~2× memory savings on weights/grads
    # ========================================
    gptoss120b_2node_fp8)
        NUM_NODES=2
        CONFIG_FILE="examples/configs/recipes/llm/grpo-gptoss-120b-2n8g-megatron.yaml"
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=2
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS120B_2node_fp8"
        
        echo "[INFO] Running GPT-OSS-120B on 2 nodes with FP8"
        echo "  - Model: openai/gpt-oss-120b (117B params, 128 experts)"
        echo "  - Nodes: ${NUM_NODES} (16 GPUs × 192GB)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, FP8 enabled"
        echo "  - Generation: vLLM TP=${G_TP}"
        
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.fp8_cfg.enabled=true \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # GPT-OSS-120B: 4 Nodes (32 B200 GPUs, 192GB each)
    # EP=8, TP=2 → DP=2, distributed optimizer works!
    # No CPU offload needed (~127GB/192GB per GPU)
    # ========================================
    gptoss120b_4node)
        NUM_NODES=4
        CONFIG_FILE="examples/configs/recipes/llm/grpo-gptoss-120b-4n8g-megatron.yaml"
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=2
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS120B_4node"
        
        echo "[INFO] Running GPT-OSS-120B on 4 nodes (32 B200 GPUs)"
        echo "  - Model: openai/gpt-oss-120b (117B params, 128 experts)"
        echo "  - Nodes: ${NUM_NODES} (32 GPUs × 192GB = 6144GB total)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, DP=2"
        echo "  - Generation: vLLM TP=${G_TP}"
        echo "  - No optimizer CPU offload needed (DP=2 distributes optimizer)"
        echo "  - 512 total samples (64 prompts × 8 gen), 10 steps"
        
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
grpo.num_prompts_per_step=64 \
grpo.num_generations_per_prompt=8 \
policy.train_global_batch_size=512 \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=10 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # GPT-OSS-120B: 8 Nodes (64 B200 GPUs)
    # EP=8, TP=4, PP=2 → 64 GPUs, grpo 64 prompts × 32 generations
    # ========================================
    gptoss120b_8node)
        NUM_NODES=8
        CONFIG_FILE="examples/configs/recipes/llm/grpo-gptoss-120b-8n8g-megatron.yaml"
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=4
        T_PP=2
        T_CP=1
        WANDB_NAME="GPTOSS120B_8node"
        
        echo "[INFO] Running GPT-OSS-120B on 8 nodes (64 B200 GPUs)"
        echo "  - Model: openai/gpt-oss-120b (117B params, 128 experts)"
        echo "  - Nodes: ${NUM_NODES} (64 GPUs × 192GB)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, PP=${T_PP}"
        echo "  - Generation: vLLM TP=${G_TP}"
        echo "  - grpo: 64 prompts/step, 32 generations/prompt"
        
        COMMAND="${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # GPT-OSS-120B: 4 Nodes UNFUSED Attention
    # Same as gptoss120b_4node but with unfused attention backend
    # For fused vs unfused attention kernel performance comparison
    # ========================================
    gptoss120b_4node_unfused)
        NUM_NODES=4
        CONFIG_FILE="examples/configs/recipes/llm/grpo-gptoss-120b-4n8g-megatron.yaml"
        G_TP=8
        G_PP=1
        T_EP=8
        T_TP=2
        T_PP=1
        T_CP=1
        WANDB_NAME="GPTOSS120B_4node_unfused_attn_kernel"
        
        echo "[INFO] Running GPT-OSS-120B on 4 nodes with UNFUSED attention (for comparison)"
        echo "  - Model: openai/gpt-oss-120b (117B params, 128 experts)"
        echo "  - Nodes: ${NUM_NODES} (32 GPUs × 192GB)"
        echo "  - Training: EP=${T_EP}, TP=${T_TP}, DP=2"
        echo "  - Generation: vLLM TP=${G_TP}"
        echo "  - attention_backend=unfused (TE UnfusedDotProductAttention)"
        echo "  - NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 (shows UnfusedAttention)"
        echo "  - defer_fp32_logits: true"
        echo "  - activation_checkpointing: true"
        echo "  - 512 total samples (64 prompts × 8 gen), 10 steps"
        
        COMMAND="export NVTE_DEBUG=1 && export NVTE_DEBUG_LEVEL=2 && ${CUDNN_SETUP}NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${G_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=${G_PP} \
policy.megatron_cfg.tensor_model_parallel_size=${T_TP} \
policy.megatron_cfg.expert_model_parallel_size=${T_EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${T_PP} \
policy.megatron_cfg.context_parallel_size=${T_CP} \
policy.megatron_cfg.attention_backend=unfused \
policy.megatron_cfg.defer_fp32_logits=true \
policy.megatron_cfg.activation_checkpointing=true \
policy.sequence_packing.enabled=false \
grpo.num_prompts_per_step=64 \
grpo.num_generations_per_prompt=8 \
policy.train_global_batch_size=512 \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=10 \
logger.wandb_enabled=True \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'"
        ;;

    # ========================================
    # GPT-OSS-120B: INTERACTIVE 4 Nodes
    # Ray cluster only, no auto command
    # ========================================
    gptoss120b_interactive_4node)
        NUM_NODES=4
        WANDB_NAME="GPTOSS120B_interactive_4node"
        
        echo "[INFO] Launching INTERACTIVE 4-node Ray cluster for GPT-OSS-120B"
        echo "  - Model: openai/gpt-oss-120b (117B params, 128 experts)"
        echo "  - Nodes: ${NUM_NODES} (32 B200 GPUs × 192GB)"
        echo "  - Training: EP=8, TP=2, DP=2 (no optimizer CPU offload needed)"
        echo "  - No command will be executed automatically"
        echo ""
        echo "  After job starts:"
        echo "    1. Run: bash JOBID-attach.sh"
        echo "    2. Then run your command:"
        echo ""
        echo "    uv run ./examples/run_grpo.py \\"
        echo "      --config examples/configs/recipes/llm/grpo-gptoss-120b-4n8g-megatron.yaml \\"
        echo "      cluster.num_nodes=4 cluster.gpus_per_node=8 \\"
        echo "      policy.generation.vllm_cfg.tensor_parallel_size=8 \\"
        echo "      policy.megatron_cfg.tensor_model_parallel_size=2 \\"
        echo "      policy.megatron_cfg.expert_model_parallel_size=8 \\"
        echo "      grpo.max_num_steps=3 \\"
        echo "      checkpointing.enabled=false"
        echo ""
        
        COMMAND=""
        ;;

    # ========================================
    # GPT-OSS-120B: INTERACTIVE 8 Nodes
    # Ray cluster only, no auto command
    # ========================================
    gptoss120b_interactive_8node)
        NUM_NODES=8
        WANDB_NAME="GPTOSS120B_interactive_8node"
        
        echo "[INFO] Launching INTERACTIVE 8-node Ray cluster for GPT-OSS-120B"
        echo "  - Model: openai/gpt-oss-120b (117B params, 128 experts)"
        echo "  - Nodes: ${NUM_NODES} (64 B200 GPUs × 192GB)"
        echo "  - Training: EP=8, TP=4, PP=2 (use grpo-gptoss-120b-8n8g-megatron.yaml)"
        echo "  - No command will be executed automatically"
        echo ""
        echo "  After job starts:"
        echo "    1. Run: bash JOBID-attach.sh"
        echo "    2. Then run your command (e.g. 8-node 120B with grpo 64×32):"
        echo ""
        echo "    uv run ./examples/run_grpo.py \\"
        echo "      --config examples/configs/recipes/llm/grpo-gptoss-120b-8n8g-megatron.yaml \\"
        echo "      cluster.num_nodes=8 cluster.gpus_per_node=8 \\"
        echo "      policy.generation.vllm_cfg.tensor_parallel_size=8 \\"
        echo "      policy.megatron_cfg.tensor_model_parallel_size=4 \\"
        echo "      policy.megatron_cfg.expert_model_parallel_size=8 \\"
        echo "      policy.megatron_cfg.pipeline_model_parallel_size=2 \\"
        echo "      grpo.max_num_steps=3 \\"
        echo "      checkpointing.enabled=false"
        echo ""
        
        COMMAND=""
        ;;

    # ========================================
    # GPT-OSS-120B: INTERACTIVE 2 Nodes
    # Ray cluster only, no auto command
    # ========================================
    gptoss120b_interactive_2node)
        NUM_NODES=2
        WANDB_NAME="GPTOSS120B_interactive_2node"
        
        echo "[INFO] Launching INTERACTIVE 2-node Ray cluster for GPT-OSS-120B"
        echo "  - Model: openai/gpt-oss-120b (117B params, 128 experts)"
        echo "  - Nodes: ${NUM_NODES} (16 B200 GPUs × 192GB)"
        echo "  - No command will be executed automatically"
        echo ""
        echo "  After job starts:"
        echo "    1. Run: bash JOBID-attach.sh"
        echo "    2. On head node, set cuDNN environment first:"
        echo ""
        echo "    export CUDNN_HOME=/tmp/cudnn-linux-x86_64-9.18.0.65"
        echo "    export LD_LIBRARY_PATH=\${CUDNN_HOME}/lib:\${LD_LIBRARY_PATH}"
        echo ""
        echo "    3. Then run your command:"
        echo ""
        echo "    uv run ./examples/run_grpo.py \\"
        echo "      --config examples/configs/recipes/llm/grpo-gptoss-120b-2n8g-megatron.yaml \\"
        echo "      cluster.num_nodes=2 cluster.gpus_per_node=8 \\"
        echo "      policy.generation.vllm_cfg.tensor_parallel_size=8 \\"
        echo "      policy.megatron_cfg.tensor_model_parallel_size=2 \\"
        echo "      policy.megatron_cfg.expert_model_parallel_size=8 \\"
        echo "      grpo.max_num_steps=20"
        echo ""
        
        COMMAND=""
        ;;

    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo ""
        echo "Available experiments:"
        echo ""
        echo "  === GPT-OSS-20B (Full Scale, 8 nodes) ==="
        echo "  yaml_default     - YAML settings only (8 nodes, 64 GPUs)"
        echo "  benchmark        - YAML + benchmark overrides (8 nodes, short run)"
        echo ""
        echo "  === GPT-OSS-20B (Baseline Tests) ==="
        echo "  yaml_2node       - YAML defaults + 2 nodes ONLY (likely fails: EP*TP=32>16)"
        echo "  yaml_2node_gtp2  - YAML defaults + 2 nodes + G_TP=2 (IDENTICAL to exp_gptoss20b.sh)"
        echo "  yaml_1node       - YAML defaults + 1 node ONLY (will fail: EP*TP=32>8)"
        echo ""
        echo "  === GPT-OSS-20B (Optimized for Fewer Nodes) ==="
        echo "  optimized_2node              - 2 nodes with adjusted parallelism (EP=4, TP=4)"
        echo "  optimized_2node_debug        - 2 nodes + NVTE_DEBUG (shows FusedAttention backend)"
        echo "  optimized_2node_unfused      - 2 nodes + UNFUSED Attention (for performance comparison)"
        echo "  optimized_2node_unfused_debug - 2 nodes + UNFUSED + DEBUG (verify unfused backend)"
        echo "  optimized_1node              - 1 node with aggressive optimizations (EP=8, TP=1)"
        echo "  optimized_1node_debug - 1 node + NVTE_DEBUG (shows FusedAttention backend)"
        echo "  alt_1node_pp          - Alternative 1 node with Pipeline Parallelism (EP=1, TP=4, PP=2)"
        echo "  lowmem_1node          - 1 node with reduced vLLM memory (gpu_mem_util=0.4)"
        echo ""
        echo "  === GPT-OSS-20B (Other Configs) ==="
        echo "  questioner_2node - Questioner's 2-node setup (EP=4, TP=4, Generation TP=4)"
        echo "  questioner_1node - Questioner's 1-node setup (EP=8, TP=1, reduced memory)"
        echo "  qwen_comp_2node  - 2 nodes, Match Qwen 2node (TP=4, PP=2, EP=2)"
        echo "  qwen_comp_8node  - 8 nodes, Match Qwen 8node (TP=4, PP=4, EP=4)"
        echo ""
        echo "  === GPT-OSS-120B (4 nodes, 32 B200 GPUs - recommended) ==="
        echo "  gptoss120b_4node       - 120B, EP=8, TP=2, DP=2 (no CPU offload needed)"
        echo "  gptoss120b_4node_unfused - 120B + UNFUSED Attention (for fused vs unfused comparison)"
        echo "  gptoss120b_interactive_4node - 120B interactive 4-node"
        echo ""
        echo "  === GPT-OSS-120B (8 nodes, 64 B200 GPUs) ==="
        echo "  gptoss120b_8node             - 120B, EP=8, TP=4, PP=2, grpo 64×32"
        echo "  gptoss120b_interactive_8node - 120B interactive 8-node (run: bash JOBID-attach.sh)"
        echo ""
        echo "  === GPT-OSS-120B (2 nodes, 16 B200 GPUs - requires CPU offload) ==="
        echo "  gptoss120b_2node       - 120B model, EP=8, TP=2, optimizer CPU offload"
        echo "  gptoss120b_2node_debug - 120B + NVTE_DEBUG"
        echo "  gptoss120b_2node_fp8   - 120B + FP8 training"
        echo "  gptoss120b_interactive_2node - 120B interactive mode"
        echo ""
        echo "  === Interactive Mode (Ray cluster only, no auto command) ==="
        echo "  interactive_1node - 1 node interactive (run: bash JOBID-attach.sh)"
        echo "  interactive_2node - 2 nodes interactive"
        echo "  interactive_8node - 8 nodes interactive"
        exit 1
        ;;
esac

echo ""

# Check if interactive mode (no COMMAND)
if [[ -z "$COMMAND" ]]; then
    echo "[INFO] INTERACTIVE MODE - No command will be executed"
    echo "[INFO] Submitting Ray cluster only..."
    echo ""
    
    CONTAINER=$CONTAINER \
    CUDNN_INSTALL=${CUDNN_INSTALL:-1} \
    CUDNN_VERSION=${CUDNN_VERSION:-9.18.0} \
    HF_HOME=$HF_HOME \
    HF_DATASETS_CACHE=$HF_DATASETS_CACHE \
    WANDB_API_KEY=$WANDB_API_KEY \
    UV_HTTP_TIMEOUT=300 \
    MOUNTS="$MOUNTS" \
    sbatch \
        --nodes=${NUM_NODES} \
        --account=${account} \
        --job-name=gptoss-${EXPERIMENT} \
        --partition=${PARTITION:-batch} \
        --time=${TIME:-04:00:00} \
        ${GRES_FLAG} \
        ray-lbd.sub
    
    echo ""
    echo "[INFO] After job starts, run: bash JOBID-attach.sh"
    echo "[INFO] Then run your uv run command on the head node"
else
    echo "[INFO] Command:"
    echo "$COMMAND"
    echo ""
    
    # GPU Idle Exemption for vLLM-Megatron hybrid RL training
    # vLLM enters sleep mode during policy training, causing expected GPU idle periods
    # Note: 90 mins to cover training phase transitions (observed ~85min idle in practice)
    GPU_IDLE_EXEMPTION='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"90","reason":"other","description":"Hybrid vLLM-Megatron RL training: vLLM sleep mode causes expected GPU idle during phase transitions"}}'
    
    COMMAND="$COMMAND" \
    CONTAINER=$CONTAINER \
    CUDNN_INSTALL=${CUDNN_INSTALL:-1} \
    CUDNN_VERSION=${CUDNN_VERSION:-9.18.0} \
    HF_HOME=$HF_HOME \
    HF_DATASETS_CACHE=$HF_DATASETS_CACHE \
    WANDB_API_KEY=$WANDB_API_KEY \
    UV_HTTP_TIMEOUT=300 \
    MOUNTS="$MOUNTS" \
    sbatch \
        --nodes=${NUM_NODES} \
        --account=${account} \
        --job-name=gptoss-${EXPERIMENT} \
        --partition=batch \
        --time=04:00:00 \
        --comment="${GPU_IDLE_EXEMPTION}" \
        ${GRES_FLAG} \
        ray-lbd.sub
fi


