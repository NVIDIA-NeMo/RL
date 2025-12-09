#!/bin/bash
# ============================================
# Cluster Configuration Auto-Detection
# Supports: GB200 (4 GPUs/node), H100 (8 GPUs/node)
# ============================================

# Detect GPUs per node from SLURM GRES configuration
detect_gpus_per_node() {
    local partition="${1:-batch}"
    
    # Try to get GPU count from SLURM GRES configuration
    local gres_gpus=$(sinfo -p "$partition" -h -o "%G" 2>/dev/null | grep -oP 'gpu(:[^:]+)?:(\d+)' | grep -oP '\d+$' | head -1 || true)
    
    if [[ -n "$gres_gpus" && "$gres_gpus" -gt 0 ]]; then
        echo "$gres_gpus"
    else
        # Fallback: default to 8 (H100)
        echo "8"
    fi
}

# Auto-detect cluster type and set configuration
setup_cluster_config() {
    local partition="${1:-batch}"
    
    # Detect GPUs per node
    if [[ -z "${GPUS_PER_NODE:-}" ]]; then
        GPUS_PER_NODE=$(detect_gpus_per_node "$partition")
    fi
    
    # Set container path based on GPU count
    if [[ -z "${CONTAINER:-}" ]]; then
        if [[ "$GPUS_PER_NODE" -eq 8 ]]; then
            # H100 cluster
            CONTAINER="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/RL/nemo_rl_v0.4.sqsh"
            CLUSTER_TYPE="H100"
        else
            # GB200 cluster (4 GPUs per node)
            CONTAINER="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/nemo_rl.sqsh"
            CLUSTER_TYPE="GB200"
        fi
    fi
    
    # Set GRES flag
    GRES_FLAG="--gres=gpu:${GPUS_PER_NODE}"
    
    # Common paths
    HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/cache}"
    MOUNTS="${MOUNTS:-/lustre:/lustre}"
    
    # Print detected configuration
    echo "[INFO] Cluster Configuration:"
    echo "  - Cluster Type: ${CLUSTER_TYPE:-Unknown}"
    echo "  - GPUs per Node: $GPUS_PER_NODE"
    echo "  - Container: $CONTAINER"
    echo "  - GRES Flag: $GRES_FLAG"
}

# Export variables for use in scripts
export_cluster_config() {
    export GPUS_PER_NODE
    export CONTAINER
    export GRES_FLAG
    export CLUSTER_TYPE
    export HF_HOME
    export HF_DATASETS_CACHE
    export MOUNTS
}

# Calculate number of nodes needed for a given total GPU count
calc_nodes() {
    local total_gpus=$1
    echo $(( (total_gpus + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
}

# Adjust parallelism settings for different GPU counts
# For H100 (8 GPUs): can use higher TP within a node
# For GB200 (4 GPUs): TP limited to 4 within a node
adjust_parallelism() {
    local requested_tp=$1
    local max_tp_per_node=$GPUS_PER_NODE
    
    if [[ $requested_tp -gt $max_tp_per_node ]]; then
        echo "[WARNING] Requested TP=$requested_tp exceeds GPUs per node ($max_tp_per_node)"
    fi
}

