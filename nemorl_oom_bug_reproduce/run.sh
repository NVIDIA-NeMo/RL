#!/bin/bash
# ---------------------------------------------------------
# Script to submit NeMo RL jobs via SLURM
# ---------------------------------------------------------

CONFIG_NAME="$1"
NUM_ACTOR_NODES="$2"
EXP_NAME="$3"

if [ -z "$CONFIG_NAME" ] || [ -z "$NUM_ACTOR_NODES" ] || [ -z "$EXP_NAME" ]; then
    echo "Usage: $0 <config_name> <num_actor_nodes> <exp_name>"
    exit 1
fi

echo "Experiment name: $EXP_NAME"

LOG_DIR="$EXP_NAME"
LOG_FILE="$LOG_DIR/slurm.log"
mkdir -p "$LOG_DIR"
echo "Log file: $LOG_FILE"

# ----- CONSTANTS -----
# NRL_FORCE_REBUILD_VENVS=true \
# NCCL_DEBUG=INFO \

read -r -d '' COMMAND <<EOF

HF_HOME=/data/$USER/cache/huggingface/hub \
NRL_VLLM_ASYNC_TIMEOUT_SECONDS=3600 \
NCCL_PROTO=simple \
NCCL_NVLS_ENABLE=0 \
NCCL_BUFFSIZE=33554432 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
NCCL_SHM_DISABLE=1 \
uv run python ether1_train.py --config $CONFIG_NAME \
    cluster.num_nodes=$NUM_ACTOR_NODES \
    logger.wandb_enabled=True \
    logger.wandb.name=$(basename "$EXP_NAME") \
    checkpointing.checkpoint_dir=$EXP_NAME \
    grpo.val_at_start=false
EOF

# ----- RUN JOB -----

COMMAND=$COMMAND \
sbatch \
    --nodes=$NUM_ACTOR_NODES \
    --partition=slurm01 \
    --time=24:0:0 \
    --job-name=$(basename "$EXP_NAME") \
    ray_nocontainer.sub

