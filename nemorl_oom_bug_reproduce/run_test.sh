#!/bin/bash
# ---------------------------------------------------------
# Script to submit NeMo RL jobs via SLURM
# ---------------------------------------------------------

# ----- REQUIRED ARGUMENTS -----
CONFIG_NAME="$1"
EXP_NAME="${CONFIG_NAME}-$(date +%Y%m%d%H%M%S)"

echo "Experiment name: $EXP_NAME"

# ----- LOGGING -----
LOG_DIR="results/$EXP_NAME"
LOG_FILE="$LOG_DIR/slurm.log"
mkdir -p "$LOG_DIR"
echo "Log file: $LOG_FILE"

# ----- RUN JOB -----
uv run ether1_train.py \
    --config "$CONFIG_NAME" \
    "cluster.num_nodes=1"
