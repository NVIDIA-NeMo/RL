#!/bin/bash
#SBATCH --job-name=nemo-rl-train
#SBATCH --nodes=2
#SBATCH --qos=high
#SBATCH --partition=hopper-prod
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=88
#SBATCH --time=24:00:00
#SBATCH --output=logs/nemo_rl_train_%j.out
#SBATCH --exclusive

set -e -x


# ==============================================================================
# Configuration - Customize these variables
# ==============================================================================

# Container configuration
export CONTAINER="/fsx/edward/docker_images/nemo-rl.sqsh"

# Mount points for container
# Overlay /opt/nemo-rl with local development copy for live code updates
export MOUNTS="/fsx:/fsx,/scratch:/scratch,/fsx/edward/work/scale-rl/nemo-rl:/opt/nemo-rl"

# NeMo-RL configuration
CONFIG_FILE="${CONFIG_FILE:-/fsx/edward/work/scale-rl/nemo-rl/container/configs/grpo-qwen3-4b-thinking-2n8g-fsdp2tp1-noncolocated-fast.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/fsx/edward/checkpoints/nemo-rl/grpo_test_fast}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"

# Optional: Set HF_TOKEN before running if using gated models
# export HF_TOKEN="your_token_here"

# Optional: Ray configuration
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export CPUS_PER_NODE=${CPUS_PER_NODE:-88}  # CPUs per node (default 88 for H100 nodes)
# export RAY_LOG_SYNC_FREQUENCY=300  # Sync Ray logs every 5 minutes

# Enhanced Ray Logging for Debugging
export RAY_BACKEND_LOG_LEVEL=${RAY_BACKEND_LOG_LEVEL:-info}     # C++ backend logging: debug, info, warning, error
export RAY_LOG_TO_STDERR=${RAY_LOG_TO_STDERR:-0}                # Set to 1 to log Python side to stderr
export RAY_logging_level=${RAY_logging_level:-INFO}             # Python logging: DEBUG, INFO, WARNING, ERROR
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}           # vLLM logging: DEBUG, INFO, WARNING, ERROR

# Override Dockerfile's NEMO_RL_VENV_DIR to use persistent /fsx storage
# (Dockerfile sets it to /opt/ray_venvs which gets deleted on container restart)
export NEMO_RL_VENV_DIR="/fsx/edward/.cache/nemo-rl/ray_venvs"

# Path to ray.sub (using patched version with CPU allocation fix)
RAY_SUB_SCRIPT="/fsx/edward/work/scale-rl/nemo-rl/container/ray_patched.sub"

# ==============================================================================
# Build command to run inside Ray cluster
# ==============================================================================

export COMMAND="cd /opt/nemo-rl && uv run python examples/run_grpo_math.py \
  --config ${CONFIG_FILE} \
  checkpointing.checkpoint_dir='${CHECKPOINT_DIR}' \
  logger.wandb_enabled=${WANDB_ENABLED}"

# ==============================================================================
# Dual Mode: Run on existing allocation OR submit new job
# ==============================================================================

if [ -n "$1" ]; then
    # =========================================================================
    # MODE 1: Run on existing job allocation using ray.sub
    # =========================================================================
    JOBID="$1"

    # Query job info
    JOB_INFO=$(scontrol show job -o "$JOBID")

    # Extract job details
    SLURM_NNODES=$(echo "$JOB_INFO" | tr ' ' '\n' | awk -F= '$1=="NumNodes"{print $2}')
    SLURM_GPUS_PER_NODE=$(echo "$JOB_INFO" | grep -oP 'TresPerNode=gres:gpu:\K\d+' || echo "8")
    SLURM_NODELIST=$(echo "$JOB_INFO" | grep -oP '\bNodeList=\K[^ ]+' | head -1)
    SLURM_ACCOUNT=$(echo "$JOB_INFO" | tr ' ' '\n' | awk -F= '$1=="Account"{print $2}')
    SLURM_PARTITION=$(echo "$JOB_INFO" | tr ' ' '\n' | awk -F= '$1=="Partition"{print $2}')

    # Create timestamped log directory
    DATETIME=$(date +%Y%m%d_%H%M%S)
    RAY_LOG_DIR="${JOBID}-${DATETIME}-logs"
    LOGFILE="logs/nemo_rl_train_${DATETIME}_${JOBID}.log"
    mkdir -p $(dirname "$LOGFILE")

    echo "=========================================" | tee "$LOGFILE"
    echo "NeMo-RL Training with Ray Cluster" | tee -a "$LOGFILE"
    echo "Mode: Using existing allocation with ray.sub" | tee -a "$LOGFILE"
    echo "=========================================" | tee -a "$LOGFILE"
    echo "Job ID: $JOBID" | tee -a "$LOGFILE"
    echo "Nodes: $SLURM_NNODES" | tee -a "$LOGFILE"
    echo "GPUs per node: $SLURM_GPUS_PER_NODE" | tee -a "$LOGFILE"
    echo "Container: ${CONTAINER}" | tee -a "$LOGFILE"
    echo "Config: ${CONFIG_FILE}" | tee -a "$LOGFILE"
    echo "Checkpoint dir: ${CHECKPOINT_DIR}" | tee -a "$LOGFILE"
    echo "Started at: $(date)" | tee -a "$LOGFILE"
    echo "Log file: $LOGFILE" | tee -a "$LOGFILE"
    echo "Ray logs: $RAY_LOG_DIR" | tee -a "$LOGFILE"
    echo "=========================================" | tee -a "$LOGFILE"
    echo "" | tee -a "$LOGFILE"

    # Set SLURM environment variables that ray.sub expects
    # These are normally set by sbatch but need to be explicit with salloc
    export SLURM_SUBMIT_DIR="$(pwd)"
    export SLURM_JOB_ID="$JOBID"
    export SLURM_JOB_NUM_NODES="$SLURM_NNODES"
    export SLURM_JOB_NODELIST="$SLURM_NODELIST"
    export SLURM_JOB_ACCOUNT="$SLURM_ACCOUNT"
    export SLURM_JOB_PARTITION="$SLURM_PARTITION"

    # Set custom log directory with datetime for ray.sub (under logs/)
    export BASE_LOG_DIR="${SLURM_SUBMIT_DIR}/logs"
    export LOG_DIR="${BASE_LOG_DIR}/${RAY_LOG_DIR}"

    # Fix Ray CPU detection in containers
    # Ray's container CPU detection is incorrect and only sees ~2 CPUs
    # This env var tells Ray to use multiprocessing.cpu_count() instead
    export RAY_USE_MULTIPROCESSING_CPU_COUNT=1

    # Clean up any stale ENDED file from previous runs
    # This file is used by ray.sub to signal termination, and if left from a previous run
    # it will cause the new run to immediately terminate
    ENDED_FILE="${LOG_DIR}/ENDED"
    if [ -f "$ENDED_FILE" ]; then
        echo "Removing stale ENDED file from previous run: $ENDED_FILE" | tee -a "$LOGFILE"
        rm -f "$ENDED_FILE"
    fi

    # Run ray.sub as a bash script (not via srun or source)
    # This allows ray.sub to execute its internal srun commands properly
    bash "$RAY_SUB_SCRIPT" 2>&1 | tee -a "$LOGFILE"

else
    # =========================================================================
    # MODE 2: Submit new job via sbatch
    # =========================================================================

    # Set SLURM_SUBMIT_DIR if not already set (when running outside sbatch)
    export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    # All logs go under logs/ directory
    export BASE_LOG_DIR="${SLURM_SUBMIT_DIR}/logs"

    # Fix Ray CPU detection in containers
    export RAY_USE_MULTIPROCESSING_CPU_COUNT=1

    echo "========================================="
    echo "NeMo-RL Training with Ray Cluster"
    echo "Mode: Submitting new job via sbatch"
    echo "========================================="
    echo "Nodes: ${SLURM_JOB_NUM_NODES:-2}"
    echo "GPUs per node: ${GPUS_PER_NODE}"
    echo "Container: ${CONTAINER}"
    echo "Config: ${CONFIG_FILE}"
    echo "Checkpoint dir: ${CHECKPOINT_DIR}"
    echo "Started at: $(date)"
    echo "Ray logs will be in: ${BASE_LOG_DIR}/\${SLURM_JOB_ID}-<datetime>-logs"
    echo "========================================="
    echo ""

    # Source the ray.sub script inline to inherit SLURM environment
    # ray.sub will create LOG_DIR as ${BASE_LOG_DIR}/${SLURM_JOB_ID}-logs
    source "$RAY_SUB_SCRIPT"
fi
