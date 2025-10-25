#!/bin/bash

# Multi-GPU LLaDA Batch Server on SLURM with Load Balancing
# This script submits a SLURM job that starts multiple worker servers and a load balancer

set -e

# Default values
NUM_GPUS=4
BASE_WORKER_PORT=8001
LOAD_BALANCER_PORT=8000
MODEL_PATH=""
DCP_PATH=""
BASE_MODEL="GSAI-ML/LLaDA-8B-Instruct"
TEMP_DIR="/tmp/llada_hf_converted"
ENGINE=""
ALGORITHM=""
BATCH_SIZE=8
MAX_WAIT_TIME=0.1
VERBOSE=false
NO_CHAT_TEMPLATE=false
HOST="0.0.0.0"

# SLURM options
JOB_NAME="llada-multi-gpu"
TIME="4:00:00"
CPUS_PER_TASK=16
MEM="128G"
PARTITION="interactive"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/mfathi/containers/nemo_rl_base.sqsh"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Launch multi-GPU LLaDA/Nemotron batch server on SLURM with load balancing."
    echo ""
    echo "Model Options:"
    echo "  -m, --model-path PATH   Path to HuggingFace model or model name"
    echo "  -d, --dcp-path PATH     Path to DCP checkpoint"
    echo "  -b, --base-model MODEL  Base model name for DCP (default: $BASE_MODEL)"
    echo ""
    echo "GPU Options:"
    echo "  --num-gpus NUM          Number of GPUs to use (default: $NUM_GPUS)"
    echo ""
    echo "Port Options:"
    echo "  --port PORT             Load balancer port (default: $LOAD_BALANCER_PORT)"
    echo "  --worker-base-port PORT Base port for workers (default: $BASE_WORKER_PORT)"
    echo ""
    echo "Server Options:"
    echo "  --engine ENGINE         Inference engine (fast-dllm, dinfer, nemotron)"
    echo "  --algorithm ALGO        Specific algorithm within engine"
    echo "  --batch-size SIZE       Batch size per worker (default: $BATCH_SIZE)"
    echo "  --max-wait-time TIME    Max wait time for batching (default: $MAX_WAIT_TIME)"
    echo "  --verbose               Enable verbose logging"
    echo "  --no-chat-template      Disable chat template"
    echo ""
    echo "SLURM Options:"
    echo "  --job-name NAME         SLURM job name (default: $JOB_NAME)"
    echo "  --time TIME             Job time limit (default: $TIME)"
    echo "  --cpus NUM              CPUs per task (default: $CPUS_PER_TASK)"
    echo "  --mem SIZE              Memory (default: $MEM)"
    echo "  --partition PART        SLURM partition (default: $PARTITION)"
    echo "  --container IMAGE       Container image path (default: $CONTAINER_IMAGE)"
    echo ""
    echo "Environment Variables:"
    echo "  \$ACCOUNT               SLURM account (required)"
    echo ""
    echo "Examples:"
    echo ""
    echo "  # Run on 4 GPUs"
    echo "  export ACCOUNT=your_account"
    echo "  $0 --model-path GSAI-ML/LLaDA-8B-Instruct --num-gpus 4"
    echo ""
    echo "  # Run on 8 GPUs with custom settings"
    echo "  $0 --model-path GSAI-ML/LLaDA-8B-Instruct --num-gpus 8 --batch-size 16"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--dcp-path)
            DCP_PATH="$2"
            shift 2
            ;;
        -b|--base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --port)
            LOAD_BALANCER_PORT="$2"
            shift 2
            ;;
        --worker-base-port)
            BASE_WORKER_PORT="$2"
            shift 2
            ;;
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        --algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-wait-time)
            MAX_WAIT_TIME="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift 1
            ;;
        --no-chat-template)
            NO_CHAT_TEMPLATE=true
            shift 1
            ;;
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --cpus)
            CPUS_PER_TASK="$2"
            shift 2
            ;;
        --mem)
            MEM="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --container)
            CONTAINER_IMAGE="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate
if [[ -z "$MODEL_PATH" && -z "$DCP_PATH" ]]; then
    print_error "Either --model-path or --dcp-path must be provided"
    exit 1
fi

if [[ -z "$ACCOUNT" ]]; then
    print_error "ACCOUNT environment variable must be set"
    exit 1
fi

# Get paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLADA_API_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$LLADA_API_DIR/../.." && pwd)"

WORKER_SCRIPT="$LLADA_API_DIR/llada_batch_server.py"
LB_SCRIPT="$LLADA_API_DIR/llada_load_balancer.py"

# Build base worker arguments
# If DCP path is provided, we'll convert it once inside the container before starting workers
WORKER_BASE_ARGS="--host localhost --batch-size $BATCH_SIZE --max-wait-time $MAX_WAIT_TIME"

# For HF models, add model-path directly
if [[ -n "$MODEL_PATH" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --model-path '$MODEL_PATH'"
fi

# For DCP models, we'll convert once and then use the converted path
# (conversion happens inside container, so we don't add it to WORKER_BASE_ARGS yet)

if [[ -n "$ENGINE" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --engine '$ENGINE'"
fi

if [[ -n "$ALGORITHM" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --algorithm '$ALGORITHM'"
fi

if [[ "$VERBOSE" == true ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --verbose"
fi

if [[ "$NO_CHAT_TEMPLATE" == true ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --no-chat-template"
fi

# Prepare DCP path (make absolute if it's a local directory)
if [[ -n "$DCP_PATH" ]] && [[ -d "$DCP_PATH" ]]; then
    DCP_ABS_PATH=$(realpath "$DCP_PATH")
else
    DCP_ABS_PATH="$DCP_PATH"
fi

# Build worker ports list
WORKER_PORTS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    WORKER_PORTS+=($((BASE_WORKER_PORT + i)))
done

# Print configuration
print_status "SLURM Multi-GPU Configuration:"
echo "  Job name: $JOB_NAME"
echo "  GPUs: $NUM_GPUS"
echo "  Time limit: $TIME"
echo "  CPUs per task: $CPUS_PER_TASK"
echo "  Memory: $MEM"
echo "  Partition: $PARTITION"
echo "  Account: $ACCOUNT"
echo "  Load Balancer Port: $LOAD_BALANCER_PORT"
echo "  Worker Ports: ${WORKER_PORTS[*]}"

# Create command block
COMMAND_BLOCK=$(cat <<'EOF_OUTER'
# Unset UV_CACHE_DIR
unset UV_CACHE_DIR

# Environment setup
export PATH="/root/.local/bin:$PATH"
export PYTHONPATH="PROJECT_DIR_PLACEHOLDER:${PYTHONPATH:-}"
VENV_DIR="/opt/nemo_rl_venv"

echo "===================================================================="
echo "Multi-GPU LLaDA Server starting on: $(hostname)"
echo "GPUs: NUM_GPUS_PLACEHOLDER"
echo "===================================================================="

# Activate environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    source $VENV_DIR/bin/activate
else
    echo "Warning: No activation script found"
fi

# Install dependencies
echo "[1/5] Syncing dependencies..."
uv sync --locked --no-install-project --extra vllm
uv pip install fastapi uvicorn httpx

# Convert DCP checkpoint once if needed (before starting workers)
CONVERTED_MODEL_PATH=""
if [[ -n "DCP_ABS_PATH_PLACEHOLDER" ]]; then
    echo "[2/5] Converting DCP checkpoint to HuggingFace format (shared by all workers)..."
    
    # Use a single shared temp directory for all workers
    SHARED_TEMP_DIR="/tmp/llada_hf_converted_shared_$$"
    mkdir -p "$SHARED_TEMP_DIR"
    
    # Run the conversion using Python
    # Capture both stdout and stderr from the conversion
    CONVERSION_OUTPUT=$($VENV_DIR/bin/python - 2>&1 <<EOF
import sys
import os

try:
    from nemo_rl.utils.native_checkpoint import convert_dcp_to_hf, convert_structured_dcp_to_hf
    
    dcp_path = "DCP_ABS_PATH_PLACEHOLDER"
    temp_dir = "$SHARED_TEMP_DIR"
    base_model = "BASE_MODEL_PLACEHOLDER"
    
    # Check if this is a structured checkpoint
    weights_dir = os.path.join(dcp_path, "weights")
    tokenizer_dir = os.path.join(dcp_path, "tokenizer")
    
    if os.path.exists(weights_dir) and os.path.exists(tokenizer_dir):
        print(f"Detected structured DCP checkpoint", flush=True)
        hf_path = convert_structured_dcp_to_hf(
            dcp_root_path=dcp_path,
            hf_ckpt_path=temp_dir,
            model_name_or_path=base_model,
            overwrite=True
        )
    else:
        print(f"Using legacy DCP checkpoint format", flush=True)
        hf_path = convert_dcp_to_hf(
            dcp_ckpt_path=dcp_path,
            hf_ckpt_path=temp_dir,
            model_name_or_path=base_model,
            tokenizer_name_or_path=base_model,
            overwrite=True
        )
    
    print(f"Conversion completed: {hf_path}", flush=True)
    
except Exception as e:
    print(f"ERROR: DCP conversion failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
)
    CONVERSION_EXIT_CODE=$?
    
    # Always print the conversion output
    echo "$CONVERSION_OUTPUT"
    
    if [ $CONVERSION_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "=========================================="
        echo "ERROR: DCP CHECKPOINT CONVERSION FAILED"
        echo "=========================================="
        echo ""
        echo "The DCP to HuggingFace conversion failed with the output above."
        echo ""
        echo "Possible causes:"
        echo "  1. Missing dependencies (torch.distributed.checkpoint)"
        echo "  2. Corrupted DCP checkpoint files"
        echo "  3. Incompatible base model"
        echo "  4. Insufficient disk space"
        echo ""
        echo "DCP Path: DCP_ABS_PATH_PLACEHOLDER"
        echo "Base Model: BASE_MODEL_PLACEHOLDER"
        echo "Target Path: $SHARED_TEMP_DIR"
        echo ""
        echo "=========================================="
        exit 1
    fi
    
    CONVERTED_MODEL_PATH="$SHARED_TEMP_DIR"
    echo "DCP converted successfully to: $CONVERTED_MODEL_PATH"
    echo "All workers will load from this shared HF checkpoint"
fi

echo "[3/5] Starting GPU workers..."

# Start workers
WORKER_PIDS=()
for i in $(seq 0 $((NUM_GPUS_PLACEHOLDER - 1))); do
    WORKER_PORT=$((BASE_WORKER_PORT_PLACEHOLDER + i))
    echo "Starting worker $i on GPU $i (port $WORKER_PORT)"
    
    # Build worker arguments
    WORKER_ARGS="WORKER_BASE_ARGS_PLACEHOLDER"
    
    # If DCP was converted, add the converted model path
    if [[ -n "$CONVERTED_MODEL_PATH" ]]; then
        WORKER_ARGS="$WORKER_ARGS --model-path '$CONVERTED_MODEL_PATH'"
    fi
    
    # Log the full command for debugging
    echo "  Command: CUDA_VISIBLE_DEVICES=$i python WORKER_SCRIPT_PLACEHOLDER --port $WORKER_PORT $WORKER_ARGS"
    
    # Start worker with detailed logging (both stdout and stderr to log file)
    CUDA_VISIBLE_DEVICES=$i $VENV_DIR/bin/python -u "WORKER_SCRIPT_PLACEHOLDER" --port $WORKER_PORT $WORKER_ARGS > "/tmp/worker_${i}.log" 2>&1 &
    WORKER_PID=$!
    WORKER_PIDS+=($WORKER_PID)
    echo "Worker $i started (PID: $WORKER_PID, log: /tmp/worker_${i}.log)"
    
    # Small delay between worker starts to avoid race conditions
    sleep 0.5
done

echo "[4/5] Waiting for workers to initialize..."
sleep 10

# Check if workers are still running and show their logs if they crashed
echo "Checking worker health..."
CRASHED_WORKERS=()
for i in $(seq 0 $((NUM_GPUS_PLACEHOLDER - 1))); do
    PID=${WORKER_PIDS[$i]}
    if ! kill -0 $PID 2>/dev/null; then
        echo "WARNING: Worker $i (PID $PID) has crashed!"
        CRASHED_WORKERS+=($i)
    else
        echo "Worker $i (PID $PID) is running"
    fi
done

# If any workers crashed, show their logs and exit
if [ ${#CRASHED_WORKERS[@]} -gt 0 ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: WORKER CRASH DETECTED DURING STARTUP"
    echo "=========================================="
    echo ""
    echo "The following workers crashed during startup:"
    for i in "${CRASHED_WORKERS[@]}"; do
        echo "  - Worker $i (GPU $i)"
    done
    echo ""
    echo "Detailed logs from crashed workers:"
    echo "=========================================="
    for i in "${CRASHED_WORKERS[@]}"; do
        echo ""
        echo "---------- Worker $i (GPU $i) Full Log ----------"
        if [ -f "/tmp/worker_${i}.log" ]; then
            # Show full log if small, or last 100 lines if large
            LOG_LINES=$(wc -l < "/tmp/worker_${i}.log" 2>/dev/null || echo "0")
            if [ "$LOG_LINES" -lt 100 ]; then
                cat "/tmp/worker_${i}.log"
            else
                echo "(Showing last 100 lines of $LOG_LINES total lines)"
                echo ""
                tail -100 "/tmp/worker_${i}.log"
            fi
        else
            echo "ERROR: Log file not found at /tmp/worker_${i}.log"
            echo "This usually means the worker failed before any output was generated."
        fi
        echo ""
        echo "------------------------------------------------"
    done
    echo "=========================================="
    echo ""
    echo "Stopping all remaining workers..."
    for PID in "${WORKER_PIDS[@]}"; do
        kill $PID 2>/dev/null || true
    done
    echo ""
    echo "TROUBLESHOOTING:"
    echo "  - Check if the model path is accessible"
    echo "  - Verify GPU availability and CUDA setup"
    echo "  - Check if required Python packages are installed"
    echo "  - Review the error messages in the logs above"
    echo ""
    exit 1
fi

echo "All workers initialized successfully!"
sleep 10

echo "[5/5] Starting load balancer..."
$VENV_DIR/bin/python "LB_SCRIPT_PLACEHOLDER" --host 0.0.0.0 --port LOAD_BALANCER_PORT_PLACEHOLDER --worker-host localhost --worker-ports WORKER_PORTS_PLACEHOLDER VERBOSE_FLAG_PLACEHOLDER 2>&1 | while IFS= read -r line; do
    echo "$line"
    
    if [[ "$line" =~ "Uvicorn running on".*":LOAD_BALANCER_PORT_PLACEHOLDER" ]]; then
        COMPUTE_NODE=$(hostname)
        echo
        echo "========== MULTI-GPU LLADA SERVER STARTED =========="
        echo
        echo "----------[ 1. Create SSH Tunnel ]----------"
        echo "Run on your LOCAL machine:"
        echo
        echo "   ssh -N -L LOAD_BALANCER_PORT_PLACEHOLDER:$COMPUTE_NODE:LOAD_BALANCER_PORT_PLACEHOLDER ${USER}@your_cluster_login_node"
        echo
        echo "----------[ 2. Access the API ]----------"
        echo "   API Base URL:    http://localhost:LOAD_BALANCER_PORT_PLACEHOLDER"
        echo "   Health Check:    http://localhost:LOAD_BALANCER_PORT_PLACEHOLDER/health"
        echo "   LB Stats:        http://localhost:LOAD_BALANCER_PORT_PLACEHOLDER/stats"
        echo "   API Docs:        http://localhost:LOAD_BALANCER_PORT_PLACEHOLDER/docs"
        echo
        echo "----------[ 3. Workers ]----------"
        for i in $(seq 0 $((NUM_GPUS_PLACEHOLDER - 1))); do
            echo "   Worker $i (GPU $i): Log at /tmp/worker_${i}.log"
        done
        echo "===================================================="
        echo
    fi
done

# Check worker health after load balancer starts (or fails)
echo ""
echo "========== FINAL HEALTH CHECK =========="
echo "Checking if all workers are still running..."
CRASHED_WORKERS=()
for i in $(seq 0 $((NUM_GPUS_PLACEHOLDER - 1))); do
    PID=${WORKER_PIDS[$i]}
    if ! kill -0 $PID 2>/dev/null; then
        echo "✗ Worker $i (PID $PID) has CRASHED"
        CRASHED_WORKERS+=($i)
    else
        echo "✓ Worker $i (PID $PID) is running"
    fi
done

# If any workers crashed, show their logs
if [ ${#CRASHED_WORKERS[@]} -gt 0 ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: DETECTED ${#CRASHED_WORKERS[@]} CRASHED WORKER(S)"
    echo "=========================================="
    echo ""
    echo "Crashed workers:"
    for i in "${CRASHED_WORKERS[@]}"; do
        echo "  - Worker $i (GPU $i)"
    done
    echo ""
    echo "Detailed logs from crashed workers:"
    echo "=========================================="
    for i in "${CRASHED_WORKERS[@]}"; do
        echo ""
        echo "---------- Worker $i (GPU $i) Full Log ----------"
        if [ -f "/tmp/worker_${i}.log" ]; then
            # Show full log if small, or last 100 lines if large
            LOG_LINES=$(wc -l < "/tmp/worker_${i}.log" 2>/dev/null || echo "0")
            if [ "$LOG_LINES" -lt 100 ]; then
                cat "/tmp/worker_${i}.log"
            else
                echo "(Showing last 100 lines of $LOG_LINES total lines)"
                echo ""
                tail -100 "/tmp/worker_${i}.log"
            fi
        else
            echo "ERROR: Log file not found at /tmp/worker_${i}.log"
        fi
        echo ""
        echo "------------------------------------------------"
    done
    echo "=========================================="
    echo ""
    echo "TROUBLESHOOTING:"
    echo "  - Check if the model path is accessible in the container"
    echo "  - Verify GPU availability (nvidia-smi)"
    echo "  - Check for CUDA/PyTorch compatibility issues"
    echo "  - Review the error messages in the logs above"
    echo "  - Look for OOM (out of memory) errors"
    echo ""
fi

# Cleanup on exit
for PID in "${WORKER_PIDS[@]}"; do
    kill $PID 2>/dev/null || true
done

# Clean up shared temp directory if it was created
if [[ -n "$CONVERTED_MODEL_PATH" ]] && [[ -d "$CONVERTED_MODEL_PATH" ]]; then
    echo "Cleaning up shared temp directory: $CONVERTED_MODEL_PATH"
    rm -rf "$CONVERTED_MODEL_PATH"
fi
EOF_OUTER
)

# Replace placeholders
COMMAND_BLOCK="${COMMAND_BLOCK//PROJECT_DIR_PLACEHOLDER/$PROJECT_DIR}"
COMMAND_BLOCK="${COMMAND_BLOCK//NUM_GPUS_PLACEHOLDER/$NUM_GPUS}"
COMMAND_BLOCK="${COMMAND_BLOCK//BASE_WORKER_PORT_PLACEHOLDER/$BASE_WORKER_PORT}"
COMMAND_BLOCK="${COMMAND_BLOCK//LOAD_BALANCER_PORT_PLACEHOLDER/$LOAD_BALANCER_PORT}"
COMMAND_BLOCK="${COMMAND_BLOCK//WORKER_SCRIPT_PLACEHOLDER/$WORKER_SCRIPT}"
COMMAND_BLOCK="${COMMAND_BLOCK//LB_SCRIPT_PLACEHOLDER/$LB_SCRIPT}"
COMMAND_BLOCK="${COMMAND_BLOCK//WORKER_BASE_ARGS_PLACEHOLDER/$WORKER_BASE_ARGS}"
COMMAND_BLOCK="${COMMAND_BLOCK//WORKER_PORTS_PLACEHOLDER/${WORKER_PORTS[*]}}"
COMMAND_BLOCK="${COMMAND_BLOCK//DCP_ABS_PATH_PLACEHOLDER/${DCP_ABS_PATH:-}}"
COMMAND_BLOCK="${COMMAND_BLOCK//BASE_MODEL_PLACEHOLDER/$BASE_MODEL}"

if [[ "$VERBOSE" == true ]]; then
    COMMAND_BLOCK="${COMMAND_BLOCK//VERBOSE_FLAG_PLACEHOLDER/--verbose}"
else
    COMMAND_BLOCK="${COMMAND_BLOCK//VERBOSE_FLAG_PLACEHOLDER/}"
fi

# Build container mounts
CONTAINER_MOUNTS="$PROJECT_DIR:$PROJECT_DIR"

if [[ -n "$MODEL_PATH" ]] && [[ -d "$MODEL_PATH" || -f "$MODEL_PATH" ]]; then
    MODEL_ABS_PATH=$(realpath "$MODEL_PATH")
    CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$MODEL_ABS_PATH:$MODEL_ABS_PATH"
fi

if [[ -n "$DCP_PATH" ]] && [[ -d "$DCP_PATH" ]]; then
    DCP_ABS_PATH=$(realpath "$DCP_PATH")
    CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$DCP_ABS_PATH:$DCP_ABS_PATH"
fi

# Submit job
print_status "Submitting SLURM job..."

srun --job-name="$JOB_NAME" \
     --time="$TIME" \
     --gpus-per-node="$NUM_GPUS" \
     --cpus-per-task="$CPUS_PER_TASK" \
     --mem="$MEM" \
     --partition="$PARTITION" \
     --account="$ACCOUNT" \
     --no-container-mount-home \
     --container-image="$CONTAINER_IMAGE" \
     --container-workdir="$PROJECT_DIR" \
     --container-mounts="$CONTAINER_MOUNTS" \
     bash -c "$COMMAND_BLOCK"

