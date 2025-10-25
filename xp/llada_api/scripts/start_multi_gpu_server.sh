#!/bin/bash

# Multi-GPU LLaDA Batch Server with Load Balancing
# This script starts multiple worker servers (one per GPU) and a load balancer in front.
# Each worker runs on its own GPU with CUDA_VISIBLE_DEVICES set appropriately.

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

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_gpu() {
    echo -e "${CYAN}[GPU]${NC} $1"
}

print_lb() {
    echo -e "${BLUE}[LB]${NC} $1"
}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Launch multi-GPU LLaDA/Nemotron batch server with load balancing."
    echo "Starts one worker per GPU, each on a different port, with a load balancer in front."
    echo ""
    echo "Model Options:"
    echo "  -m, --model-path PATH   Path to HuggingFace model or model name"
    echo "  -d, --dcp-path PATH     Path to DCP checkpoint"
    echo "  -b, --base-model MODEL  Base model name for DCP (default: $BASE_MODEL)"
    echo ""
    echo "GPU Options:"
    echo "  --num-gpus NUM          Number of GPUs to use (default: $NUM_GPUS)"
    echo "  --gpu-ids IDS           Specific GPU IDs to use (comma-separated, e.g., '0,1,2,3')"
    echo "                          If not specified, uses GPUs 0 through NUM_GPUS-1"
    echo ""
    echo "Port Options:"
    echo "  --port PORT             Load balancer port (default: $LOAD_BALANCER_PORT)"
    echo "  --worker-base-port PORT Base port for workers (default: $BASE_WORKER_PORT)"
    echo "                          Workers will use ports: BASE_PORT, BASE_PORT+1, ..."
    echo ""
    echo "Server Options:"
    echo "  --engine ENGINE         Inference engine (fast-dllm, dinfer, nemotron)"
    echo "  --algorithm ALGO        Specific algorithm within engine"
    echo "  --batch-size SIZE       Batch size per worker (default: $BATCH_SIZE)"
    echo "  --max-wait-time TIME    Max wait time for batching (default: $MAX_WAIT_TIME)"
    echo "  --verbose               Enable verbose logging"
    echo "  --no-chat-template      Disable chat template"
    echo ""
    echo "Examples:"
    echo ""
    echo "  # Run on 4 GPUs with LLaDA"
    echo "  $0 --model-path GSAI-ML/LLaDA-8B-Instruct --num-gpus 4"
    echo ""
    echo "  # Run on specific GPUs"
    echo "  $0 --model-path GSAI-ML/LLaDA-8B-Instruct --gpu-ids 0,2,4,6"
    echo ""
    echo "  # Run with custom ports"
    echo "  $0 --model-path GSAI-ML/LLaDA-8B-Instruct --num-gpus 2 --port 9000 --worker-base-port 9001"
    echo ""
    echo "  # Run with DCP checkpoint"
    echo "  $0 --dcp-path /path/to/checkpoint.dcp --num-gpus 4"
}

# Parse arguments
GPU_IDS=""
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
        --gpu-ids)
            GPU_IDS="$2"
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
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ -z "$MODEL_PATH" && -z "$DCP_PATH" ]]; then
    print_error "Either --model-path or --dcp-path must be provided"
    show_usage
    exit 1
fi

# Determine GPU IDs to use
if [[ -n "$GPU_IDS" ]]; then
    # User specified GPU IDs
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
    NUM_GPUS=${#GPU_ARRAY[@]}
    print_status "Using specified GPU IDs: ${GPU_ARRAY[*]}"
else
    # Use GPUs 0 through NUM_GPUS-1
    GPU_ARRAY=()
    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_ARRAY+=($i)
    done
    print_status "Using GPUs 0-$((NUM_GPUS-1))"
fi

# Get script paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLADA_API_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$LLADA_API_DIR/../.." && pwd)"

WORKER_SCRIPT="$LLADA_API_DIR/llada_batch_server.py"
LB_SCRIPT="$LLADA_API_DIR/llada_load_balancer.py"

if [[ ! -f "$WORKER_SCRIPT" ]]; then
    print_error "Worker script not found: $WORKER_SCRIPT"
    exit 1
fi

if [[ ! -f "$LB_SCRIPT" ]]; then
    print_error "Load balancer script not found: $LB_SCRIPT"
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# Convert DCP checkpoint once if needed (before starting workers)
CONVERTED_MODEL_PATH=""
if [[ -n "$DCP_PATH" ]]; then
    print_status "Converting DCP checkpoint to HuggingFace format (shared by all workers)..."
    
    # Use a single shared temp directory for all workers
    SHARED_TEMP_DIR="/tmp/llada_hf_converted_shared_$$"
    mkdir -p "$SHARED_TEMP_DIR"
    
    # Run the conversion using Python
    CONVERSION_SCRIPT=$(cat <<EOF
import sys
import os
sys.path.insert(0, "$PROJECT_DIR")

try:
    from nemo_rl.utils.native_checkpoint import convert_dcp_to_hf, convert_structured_dcp_to_hf
    
    dcp_path = "$DCP_PATH"
    temp_dir = "$SHARED_TEMP_DIR"
    base_model = "$BASE_MODEL"
    
    # Check if this is a structured checkpoint
    weights_dir = os.path.join(dcp_path, "weights")
    tokenizer_dir = os.path.join(dcp_path, "tokenizer")
    
    if os.path.exists(weights_dir) and os.path.exists(tokenizer_dir):
        print(f"Detected structured DCP checkpoint")
        hf_path = convert_structured_dcp_to_hf(
            dcp_root_path=dcp_path,
            hf_ckpt_path=temp_dir,
            model_name_or_path=base_model,
            overwrite=True
        )
    else:
        print(f"Using legacy DCP checkpoint format")
        hf_path = convert_dcp_to_hf(
            dcp_ckpt_path=dcp_path,
            hf_ckpt_path=temp_dir,
            model_name_or_path=base_model,
            tokenizer_name_or_path=base_model,
            overwrite=True
        )
    
    print(f"Conversion completed: {hf_path}")
    
except Exception as e:
    print(f"Error during conversion: {e}", file=sys.stderr)
    sys.exit(1)
EOF
)
    
    python3 -c "$CONVERSION_SCRIPT"
    if [ $? -ne 0 ]; then
        print_error "Failed to convert DCP checkpoint"
        exit 1
    fi
    
    CONVERTED_MODEL_PATH="$SHARED_TEMP_DIR"
    print_status "DCP converted successfully to: $CONVERTED_MODEL_PATH"
    print_status "All workers will load from this shared HF checkpoint"
fi

# Build base worker arguments (common to all workers)
WORKER_BASE_ARGS="--host localhost --batch-size $BATCH_SIZE --max-wait-time $MAX_WAIT_TIME"

# Use converted model path if DCP was converted, otherwise use MODEL_PATH
if [[ -n "$CONVERTED_MODEL_PATH" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --model-path '$CONVERTED_MODEL_PATH'"
elif [[ -n "$MODEL_PATH" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --model-path '$MODEL_PATH'"
fi

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

# Print configuration
echo "=============================================================="
print_status "Multi-GPU LLaDA Batch Server Configuration"
echo "=============================================================="
echo "  🖥️  Total GPUs: $NUM_GPUS"
echo "  🔢 GPU IDs: ${GPU_ARRAY[*]}"
echo "  🌐 Load Balancer: http://$HOST:$LOAD_BALANCER_PORT"
echo "  👷 Worker Ports: $BASE_WORKER_PORT - $((BASE_WORKER_PORT + NUM_GPUS - 1))"
echo "  📊 Batch Size (per worker): $BATCH_SIZE"
echo "  ⏱️  Max Wait Time: $MAX_WAIT_TIME s"
if [[ -n "$ENGINE" ]]; then
    echo "  ⚡ Engine: $ENGINE"
fi
if [[ -n "$ALGORITHM" ]]; then
    echo "  🔧 Algorithm: $ALGORITHM"
fi
echo "=============================================================="
echo ""

# Array to store PIDs
WORKER_PIDS=()

# Cleanup function
cleanup() {
    print_status "Shutting down..."
    
    # Kill load balancer
    if [[ -n "${LB_PID:-}" ]]; then
        print_lb "Stopping load balancer (PID: $LB_PID)"
        kill $LB_PID 2>/dev/null || true
    fi
    
    # Kill all workers
    for i in "${!WORKER_PIDS[@]}"; do
        PID=${WORKER_PIDS[$i]}
        if [[ -n "$PID" ]]; then
            print_gpu "Stopping worker $i (GPU ${GPU_ARRAY[$i]}, PID: $PID)"
            kill $PID 2>/dev/null || true
        fi
    done
    
    # Clean up shared temp directory if it was created
    if [[ -n "${CONVERTED_MODEL_PATH:-}" ]] && [[ -d "$CONVERTED_MODEL_PATH" ]]; then
        print_status "Cleaning up shared temp directory: $CONVERTED_MODEL_PATH"
        rm -rf "$CONVERTED_MODEL_PATH"
    fi
    
    print_status "All processes stopped"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM EXIT

# Start workers
print_status "Starting workers on $NUM_GPUS GPUs..."
echo ""

for i in "${!GPU_ARRAY[@]}"; do
    GPU_ID=${GPU_ARRAY[$i]}
    WORKER_PORT=$((BASE_WORKER_PORT + i))
    
    print_gpu "Starting worker $i on GPU $GPU_ID (port $WORKER_PORT)"
    
    # All workers use the same base arguments (including shared converted model path if DCP was used)
    WORKER_ARGS="$WORKER_BASE_ARGS"
    
    # Build the command
    CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python3 '$WORKER_SCRIPT' --port $WORKER_PORT $WORKER_ARGS"
    
    # Start worker in background
    eval "$CMD" > "/tmp/llada_worker_${i}.log" 2>&1 &
    WORKER_PID=$!
    WORKER_PIDS+=($WORKER_PID)
    
    print_gpu "Worker $i started (PID: $WORKER_PID, log: /tmp/llada_worker_${i}.log)"
done

echo ""
print_status "All workers started. Waiting for initialization..."
sleep 10

# Build worker ports list for load balancer
WORKER_PORTS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    WORKER_PORTS+=($((BASE_WORKER_PORT + i)))
done

# Start load balancer
print_lb "Starting load balancer on port $LOAD_BALANCER_PORT"
LB_CMD="python3 '$LB_SCRIPT' --host $HOST --port $LOAD_BALANCER_PORT --worker-host localhost --worker-ports ${WORKER_PORTS[*]}"

if [[ "$VERBOSE" == true ]]; then
    LB_CMD="$LB_CMD --verbose"
fi

eval "$LB_CMD" > "/tmp/llada_load_balancer.log" 2>&1 &
LB_PID=$!

print_lb "Load balancer started (PID: $LB_PID, log: /tmp/llada_load_balancer.log)"

echo ""
echo "=============================================================="
print_status "🚀 Multi-GPU Server is Running!"
echo "=============================================================="
echo ""
echo "  📍 Load Balancer API:     http://$HOST:$LOAD_BALANCER_PORT"
echo "  🏥 Health Check:          http://$HOST:$LOAD_BALANCER_PORT/health"
echo "  📊 Load Balancer Stats:   http://$HOST:$LOAD_BALANCER_PORT/stats"
echo "  📖 API Documentation:     http://$HOST:$LOAD_BALANCER_PORT/docs"
echo ""
echo "  👷 Workers:"
for i in "${!GPU_ARRAY[@]}"; do
    WORKER_PORT=$((BASE_WORKER_PORT + i))
    echo "     Worker $i (GPU ${GPU_ARRAY[$i]}): http://localhost:$WORKER_PORT"
done
echo ""
echo "  📋 Logs:"
echo "     Load Balancer: /tmp/llada_load_balancer.log"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "     Worker $i:      /tmp/llada_worker_${i}.log"
done
echo ""
echo "  💡 To view logs in real-time:"
echo "     tail -f /tmp/llada_load_balancer.log"
echo "     tail -f /tmp/llada_worker_0.log"
echo ""
echo "=============================================================="
print_status "Press Ctrl+C to stop all servers"
echo "=============================================================="

# Give everything a moment to stabilize, then check health
sleep 5

echo ""
echo "=============================================================="
print_status "Final Health Check"
echo "=============================================================="
print_status "Checking if all workers are still running..."

CRASHED_WORKERS=()
for i in "${!WORKER_PIDS[@]}"; do
    PID=${WORKER_PIDS[$i]}
    GPU_ID=${GPU_ARRAY[$i]}
    if ! kill -0 $PID 2>/dev/null; then
        print_error "✗ Worker $i (GPU $GPU_ID, PID $PID) has CRASHED"
        CRASHED_WORKERS+=($i)
    else
        print_gpu "✓ Worker $i (GPU $GPU_ID, PID $PID) is running"
    fi
done

# Check load balancer
if [[ -n "${LB_PID:-}" ]]; then
    if ! kill -0 $LB_PID 2>/dev/null; then
        print_error "✗ Load balancer (PID $LB_PID) has CRASHED"
    else
        print_lb "✓ Load balancer (PID $LB_PID) is running"
    fi
fi

# If any workers crashed, show their logs
if [ ${#CRASHED_WORKERS[@]} -gt 0 ]; then
    echo ""
    echo "=============================================================="
    print_error "DETECTED ${#CRASHED_WORKERS[@]} CRASHED WORKER(S)"
    echo "=============================================================="
    echo ""
    echo "Crashed workers:"
    for i in "${CRASHED_WORKERS[@]}"; do
        echo "  - Worker $i (GPU ${GPU_ARRAY[$i]})"
    done
    echo ""
    echo "Showing logs from crashed workers:"
    echo "=============================================================="
    for i in "${CRASHED_WORKERS[@]}"; do
        echo ""
        echo "---------- Worker $i (GPU ${GPU_ARRAY[$i]}) Log ----------"
        tail -100 "/tmp/llada_worker_${i}.log" 2>/dev/null || echo "Log file not found"
        echo ""
    done
    echo "=============================================================="
    echo ""
    print_warning "Some workers have crashed. Server may not work correctly."
    echo "Check the logs above for error details."
else
    echo ""
    print_status "✅ All workers are healthy!"
fi

echo ""
echo "=============================================================="

# Wait for user interrupt
wait

