#!/bin/bash

# Single Checkpoint Evaluation Script
# Launches server and runs evaluation automatically, capturing server address
# Works for both local and SLURM modes

set -e

# Default values
DCP_PATH=""
BASE_MODEL="nvidia/Nemotron-Diffusion-Research-4B-v0"
BENCHMARK="gsm8k:1"
OUTPUT_DIR=""
MODEL_NAME="nemotron-4b"
GENERATION_ALGORITHM="nemotron"
THRESHOLD=0.9
TOKENS_TO_GENERATE=512
STEPS=512
BLOCK_LENGTH=32
BATCH_SIZE=1
SERVER_GPUS=8
VERBOSE=false
PORT=8000
MAX_SERVER_WAIT=300  # 5 minutes

# SLURM options
USE_SLURM=false
SERVER_PARTITION="interactive"
EVAL_PARTITION="interactive"
SERVER_TIME="4:00:00"
EVAL_TIME="2:00:00"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/mfathi/containers/nemo_rl_base.sqsh"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

A single script that launches a server and runs evaluation automatically.
Captures the server address so you don't have to manually type it!

Required Options:
  --dcp-path PATH             Path to DCP checkpoint (required)
  --output-dir DIR            Output directory for results (required)

Model Options:
  --base-model MODEL          Base model for DCP (default: $BASE_MODEL)
  --model-name NAME           Model name for eval (default: $MODEL_NAME)
  
Benchmark Options:
  --benchmark BENCH           Benchmark to evaluate (default: $BENCHMARK)
  --generation-algorithm ALGO Algorithm (default: $GENERATION_ALGORITHM)
  --threshold VAL             Threshold (default: $THRESHOLD)
  --tokens-to-generate NUM    Max tokens (default: $TOKENS_TO_GENERATE)
  --steps NUM                 Diffusion steps (default: $STEPS)
  --block-length NUM          Block length (default: $BLOCK_LENGTH)
  
Server Options:
  --batch-size SIZE           Server batch size (default: $BATCH_SIZE)
  --server-gpus NUM           Number of GPUs (default: $SERVER_GPUS)
  --port PORT                 Server port (default: $PORT)
  --verbose                   Enable verbose logging
  
Execution Mode:
  --slurm                     Run on SLURM (default: local/interactive)
  --server-partition PART     SLURM partition for server (default: $SERVER_PARTITION)
  --eval-partition PART       SLURM partition for eval (default: $EVAL_PARTITION)
  --container IMAGE           Container image path (default: $CONTAINER_IMAGE)

Examples:
  # Local/Interactive mode (server runs in background)
  $0 --dcp-path /path/to/checkpoint --output-dir /path/to/results
  
  # SLURM mode (both as SLURM jobs)
  export ACCOUNT=your_account
  $0 --slurm --dcp-path /path/to/checkpoint --output-dir /path/to/results
  
  # Custom benchmark and parameters
  $0 --dcp-path /path/to/checkpoint --output-dir /path/to/results \\
     --benchmark math:2 --steps 256 --server-gpus 4

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) show_usage; exit 0 ;;
        --dcp-path) DCP_PATH="$2"; shift 2 ;;
        --base-model) BASE_MODEL="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --benchmark) BENCHMARK="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --generation-algorithm) GENERATION_ALGORITHM="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --tokens-to-generate) TOKENS_TO_GENERATE="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --block-length) BLOCK_LENGTH="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --server-gpus) SERVER_GPUS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --verbose) VERBOSE=true; shift 1 ;;
        --slurm) USE_SLURM=true; shift 1 ;;
        --server-partition) SERVER_PARTITION="$2"; shift 2 ;;
        --eval-partition) EVAL_PARTITION="$2"; shift 2 ;;
        --container) CONTAINER_IMAGE="$2"; shift 2 ;;
        --server-time) SERVER_TIME="$2"; shift 2 ;;
        --eval-time) EVAL_TIME="$2"; shift 2 ;;
        *) print_error "Unknown option: $1"; show_usage; exit 1 ;;
    esac
done

# Validate required arguments
if [[ -z "$DCP_PATH" ]]; then
    print_error "--dcp-path is required"
    show_usage
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    print_error "--output-dir is required"
    show_usage
    exit 1
fi

if [[ "$USE_SLURM" == true ]] && [[ -z "$ACCOUNT" ]]; then
    print_error "ACCOUNT environment variable must be set for SLURM mode"
    exit 1
fi

# Get script paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EVAL_SCRIPT="$PROJECT_DIR/xp/nemo-skills/eval_llada.py"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create log directory
LOG_DIR="$OUTPUT_DIR/logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server.log"
EVAL_LOG="$LOG_DIR/eval.log"

echo "=============================================================="
print_status "Single Checkpoint Evaluation"
echo "=============================================================="
echo "DCP Path: $DCP_PATH"
echo "Base Model: $BASE_MODEL"
echo "Benchmark: $BENCHMARK"
echo "Output: $OUTPUT_DIR"
echo "Logs: $LOG_DIR"
echo "Mode: $([ "$USE_SLURM" == true ] && echo "SLURM" || echo "Local/Interactive")"
echo "=============================================================="
echo ""

if [[ "$USE_SLURM" == true ]]; then
    # SLURM mode - use the orchestration script
    print_status "Running in SLURM mode..."
    
    ORCH_SCRIPT="$SCRIPT_DIR/slurm_launch_and_eval.sh"
    
    ARGS="--dcp-path '$DCP_PATH' --base-model '$BASE_MODEL'"
    ARGS="$ARGS --benchmark '$BENCHMARK' --output-dir '$OUTPUT_DIR'"
    ARGS="$ARGS --eval-model '$MODEL_NAME' --generation-algorithm '$GENERATION_ALGORITHM'"
    ARGS="$ARGS --threshold $THRESHOLD --tokens-to-generate $TOKENS_TO_GENERATE"
    ARGS="$ARGS --steps $STEPS --block-length $BLOCK_LENGTH"
    ARGS="$ARGS --batch-size $BATCH_SIZE --server-gpus $SERVER_GPUS --port $PORT"
    ARGS="$ARGS --server-partition '$SERVER_PARTITION' --eval-partition '$EVAL_PARTITION'"
    ARGS="$ARGS --container '$CONTAINER_IMAGE'"
    ARGS="$ARGS --server-time '$SERVER_TIME' --eval-time '$EVAL_TIME'"
    
    if [[ "$VERBOSE" == true ]]; then
        ARGS="$ARGS --verbose"
    fi
    
    eval "$ORCH_SCRIPT $ARGS"
    
else
    # Local/Interactive mode - start server in background, run eval in foreground
    print_status "Running in Local/Interactive mode..."
    print_status "Step 1: Starting server in background..."
    
    SERVER_SCRIPT="$SCRIPT_DIR/start_llada_batch_server.sh"
    
    SERVER_ARGS="--local --dcp-path '$DCP_PATH' --base-model '$BASE_MODEL'"
    SERVER_ARGS="$SERVER_ARGS --batch-size $BATCH_SIZE --gpus $SERVER_GPUS --port $PORT"
    
    if [[ "$VERBOSE" == true ]]; then
        SERVER_ARGS="$SERVER_ARGS --verbose"
    fi
    
    # Start server in background
    eval "$SERVER_SCRIPT $SERVER_ARGS" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    
    print_status "Server started (PID: $SERVER_PID)"
    print_status "Server log: $SERVER_LOG"
    
    # Function to cleanup on exit
    cleanup() {
        if [[ -n "$SERVER_PID" ]] && kill -0 $SERVER_PID 2>/dev/null; then
            print_status "Stopping server (PID: $SERVER_PID)..."
            kill $SERVER_PID 2>/dev/null || true
            wait $SERVER_PID 2>/dev/null || true
        fi
    }
    trap cleanup EXIT INT TERM
    
    # Wait for server to be ready
    print_status "Step 2: Waiting for server to be ready..."
    WAITED=0
    while [[ $WAITED -lt $MAX_SERVER_WAIT ]]; do
        if grep -q "Uvicorn running on" "$SERVER_LOG" 2>/dev/null; then
            print_status "✓ Server is ready!"
            break
        fi
        
        # Check if server crashed
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            print_error "Server process terminated unexpectedly"
            print_error "Check log: $SERVER_LOG"
            exit 1
        fi
        
        sleep 2
        WAITED=$((WAITED + 2))
    done
    
    if [[ $WAITED -ge $MAX_SERVER_WAIT ]]; then
        print_error "Timeout waiting for server (${MAX_SERVER_WAIT}s)"
        print_error "Check log: $SERVER_LOG"
        exit 1
    fi
    
    # Get server URL
    SERVER_URL="http://localhost:${PORT}/v1"
    print_status "Server URL: $SERVER_URL"
    echo ""
    
    # Run evaluation
    print_status "Step 3: Running evaluation..."
    
    export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
    
    EVAL_ARGS="--server-address '$SERVER_URL' --model '$MODEL_NAME'"
    EVAL_ARGS="$EVAL_ARGS --benchmark '$BENCHMARK' --output-dir '$OUTPUT_DIR'"
    EVAL_ARGS="$EVAL_ARGS --generation-algorithm '$GENERATION_ALGORITHM'"
    EVAL_ARGS="$EVAL_ARGS --threshold $THRESHOLD --tokens-to-generate $TOKENS_TO_GENERATE"
    EVAL_ARGS="$EVAL_ARGS --steps $STEPS --block-length $BLOCK_LENGTH"
    
    print_status "Starting evaluation..."
    print_status "Command: python3 $EVAL_SCRIPT $EVAL_ARGS"
    echo ""
    
    # Run evaluation and capture output
    eval "python3 $EVAL_SCRIPT $EVAL_ARGS" 2>&1 | tee "$EVAL_LOG"
    EVAL_EXIT_CODE=${PIPESTATUS[0]}
    
    echo ""
    echo "=============================================================="
    if [[ $EVAL_EXIT_CODE -eq 0 ]]; then
        print_status "✓ Evaluation completed successfully!"
    else
        print_error "✗ Evaluation failed with exit code $EVAL_EXIT_CODE"
    fi
    echo "=============================================================="
    echo ""
    print_status "Logs saved to:"
    echo "  Server: $SERVER_LOG"
    echo "  Eval: $EVAL_LOG"
    echo ""
    print_status "Results saved to: $OUTPUT_DIR"
    
    exit $EVAL_EXIT_CODE
fi

