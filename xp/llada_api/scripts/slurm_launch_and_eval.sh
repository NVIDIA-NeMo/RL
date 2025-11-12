#!/bin/bash

# SLURM Orchestration Script for LLaDA Server + Evaluation
# This script launches the inference server on SLURM, waits for it to be ready,
# then launches the evaluation script on SLURM with the correct server address.

set -e

# Default values - Server
MODEL_PATH=""
DCP_PATH=""
BASE_MODEL="GSAI-ML/LLaDA-8B-Instruct"
ENGINE=""
ALGORITHM=""
BATCH_SIZE=16
MAX_WAIT_TIME=0.01
PORT=8000
VERBOSE=false
NO_CHAT_TEMPLATE=false

# Default values - Server SLURM
SERVER_JOB_NAME="llada-batch-server"
SERVER_TIME="4:00:00"
SERVER_GPUS=1
SERVER_CPUS=16
SERVER_MEM="128G"
SERVER_PARTITION="interactive"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/mfathi/containers/nemo_rl_base.sqsh"

# Default values - Evaluation
BENCHMARK="gsm8k:4"
TEMPERATURE=0.7
TOP_P=0.95
TOKENS_TO_GENERATE=512
STEPS=256
BLOCK_LENGTH=8
CFG_SCALE=0.0
REMASKING="low_confidence"
GENERATION_ALGORITHM="dual_cache"
EVAL_MODEL="llada-8b-instruct"
OUTPUT_DIR="."
EXPNAME=""
MAX_SAMPLES=""
QUICK_TEST=false
KEEP_THINKING=false
THRESHOLD=""
FACTOR=""

# Default values - Evaluation SLURM
EVAL_JOB_NAME="llada-eval"
EVAL_TIME="2:00:00"
EVAL_GPUS=0  # Evaluation typically doesn't need GPU
EVAL_CPUS=8
EVAL_MEM="32G"
EVAL_PARTITION="interactive"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_eval() {
    echo -e "${BLUE}[EVAL]${NC} $1"
}

# Function to show usage
show_usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

This script orchestrates launching an inference server and evaluation on SLURM.
It will:
  1. Launch the LLaDA batch server on SLURM
  2. Wait for the server to be ready and capture its address
  3. Launch the evaluation script on SLURM with the server address

================================================================================
SERVER OPTIONS
================================================================================

Model Options:
  -m, --model-path PATH       Path to HuggingFace model or model name
  -d, --dcp-path PATH         Path to DCP checkpoint
  -b, --base-model MODEL      Base model for DCP conversion (default: $BASE_MODEL)
  
Engine Options:
  --engine ENGINE             Inference engine: fast-dllm, dinfer, nemotron
  --algorithm ALGO            Algorithm: basic, prefix_cache, dual_cache, nemotron, etc.
  
Server Processing Options:
  --batch-size SIZE           Batch size (default: $BATCH_SIZE)
  --max-wait-time TIME        Max wait time in seconds (default: $MAX_WAIT_TIME)
  --port PORT                 Server port (default: $PORT)
  --verbose                   Enable verbose logging
  --no-chat-template          Disable chat template

Server SLURM Options:
  --server-job-name NAME      Server job name (default: $SERVER_JOB_NAME)
  --server-time TIME          Server time limit (default: $SERVER_TIME)
  --server-gpus NUM           Server GPUs (default: $SERVER_GPUS)
  --server-cpus NUM           Server CPUs (default: $SERVER_CPUS)
  --server-mem SIZE           Server memory (default: $SERVER_MEM)
  --server-partition PART     Server partition (default: $SERVER_PARTITION)

================================================================================
EVALUATION OPTIONS
================================================================================

Benchmark Options:
  --benchmark BENCH           Benchmark to evaluate (default: $BENCHMARK)
  --output-dir DIR            Output directory (default: $OUTPUT_DIR)
  --expname NAME              Experiment name
  --max-samples NUM           Maximum samples to evaluate
  --quick-test                Quick test mode
  --keep-thinking             Keep <think> tags in output
  
Inference Options:
  --eval-model NAME           Model name for eval (default: $EVAL_MODEL)
  --temperature TEMP          Temperature (default: $TEMPERATURE)
  --top-p PROB                Top-p (default: $TOP_P)
  --tokens-to-generate NUM    Max tokens (default: $TOKENS_TO_GENERATE)
  
Diffusion Options:
  --steps NUM                 Diffusion steps (default: $STEPS)
  --block-length NUM          Block length (default: $BLOCK_LENGTH)
  --cfg-scale SCALE           CFG scale (default: $CFG_SCALE)
  --remasking STRATEGY        Remasking strategy (default: $REMASKING)
  --generation-algorithm ALGO Generation algorithm (default: $GENERATION_ALGORITHM)
  --threshold VAL             Confidence threshold (optional)
  --factor VAL                Factor for dynamic decoding (optional)
  
Evaluation SLURM Options:
  --eval-job-name NAME        Eval job name (default: $EVAL_JOB_NAME)
  --eval-time TIME            Eval time limit (default: $EVAL_TIME)
  --eval-gpus NUM             Eval GPUs (default: $EVAL_GPUS)
  --eval-cpus NUM             Eval CPUs (default: $EVAL_CPUS)
  --eval-mem SIZE             Eval memory (default: $EVAL_MEM)
  --eval-partition PART       Eval partition (default: $EVAL_PARTITION)

================================================================================
GENERAL OPTIONS
================================================================================

  --container IMAGE           Container image path (default: $CONTAINER_IMAGE)
  -h, --help                  Show this help message

Environment Variables:
  \$ACCOUNT                    SLURM account (required)

================================================================================
EXAMPLES
================================================================================

# Basic usage with HuggingFace model
export ACCOUNT=your_account
$0 --model-path GSAI-ML/LLaDA-8B-Instruct --benchmark gsm8k:4

# With DCP checkpoint
$0 --dcp-path /path/to/checkpoint.dcp --base-model GSAI-ML/LLaDA-8B-Instruct

# Quick test with custom settings
$0 --model-path GSAI-ML/LLaDA-8B-Instruct --quick-test --steps 128

# Multi-GPU server with specific algorithm
$0 --model-path GSAI-ML/LLaDA-8B-Instruct --server-gpus 4 --algorithm dinfer_hierarchy

# Custom SLURM resources
$0 --model-path GSAI-ML/LLaDA-8B-Instruct \\
   --server-gpus 2 --server-mem 256G --server-partition batch \\
   --eval-partition interactive --eval-cpus 16

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        # Server model options
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
        --port)
            PORT="$2"
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
        # Server SLURM options
        --server-job-name)
            SERVER_JOB_NAME="$2"
            shift 2
            ;;
        --server-time)
            SERVER_TIME="$2"
            shift 2
            ;;
        --server-gpus)
            SERVER_GPUS="$2"
            shift 2
            ;;
        --server-cpus)
            SERVER_CPUS="$2"
            shift 2
            ;;
        --server-mem)
            SERVER_MEM="$2"
            shift 2
            ;;
        --server-partition)
            SERVER_PARTITION="$2"
            shift 2
            ;;
        # Evaluation options
        --benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --expname)
            EXPNAME="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --quick-test)
            QUICK_TEST=true
            shift 1
            ;;
        --keep-thinking)
            KEEP_THINKING=true
            shift 1
            ;;
        --eval-model)
            EVAL_MODEL="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --tokens-to-generate)
            TOKENS_TO_GENERATE="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --block-length)
            BLOCK_LENGTH="$2"
            shift 2
            ;;
        --cfg-scale)
            CFG_SCALE="$2"
            shift 2
            ;;
        --remasking)
            REMASKING="$2"
            shift 2
            ;;
        --generation-algorithm)
            GENERATION_ALGORITHM="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --factor)
            FACTOR="$2"
            shift 2
            ;;
        # Evaluation SLURM options
        --eval-job-name)
            EVAL_JOB_NAME="$2"
            shift 2
            ;;
        --eval-time)
            EVAL_TIME="$2"
            shift 2
            ;;
        --eval-gpus)
            EVAL_GPUS="$2"
            shift 2
            ;;
        --eval-cpus)
            EVAL_CPUS="$2"
            shift 2
            ;;
        --eval-mem)
            EVAL_MEM="$2"
            shift 2
            ;;
        --eval-partition)
            EVAL_PARTITION="$2"
            shift 2
            ;;
        # General options
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

# Validate arguments
if [[ -z "$MODEL_PATH" && -z "$DCP_PATH" ]]; then
    print_error "Either --model-path or --dcp-path must be provided"
    show_usage
    exit 1
fi

if [[ -n "$MODEL_PATH" && -n "$DCP_PATH" ]]; then
    print_error "Only one of --model-path or --dcp-path can be provided"
    show_usage
    exit 1
fi

# Validate SLURM account
if [[ -z "$ACCOUNT" ]]; then
    print_error "ACCOUNT environment variable must be set"
    echo "  export ACCOUNT=your_slurm_account"
    exit 1
fi

# Get script paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLADA_API_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$LLADA_API_DIR/../.." && pwd)"
EVAL_SCRIPT="$PROJECT_DIR/xp/nemo-skills/eval_llada.py"

# Verify eval script exists
if [[ ! -f "$EVAL_SCRIPT" ]]; then
    print_error "Evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi

# Create log directory
LOG_DIR="$PROJECT_DIR/logs/slurm_launch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server.log"
EVAL_LOG="$LOG_DIR/eval.log"

print_status "Log directory created: $LOG_DIR"

echo "=============================================================="
print_status "SLURM Orchestration: Server + Evaluation"
echo "=============================================================="
print_status "Server configuration:"
echo "  • Job name: $SERVER_JOB_NAME"
echo "  • GPUs: $SERVER_GPUS | CPUs: $SERVER_CPUS | Memory: $SERVER_MEM"
echo "  • Partition: $SERVER_PARTITION | Time: $SERVER_TIME"
echo "  • Port: $PORT | Batch size: $BATCH_SIZE"
if [[ -n "$ENGINE" ]]; then
    echo "  • Engine: $ENGINE"
fi
if [[ -n "$ALGORITHM" ]]; then
    echo "  • Algorithm: $ALGORITHM"
fi
echo ""
print_eval "Evaluation configuration:"
echo "  • Job name: $EVAL_JOB_NAME"
echo "  • GPUs: $EVAL_GPUS | CPUs: $EVAL_CPUS | Memory: $EVAL_MEM"
echo "  • Partition: $EVAL_PARTITION | Time: $EVAL_TIME"
echo "  • Benchmark: $BENCHMARK"
echo "  • Generation algorithm: $GENERATION_ALGORITHM"
echo "=============================================================="
echo ""

# Build server command
SERVER_CMD="$SCRIPT_DIR/start_llada_batch_server.sh"
SERVER_ARGS=""

if [[ -n "$MODEL_PATH" ]]; then
    SERVER_ARGS="$SERVER_ARGS --model-path '$MODEL_PATH'"
fi

if [[ -n "$DCP_PATH" ]]; then
    SERVER_ARGS="$SERVER_ARGS --dcp-path '$DCP_PATH' --base-model '$BASE_MODEL'"
fi

if [[ -n "$ENGINE" ]]; then
    SERVER_ARGS="$SERVER_ARGS --engine '$ENGINE'"
fi

if [[ -n "$ALGORITHM" ]]; then
    SERVER_ARGS="$SERVER_ARGS --algorithm '$ALGORITHM'"
fi

SERVER_ARGS="$SERVER_ARGS --port $PORT --batch-size $BATCH_SIZE --max-wait-time $MAX_WAIT_TIME"
SERVER_ARGS="$SERVER_ARGS --job-name '$SERVER_JOB_NAME' --time '$SERVER_TIME'"
SERVER_ARGS="$SERVER_ARGS --gpus $SERVER_GPUS --cpus $SERVER_CPUS --mem '$SERVER_MEM'"
SERVER_ARGS="$SERVER_ARGS --partition '$SERVER_PARTITION' --container '$CONTAINER_IMAGE'"

if [[ "$VERBOSE" == true ]]; then
    SERVER_ARGS="$SERVER_ARGS --verbose"
fi

if [[ "$NO_CHAT_TEMPLATE" == true ]]; then
    SERVER_ARGS="$SERVER_ARGS --no-chat-template"
fi

# Step 1: Launch the server in the background
print_status "Step 1: Launching inference server on SLURM..."
print_status "Command: $SERVER_CMD $SERVER_ARGS"
echo ""

# Start server in background and redirect output to log
eval "$SERVER_CMD $SERVER_ARGS" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

print_status "Server job started (PID: $SERVER_PID)"
print_status "Server log: $SERVER_LOG"
echo ""

# Step 2: Monitor server log for node and port information
print_status "Step 2: Waiting for server to be ready..."
print_status "Monitoring server log for connection information..."

SERVER_NODE=""
MAX_WAIT=600  # Wait up to 10 minutes
WAITED=0

while [[ -z "$SERVER_NODE" && $WAITED -lt $MAX_WAIT ]]; do
    if [[ -f "$SERVER_LOG" ]]; then
        # Look for the line that shows the compute node
        SERVER_NODE=$(grep -m1 "Server starting on compute node:" "$SERVER_LOG" | awk '{print $NF}' || true)
        
        if [[ -n "$SERVER_NODE" ]]; then
            print_status "Server detected on node: $SERVER_NODE"
            break
        fi
    fi
    
    sleep 2
    WAITED=$((WAITED + 2))
    
    # Check if server process is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        print_error "Server process terminated unexpectedly"
        print_error "Check server log: $SERVER_LOG"
        exit 1
    fi
done

if [[ -z "$SERVER_NODE" ]]; then
    print_error "Timeout waiting for server to start (waited ${MAX_WAIT}s)"
    print_error "Check server log: $SERVER_LOG"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Wait for the server to be fully ready (check for Uvicorn running message)
print_status "Waiting for server to be fully initialized..."
WAITED=0
while ! grep -q "Uvicorn running on" "$SERVER_LOG" && [[ $WAITED -lt $MAX_WAIT ]]; do
    sleep 2
    WAITED=$((WAITED + 2))
    
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        print_error "Server process terminated unexpectedly"
        print_error "Check server log: $SERVER_LOG"
        exit 1
    fi
done

if ! grep -q "Uvicorn running on" "$SERVER_LOG"; then
    print_error "Server failed to start properly (waited ${MAX_WAIT}s)"
    print_error "Check server log: $SERVER_LOG"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

print_status "✓ Server is ready!"
SERVER_URL="http://${SERVER_NODE}:${PORT}/v1"
print_status "Server URL: $SERVER_URL"
echo ""

# Step 3: Launch evaluation job
print_eval "Step 3: Launching evaluation job on SLURM..."

# Build evaluation command
EVAL_ARGS="--server-address '$SERVER_URL' --model '$EVAL_MODEL'"
EVAL_ARGS="$EVAL_ARGS --benchmark '$BENCHMARK' --output-dir '$OUTPUT_DIR'"
EVAL_ARGS="$EVAL_ARGS --temperature $TEMPERATURE --top-p $TOP_P"
EVAL_ARGS="$EVAL_ARGS --tokens-to-generate $TOKENS_TO_GENERATE"
EVAL_ARGS="$EVAL_ARGS --steps $STEPS --block-length $BLOCK_LENGTH"
EVAL_ARGS="$EVAL_ARGS --cfg-scale $CFG_SCALE --remasking '$REMASKING'"
EVAL_ARGS="$EVAL_ARGS --generation-algorithm '$GENERATION_ALGORITHM'"

if [[ -n "$THRESHOLD" ]]; then
    EVAL_ARGS="$EVAL_ARGS --threshold $THRESHOLD"
fi

if [[ -n "$FACTOR" ]]; then
    EVAL_ARGS="$EVAL_ARGS --factor $FACTOR"
fi

if [[ -n "$EXPNAME" ]]; then
    EVAL_ARGS="$EVAL_ARGS --expname '$EXPNAME'"
fi

if [[ -n "$MAX_SAMPLES" ]]; then
    EVAL_ARGS="$EVAL_ARGS --max-samples $MAX_SAMPLES"
fi

if [[ "$QUICK_TEST" == true ]]; then
    EVAL_ARGS="$EVAL_ARGS --quick-test"
fi

if [[ "$KEEP_THINKING" == true ]]; then
    EVAL_ARGS="$EVAL_ARGS --keep-thinking"
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

# Create the evaluation command block
EVAL_COMMAND_BLOCK=$(cat <<EOF
# Unset UV_CACHE_DIR to prevent conflicts
unset UV_CACHE_DIR

# Environment setup
export PATH="/root/.local/bin:\$PATH"
export PYTHONPATH="$PROJECT_DIR:\${PYTHONPATH:-}"
VENV_DIR="/opt/nemo_rl_venv"

echo "===================================================================="
echo "LLaDA Evaluation starting on compute node: \$(hostname)"
echo "===================================================================="

# Activate container environment
echo "[1/3] Activating container's Python environment..."
if [ -f "\$VENV_DIR/bin/activate" ]; then
    source \$VENV_DIR/bin/activate
    echo "Container Python environment activated."
else
    echo "Warning: No activation script found"
fi

# Sync dependencies
echo "[2/3] Syncing dependencies from uv.lock..."
uv sync --locked --no-install-project --extra vllm
if [ \$? -ne 0 ]; then
    echo "Error: Failed to sync dependencies. Exiting."
    exit 1
fi
echo "Dependencies installed successfully."

# Run evaluation
echo "[3/3] Starting evaluation..."
echo "Server URL: $SERVER_URL"
echo "Benchmark: $BENCHMARK"
echo "=================================================="

python3 "$EVAL_SCRIPT" $EVAL_ARGS

echo "=================================================="
echo "Evaluation completed!"
EOF
)

print_eval "Submitting evaluation job..."
print_eval "Evaluation log: $EVAL_LOG"

# Determine GPU flag for eval job
if [[ "$EVAL_GPUS" -gt 0 ]]; then
    EVAL_GPU_FLAG="--gpus-per-node=$EVAL_GPUS"
else
    EVAL_GPU_FLAG=""
fi

srun --job-name="$EVAL_JOB_NAME" \
     --time="$EVAL_TIME" \
     $EVAL_GPU_FLAG \
     --cpus-per-task="$EVAL_CPUS" \
     --mem="$EVAL_MEM" \
     --partition="$EVAL_PARTITION" \
     --account="$ACCOUNT" \
     --no-container-mount-home \
     --container-image="$CONTAINER_IMAGE" \
     --container-workdir="$PROJECT_DIR" \
     --container-mounts="$CONTAINER_MOUNTS" \
     bash -c "$EVAL_COMMAND_BLOCK" > "$EVAL_LOG" 2>&1 &

EVAL_PID=$!

print_eval "✓ Evaluation job started (PID: $EVAL_PID)"
echo ""

# Step 4: Monitor both jobs
print_status "Step 4: Monitoring jobs..."
echo ""
print_status "Both jobs are running in the background:"
echo "  • Server (PID $SERVER_PID): $SERVER_LOG"
echo "  • Evaluation (PID $EVAL_PID): $EVAL_LOG"
echo ""
print_status "You can monitor progress with:"
echo "  tail -f $SERVER_LOG"
echo "  tail -f $EVAL_LOG"
echo ""
print_status "Waiting for evaluation to complete..."

# Wait for evaluation to complete
wait $EVAL_PID
EVAL_EXIT_CODE=$?

echo ""
echo "=============================================================="
if [[ $EVAL_EXIT_CODE -eq 0 ]]; then
    print_status "✓ Evaluation completed successfully!"
else
    print_error "✗ Evaluation failed with exit code $EVAL_EXIT_CODE"
    print_error "Check evaluation log: $EVAL_LOG"
fi
echo "=============================================================="
echo ""

# Ask user about server
print_status "The inference server is still running."
echo "  • Server node: $SERVER_NODE"
echo "  • Server URL: $SERVER_URL"
echo "  • Server log: $SERVER_LOG"
echo ""
print_status "To stop the server, kill the process (PID $SERVER_PID):"
echo "  kill $SERVER_PID"
echo ""
print_status "Or let it continue running for additional evaluations."

exit $EVAL_EXIT_CODE

