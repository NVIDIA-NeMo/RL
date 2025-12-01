#!/bin/bash

# LLaDA Batch OpenAI API Server Startup Script
# This script provides easy ways to start the LLaDA batch server with different configurations.
# Can run locally (--local) or as a SLURM job (default).
# Defaults to batch server, but can launch standard streaming server with --streaming.

set -e

# Default values - LLaDA Server
HOST="0.0.0.0"
PORT=8000
MODEL_PATH=""
DCP_PATH=""
BASE_MODEL="GSAI-ML/LLaDA-8B-Instruct"
TEMP_DIR="/tmp/llada_hf_converted"
ENGINE=""  # Auto-detected based on model type (LLaDA→dinfer, Nemotron→nemotron)
ALGORITHM=""  # Optional specific algorithm within engine

# Default values - Batch Processing (Optimized for performance)
SERVER_MODE="batch"  # "batch" or "streaming"
BATCH_SIZE=16  # Increased for better GPU utilization
MAX_WAIT_TIME=0.01  # Reduced for lower latency
VERBOSE=false
NO_CHAT_TEMPLATE=false

# Server info sharing
SERVER_INFO_FILE=""

# Default values - SLURM
LOCAL_MODE=false
JOB_NAME="llada-batch-server"
TIME="4:00:00"
GPUS_PER_NODE=1
CPUS_PER_TASK=16
MEM="128G"
PARTITION="interactive"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/degert/data/enroot/nemo-rl:big-version-bump-from-githubci-18672942438.squashfs"

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

print_batch_info() {
    echo -e "${BLUE}[BATCH]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script launches the LLaDA API server with batch processing capabilities."
    echo "By default, it launches the BATCH server for improved throughput."
    echo ""
    echo "Server Type Options:"
    echo "  --batch                 Launch batch processing server (default)"
    echo "  --streaming             Launch standard streaming server"
    echo ""
    echo "LLaDA Server Options:"
    echo "  -h, --help              Show this help message"
    echo "  -H, --host HOST         Host to bind to (default: $HOST)"
    echo "  -p, --port PORT         Port to bind to (default: $PORT)"
    echo "  -m, --model-path PATH   Path to HuggingFace model directory or HuggingFace model name"
    echo "  -d, --dcp-path PATH     Path to DCP checkpoint"
    echo "  -b, --base-model MODEL  Base model name for DCP conversion (default: $BASE_MODEL)"
    echo "  -t, --temp-dir DIR      Temporary directory for DCP conversion (default: $TEMP_DIR)"
    echo ""
    echo "Inference Engine Options:"
    echo "  --engine ENGINE         Inference engine: fast-dllm, dinfer, nemotron"
    echo "                          (default: auto-detected - dinfer for LLaDA, nemotron for Nemotron)"
    echo "  --algorithm ALGO        Specific algorithm within engine (optional, uses engine default)"
    echo "                          fast-dllm: basic, prefix_cache, dual_cache"
    echo "                          dinfer: dinfer_blockwise, dinfer_hierarchy, dinfer_credit"
    echo "                          nemotron: nemotron"
    echo ""
    echo "Batch Processing Options (ignored with --streaming):"
    echo "  --batch-size SIZE       Maximum batch size (default: $BATCH_SIZE)"
    echo "  --max-wait-time TIME    Maximum time to wait for batch in seconds (default: $MAX_WAIT_TIME)"
    echo ""
    echo "Execution Mode:"
    echo "  --local                 Run locally (default: run as SLURM job)"
    echo "  --verbose               Enable verbose debug logging (helpful for troubleshooting)"
    echo "  --no-chat-template      Disable chat template application (feed raw text to tokenizer)"
    echo "  --server-info-file FILE Path to write server connection details for evaluation jobs"
    echo ""
    echo "GPU Options:"
    echo "  --gpus NUM              Number of GPUs (default: $GPUS_PER_NODE)"
    echo "                          If NUM > 1: automatically enables multi-GPU mode with load balancing"
    echo "                          If NUM = 1: runs standard single-GPU server"
    echo ""
    echo "SLURM Job Options (ignored with --local):"
    echo "  --job-name NAME         SLURM job name (default: $JOB_NAME)"
    echo "  --time TIME             Job time limit (default: $TIME)"
    echo "  --cpus NUM              CPUs per task (default: $CPUS_PER_TASK)"
    echo "  --mem SIZE              Memory per node (default: $MEM)"
    echo "  --partition PART        SLURM partition (default: $PARTITION)"
    echo "  --container IMAGE       Container image path (default: $CONTAINER_IMAGE)"
    echo ""
    echo "Environment Variables (for SLURM mode):"
    echo "  \$ACCOUNT               SLURM account (required for SLURM jobs)"
    echo "  \$LOG                   Log directory (optional, defaults to current dir)"
    echo ""
    echo "Examples:"
    echo ""
    echo "  # Local batch server with LLaDA (auto-selects dInfer - RECOMMENDED)"
    echo "  $0 --local --model-path GSAI-ML/LLaDA-8B-Instruct"
    echo ""
    echo "  # Local with explicit Fast-dLLM engine"
    echo "  $0 --local --model-path GSAI-ML/LLaDA-8B-Instruct --engine fast-dllm"
    echo ""
    echo "  # Local with specific algorithm"
    echo "  $0 --local --model-path GSAI-ML/LLaDA-8B-Instruct --algorithm dinfer_hierarchy"
    echo ""
    echo "  # Nemotron model (auto-selects nemotron engine)"
    echo "  $0 --local --model-path nvidia/Nemotron-Diffusion-Research-4B-v0"
    echo ""
    echo "  # SLURM batch server (auto-selects engine)"
    echo "  export ACCOUNT=your_account"
    echo "  $0 --model-path GSAI-ML/LLaDA-8B-Instruct --batch-size 16 --gpus 2"
    echo ""
    echo "  # SLURM with DCP checkpoint"
    echo "  $0 --dcp-path /path/to/checkpoint.dcp --base-model GSAI-ML/LLaDA-8B-Instruct"
    echo ""
    echo "  # Multi-GPU mode (auto-enabled when --gpus > 1)"
    echo "  $0 --local --gpus 4 --model-path GSAI-ML/LLaDA-8B-Instruct"
    echo ""
    echo "  # Multi-GPU on SLURM"
    echo "  export ACCOUNT=your_account"
    echo "  $0 --gpus 8 --model-path GSAI-ML/LLaDA-8B-Instruct"
    echo ""
    echo "Performance Tips:"
    echo "  • Engine auto-selects dInfer for LLaDA (10x+ faster than fast-dllm)"
    echo "  • Batch server provides additional 3-5x speedup for evaluation workloads"
    echo "  • Multi-GPU mode (--gpus > 1) scales linearly with load balancing"
    echo "  • Increase --batch-size for higher throughput (requires more GPU memory)"
    echo "  • Decrease --max-wait-time for lower latency"
    echo "  • Use --streaming only if you need real-time streaming responses"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        # Server type options
        --batch)
            SERVER_MODE="batch"
            shift 1
            ;;
        --streaming)
            SERVER_MODE="streaming"
            shift 1
            ;;
        # LLaDA Server options
        -H|--host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
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
        -t|--temp-dir)
            TEMP_DIR="$2"
            shift 2
            ;;
        # Engine options
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        --algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        # Batch processing options
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-wait-time)
            MAX_WAIT_TIME="$2"
            shift 2
            ;;
        # Execution mode
        --local)
            LOCAL_MODE=true
            shift 1
            ;;
        --verbose)
            VERBOSE=true
            shift 1
            ;;
        --no-chat-template)
            NO_CHAT_TEMPLATE=true
            shift 1
            ;;
        --server-info-file)
            SERVER_INFO_FILE="$2"
            shift 2
            ;;
        # SLURM options
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --gpus)
            GPUS_PER_NODE="$2"
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

# SLURM-specific validation
if [[ "$LOCAL_MODE" == false ]]; then
    # Validate environment variables for SLURM
    if [[ -z "$ACCOUNT" ]]; then
        print_error "ACCOUNT environment variable must be set for SLURM jobs"
        echo "  export ACCOUNT=your_slurm_account"
        exit 1
    fi
    
    # Update job name based on server mode
    if [[ "$SERVER_MODE" == "streaming" ]]; then
        JOB_NAME="${JOB_NAME/-batch/-stream}"
    fi
    
    print_status "Logs will be output directly to stdout for real-time viewing"
fi

# Get the absolute path to the Python scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLADA_API_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$LLADA_API_DIR/../.." && pwd)"  # NeMo-RL project root

if [[ -z "$SERVER_INFO_FILE" ]]; then
    SERVER_INFO_FILE="$LLADA_API_DIR/.llada_server_info"
fi

SERVER_INFO_FILE="$(realpath -m "$SERVER_INFO_FILE")"
SERVER_INFO_DIR="$(dirname "$SERVER_INFO_FILE")"
mkdir -p "$SERVER_INFO_DIR"

if [[ "$SERVER_MODE" == "batch" ]]; then
    SCRIPT_PATH="$LLADA_API_DIR/llada_batch_server.py"
else
    SCRIPT_PATH="$LLADA_API_DIR/llada_openai_server.py"
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
    print_error "Server script not found: $SCRIPT_PATH"
    if [[ "$SERVER_MODE" == "batch" ]]; then
        print_error "Make sure llada_batch_server.py exists in xp/llada_api/"
    fi
    exit 1
fi

# Build server command arguments
LLADA_ARGS="--host '$HOST' --port '$PORT'"

if [[ -n "$MODEL_PATH" ]]; then
    # Check if it's a local path or HuggingFace model name
    if [[ -d "$MODEL_PATH" ]] || [[ -f "$MODEL_PATH" ]]; then
        print_status "Using local HuggingFace model: $MODEL_PATH"
        MODEL_TYPE="local path"
    else
        print_status "Using HuggingFace model name: $MODEL_PATH"
        MODEL_TYPE="HuggingFace Hub"
    fi
    LLADA_ARGS="$LLADA_ARGS --model-path '$MODEL_PATH'"
    
elif [[ -n "$DCP_PATH" ]]; then
    print_status "Using DCP checkpoint: $DCP_PATH"
    print_status "Base model for conversion: $BASE_MODEL"
    print_status "Temporary HF directory: $TEMP_DIR"
    
    # For local mode, warn about DCP limitations
    if [[ "$LOCAL_MODE" == true ]]; then
        if [[ ! -d "$DCP_PATH" ]]; then
            print_error "DCP path does not exist: $DCP_PATH"
            exit 1
        fi
        
        print_warning "DCP checkpoints in local mode require NeMo-RL dependencies"
        print_warning "Make sure you have run: uv sync --locked --no-install-project"
        print_warning "For easier local testing, consider using: --model-path GSAI-ML/LLaDA-8B-Instruct"
    fi
    
    # Use absolute path for DCP to match the mounted path in container
    if [[ -d "$DCP_PATH" ]]; then
        DCP_ABS_PATH_FOR_ARGS=$(realpath "$DCP_PATH")
        LLADA_ARGS="$LLADA_ARGS --dcp-path '$DCP_ABS_PATH_FOR_ARGS' --base-model '$BASE_MODEL' --temp-dir '$TEMP_DIR'"
    else
        LLADA_ARGS="$LLADA_ARGS --dcp-path '$DCP_PATH' --base-model '$BASE_MODEL' --temp-dir '$TEMP_DIR'"
    fi
fi

# Add engine argument (only if specified)
if [[ -n "$ENGINE" ]]; then
    LLADA_ARGS="$LLADA_ARGS --engine '$ENGINE'"
fi

# Add algorithm argument (only if specified)
if [[ -n "$ALGORITHM" ]]; then
    LLADA_ARGS="$LLADA_ARGS --algorithm '$ALGORITHM'"
fi

# Add batch-specific arguments
if [[ "$SERVER_MODE" == "batch" ]]; then
    LLADA_ARGS="$LLADA_ARGS --batch-size '$BATCH_SIZE' --max-wait-time '$MAX_WAIT_TIME'"
fi

# Add verbose flag if enabled
if [[ "$VERBOSE" == true ]]; then
    LLADA_ARGS="$LLADA_ARGS --verbose"
fi

# Add no-chat-template flag if enabled
if [[ "$NO_CHAT_TEMPLATE" == true ]]; then
    LLADA_ARGS="$LLADA_ARGS --no-chat-template"
fi

# Function to run locally
run_local() {
    if [[ "$SERVER_MODE" == "batch" ]]; then
        print_batch_info "Running LLaDA BATCH server locally"
    else
        print_status "Running LLaDA STREAMING server locally"
    fi
    
    # Check for required Python packages
    print_status "Checking Python dependencies..."
    export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
    
    python3 -c "import fastapi, uvicorn, torch, transformers" 2>/dev/null || {
        print_error "Missing required Python packages. Please install using uv:"
        echo "  uv sync --locked --no-install-project  # Sync from uv.lock"
        echo "  uv pip install fastapi uvicorn                     # Install additional deps"
        echo ""
        echo "  # Alternative with traditional pip:"
        echo "  pip install fastapi uvicorn torch transformers"
        exit 1
    }
    
    # Check if NeMo-RL is available for DCP functionality
    if python3 -c "import nemo_rl.utils.native_checkpoint" 2>/dev/null; then
        print_status "NeMo-RL available - DCP checkpoint support enabled"
    else
        print_warning "NeMo-RL not available - DCP checkpoints will not work in local mode"
        print_warning "Use HuggingFace models for local testing: --model-path GSAI-ML/LLaDA-8B-Instruct"
    fi

    # Check GPU availability
    if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep -q "True"; then
        GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
        print_status "Found $GPU_COUNT GPU(s) available"
    else
        print_warning "No CUDA GPUs detected. Model will run on CPU (slower)"
    fi

    # Create temp directory if using DCP
    if [[ -n "$DCP_PATH" ]]; then
        mkdir -p "$TEMP_DIR"
        print_status "Created temporary directory: $TEMP_DIR"
    fi

    CLIENT_HOST="$HOST"
    if [[ "$CLIENT_HOST" == "0.0.0.0" || "$CLIENT_HOST" == "::" ]]; then
        CLIENT_HOST="localhost"
    fi

    SERVER_BASE_URL="http://${CLIENT_HOST}:${PORT}"
    SERVER_ADDRESS="${SERVER_BASE_URL}/v1"
    SERVER_HEALTH_URL="${SERVER_BASE_URL}/health"
    SERVER_INFO_GENERATED_AT="$(date -Iseconds)"

    cat > "$SERVER_INFO_FILE" <<EOF
# Auto-generated by start_llada_batch_server.sh
SERVER_INFO_GENERATED_AT="$SERVER_INFO_GENERATED_AT"
SERVER_MODE="$SERVER_MODE"
SERVER_INFO_SOURCE="local"
SERVER_STATUS="starting"
SERVER_BIND_HOST="$HOST"
SERVER_CLIENT_HOST="$CLIENT_HOST"
SERVER_PORT="$PORT"
SERVER_BASE_URL="$SERVER_BASE_URL"
SERVER_ADDRESS="$SERVER_ADDRESS"
SERVER_HEALTH_URL="$SERVER_HEALTH_URL"
EOF

    if [[ "$SERVER_MODE" == "batch" ]]; then
        SERVER_BATCH_STATS_URL="${SERVER_BASE_URL}/batch/stats"
        echo "SERVER_BATCH_STATS_URL=\"$SERVER_BATCH_STATS_URL\"" >> "$SERVER_INFO_FILE"
    fi

    print_status "Server connection info written to $SERVER_INFO_FILE"

    # Show server info
    if [[ "$SERVER_MODE" == "batch" ]]; then
        print_batch_info "Batch server configuration:"
        echo "  • Server Type: BATCH PROCESSING"
        echo "  • Inference Engine: $ENGINE"
        if [[ -n "$ALGORITHM" ]]; then
            echo "  • Algorithm: $ALGORITHM"
        else
            echo "  • Algorithm: (engine default)"
        fi
        echo "  • Batch Size: $BATCH_SIZE requests"
        echo "  • Max Wait Time: $MAX_WAIT_TIME seconds"
        echo "  • Host: $HOST"
        echo "  • Port: $PORT"
        echo "  • API Base URL: http://$HOST:$PORT"
        echo "  • Health Check: http://$HOST:$PORT/health"
        echo "  • Batch Stats: http://$HOST:$PORT/batch/stats"
        echo "  • API Docs: http://$HOST:$PORT/docs"
        echo ""
        print_batch_info "Performance:"
        if [[ "$ENGINE" == "dinfer" ]]; then
            echo "  • dInfer engine: 10x+ faster than Fast-dLLM"
        elif [[ "$ENGINE" == "fast-dllm" ]]; then
            echo "  • Fast-dLLM engine: Optimized LLaDA inference"
        fi
        echo "  • Batch processing: 3-5x additional speedup for evaluations"
        echo "  • Compatible with all existing evaluation scripts"
        echo "  • Note: Streaming responses are not available in batch mode"
    else
        print_status "Streaming server configuration:"
        echo "  • Server Type: STREAMING (real-time responses)"
        echo "  • Inference Engine: $ENGINE"
        if [[ -n "$ALGORITHM" ]]; then
            echo "  • Algorithm: $ALGORITHM"
        else
            echo "  • Algorithm: (engine default)"
        fi
        echo "  • Host: $HOST"
        echo "  • Port: $PORT"
        echo "  • API Base URL: http://$HOST:$PORT"
        echo "  • Health Check: http://$HOST:$PORT/health"
        echo "  • API Docs: http://$HOST:$PORT/docs"
    fi

    # Set up PYTHONPATH for local execution to include NeMo-RL
    export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
    print_status "Set PYTHONPATH to include NeMo-RL project: $PROJECT_DIR"

    # Build and execute command
    CMD="python3 '$SCRIPT_PATH' $LLADA_ARGS"
    if [[ "$SERVER_MODE" == "batch" ]]; then
        print_batch_info "Starting LLaDA Batch OpenAI API server..."
    else
        print_status "Starting LLaDA Streaming OpenAI API server..."
    fi
    echo "Command: $CMD"
    echo ""
    print_status "Press Ctrl+C to stop the server"
    echo "=================================================="

    eval "$CMD"
}

# Function to run on SLURM
run_slurm() {
    if [[ "$SERVER_MODE" == "batch" ]]; then
        print_batch_info "Submitting LLaDA BATCH server as SLURM job"
    else
        print_status "Submitting LLaDA STREAMING server as SLURM job"
    fi
    
    # Show SLURM job info
    print_status "SLURM job configuration:"
    echo "  • Job name: $JOB_NAME"
    echo "  • Time limit: $TIME"
    echo "  • GPUs per node: $GPUS_PER_NODE"
    echo "  • CPUs per task: $CPUS_PER_TASK"
    echo "  • Memory: $MEM"
    echo "  • Partition: $PARTITION"
    echo "  • Account: $ACCOUNT"
    echo "  • Container: $CONTAINER_IMAGE"
    echo "  • Logs: stdout (real-time)"

    # Show server info
    if [[ "$SERVER_MODE" == "batch" ]]; then
        print_batch_info "Batch server configuration:"
        echo "  • Server Type: BATCH PROCESSING"
        echo "  • Inference Engine: $ENGINE"
        if [[ -n "$ALGORITHM" ]]; then
            echo "  • Algorithm: $ALGORITHM"
        else
            echo "  • Algorithm: (engine default)"
        fi
        echo "  • Batch Size: $BATCH_SIZE requests"
        echo "  • Max Wait Time: $MAX_WAIT_TIME seconds"
        echo "  • Host: $HOST"
        echo "  • Port: $PORT"
        echo "  • Compute node URL: http://\$SLURMD_NODENAME:$PORT (from within job)"
        echo "  • Local access: http://localhost:$PORT (via SSH tunnel)"
        echo "  • Batch stats: http://localhost:$PORT/batch/stats"
    else
        print_status "Streaming server configuration:"
        echo "  • Server Type: STREAMING"
        if [[ -n "$ENGINE" ]]; then
            echo "  • Inference Engine: $ENGINE"
        else
            echo "  • Inference Engine: (auto-detected from model type)"
        fi
        if [[ -n "$ALGORITHM" ]]; then
            echo "  • Algorithm: $ALGORITHM"
        else
            echo "  • Algorithm: (engine default)"
        fi
        echo "  • Batch Size: $BATCH_SIZE requests"
        echo "  • Max Wait Time: $MAX_WAIT_TIME seconds"
        echo "  • Host: $HOST"
        echo "  • Port: $PORT"
        echo "  • Compute node URL: http://\$SLURMD_NODENAME:$PORT (from within job)"
        echo "  • Local access: http://localhost:$PORT (via SSH tunnel)"
    fi
    
    print_status "Connection setup:"
    echo "  • The server will display SSH tunnel commands when it starts"
    echo "  • Connection instructions will appear in the terminal output"
    echo "  • Server logs will be shown directly in stdout"
    print_status "Server info will be written to: $SERVER_INFO_FILE"
    print_status "Server info file resolved path: $(realpath -m "$SERVER_INFO_FILE")"
    print_status "Server info directory: $(dirname "$SERVER_INFO_FILE")"
    
    # Verify we can write to the file
    if ! touch "$SERVER_INFO_FILE" 2>/dev/null; then
        print_error "Cannot write to server info file: $SERVER_INFO_FILE"
        print_error "Check permissions for directory: $(dirname "$SERVER_INFO_FILE")"
        exit 1
    fi
    print_status "Verified write access to server info file"

    SERVER_INFO_PLACEHOLDER_TIME="$(date -Iseconds)"
    cat > "$SERVER_INFO_FILE" <<EOF
# Auto-generated by start_llada_batch_server.sh
SERVER_INFO_GENERATED_AT="$SERVER_INFO_PLACEHOLDER_TIME"
SERVER_MODE="$SERVER_MODE"
SERVER_INFO_SOURCE="slurm"
SERVER_STATUS="pending"
SERVER_BIND_HOST="$HOST"
SERVER_PORT="$PORT"
EOF
    print_status "Server info placeholder created."
    
    # Verify the write succeeded
    if [[ -f "$SERVER_INFO_FILE" ]] && grep -q "SERVER_STATUS=\"pending\"" "$SERVER_INFO_FILE"; then
        print_status "Verified placeholder write - file contains SERVER_STATUS=\"pending\""
    else
        print_error "Failed to verify server info file write!"
        if [[ -f "$SERVER_INFO_FILE" ]]; then
            print_error "File exists but doesn't contain expected content. Contents:"
            cat "$SERVER_INFO_FILE"
        else
            print_error "File doesn't exist after write attempt"
        fi
        exit 1
    fi

    # Create the command block that will run inside the container
    SERVER_TYPE_DISPLAY=$(echo "$SERVER_MODE" | tr '[:lower:]' '[:upper:]')
    COMMAND_BLOCK=$(cat <<EOF
# Unset UV_CACHE_DIR to prevent conflicts with host cache
unset UV_CACHE_DIR

# Environment setup
export PATH="/root/.local/bin:\$PATH"
export PYTHONPATH="$PROJECT_DIR:\${PYTHONPATH:-}"
VENV_DIR="/opt/nemo_rl_venv"

echo "===================================================================="
echo "LLaDA $SERVER_TYPE_DISPLAY OpenAI API Server starting on compute node: \$(hostname)"
echo "Using container's Python environment: \${VENV_DIR}"
echo "===================================================================="

# Activate the container's existing Python environment
echo "[1/3] Activating container's Python environment..."
if [ -f "\$VENV_DIR/bin/activate" ]; then
    source \$VENV_DIR/bin/activate
    echo "Container Python environment activated."
else
    echo "Warning: No activation script found at \$VENV_DIR/bin/activate"
    echo "Proceeding with container's default Python environment..."
fi

# Step 2: Prepare environment from uv.lock
echo "[2/3] Syncing dependencies to container environment from uv.lock..."
uv sync --locked --no-install-project --extra vllm
if [ \$? -ne 0 ]; then
    echo "Error: Failed to sync dependencies from uv.lock. Exiting."
    exit 1
fi

# Server-specific deps not in uv.lock
uv pip install fastapi uvicorn
if [ \$? -ne 0 ]; then
    echo "Error: Failed to install server dependencies. Exiting."
    exit 1
fi
echo "Dependencies installed successfully."

# Create temp directory if using DCP
if [[ -n "$DCP_PATH" ]]; then
    mkdir -p "$TEMP_DIR"
    echo "Created temporary directory: $TEMP_DIR"
fi

echo "[3/3] Starting LLaDA $SERVER_TYPE_DISPLAY OpenAI API server..."
COMPUTE_NODE=\$(hostname)
SERVER_INFO_FILE="$SERVER_INFO_FILE"
SERVER_INFO_GENERATED_AT="\$(date -Iseconds)"
SERVER_COMPUTE_HOST="\$COMPUTE_NODE"
SERVER_BASE_BIND_URL="http://$HOST:$PORT"
SERVER_BASE_URL="http://\$SERVER_COMPUTE_HOST:$PORT"
SERVER_ADDRESS="\$SERVER_BASE_URL/v1"
SERVER_HEALTH_URL="\$SERVER_BASE_URL/health"
SERVER_BATCH_STATS_URL="\$SERVER_BASE_URL/batch/stats"

echo "[Container] Writing initial server info to: \$SERVER_INFO_FILE"
echo "[Container] Server info directory: \$(dirname "\$SERVER_INFO_FILE")"
if [[ ! -d "\$(dirname "\$SERVER_INFO_FILE")" ]]; then
    echo "[Container] ERROR: Server info directory does not exist, creating it..."
    mkdir -p "\$(dirname "\$SERVER_INFO_FILE")" || {
        echo "[Container] ERROR: Failed to create server info directory!"
        echo "[Container] Check mount permissions for: \$(dirname "\$SERVER_INFO_FILE")"
        exit 1
    }
fi

cat > "\$SERVER_INFO_FILE" <<INFO
# Auto-generated by start_llada_batch_server.sh
SERVER_INFO_GENERATED_AT="$SERVER_INFO_GENERATED_AT"
SERVER_MODE="$SERVER_MODE"
SERVER_INFO_SOURCE="slurm"
SERVER_STATUS="initializing"
SERVER_BIND_HOST="$HOST"
SERVER_COMPUTE_NODE="\$SERVER_COMPUTE_HOST"
SERVER_CLIENT_HOST="\$SERVER_COMPUTE_HOST"
SERVER_PORT="$PORT"
SERVER_BASE_BIND_URL="$SERVER_BASE_BIND_URL"
SERVER_BASE_URL="\$SERVER_BASE_URL"
SERVER_ADDRESS="\$SERVER_ADDRESS"
SERVER_HEALTH_URL="\$SERVER_HEALTH_URL"
INFO

if [[ "$SERVER_MODE" == "batch" ]]; then
cat >> "\$SERVER_INFO_FILE" <<INFOBATCH
SERVER_BATCH_STATS_URL="\$SERVER_BATCH_STATS_URL"
INFOBATCH
fi

cat >> "\$SERVER_INFO_FILE" <<INFOEND
SLURM_JOB_ID="\${SLURM_JOB_ID:-}"
SLURM_JOB_NAME="\${SLURM_JOB_NAME:-}"
SLURM_JOB_NODELIST="\${SLURM_JOB_NODELIST:-}"
SLURMD_NODENAME="\${SLURMD_NODENAME:-}"
INFOEND

if [[ -f "\$SERVER_INFO_FILE" ]]; then
    echo "[Container] Successfully wrote server info file"
    echo "[Container] File contents:"
    cat "\$SERVER_INFO_FILE" | head -20
else
    echo "[Container] ERROR: Failed to write server info file at \$SERVER_INFO_FILE"
    exit 1
fi

echo "Server starting on compute node: \$COMPUTE_NODE"
echo "Server connection info: \$SERVER_INFO_FILE"
echo "=================================================="

# Start the server and provide connection info when it starts
\$VENV_DIR/bin/python "$SCRIPT_PATH" $LLADA_ARGS 2>&1 | while IFS= read -r line; do
    echo "\$line"
    
    # When we see the server startup message, show connection instructions  
    if [[ "\$line" =~ "Uvicorn running on".*":$PORT" ]]; then
        SERVER_READY_AT="\$(date -Iseconds)"
cat > "\$SERVER_INFO_FILE" <<READYINFO
# Auto-generated by start_llada_batch_server.sh
SERVER_INFO_GENERATED_AT="$SERVER_INFO_GENERATED_AT"
SERVER_READY_AT="\$SERVER_READY_AT"
SERVER_MODE="$SERVER_MODE"
SERVER_INFO_SOURCE="slurm"
SERVER_STATUS="running"
SERVER_BIND_HOST="$HOST"
SERVER_COMPUTE_NODE="\$SERVER_COMPUTE_HOST"
SERVER_CLIENT_HOST="\$SERVER_COMPUTE_HOST"
SERVER_PORT="$PORT"
SERVER_BASE_BIND_URL="$SERVER_BASE_BIND_URL"
SERVER_BASE_URL="\$SERVER_BASE_URL"
SERVER_ADDRESS="\$SERVER_ADDRESS"
SERVER_HEALTH_URL="\$SERVER_HEALTH_URL"
READYINFO
        if [[ "$SERVER_MODE" == "batch" ]]; then
cat >> "\$SERVER_INFO_FILE" <<READYBATCH
SERVER_BATCH_STATS_URL="\$SERVER_BATCH_STATS_URL"
READYBATCH
        fi
cat >> "\$SERVER_INFO_FILE" <<READYEND
SLURM_JOB_ID="\${SLURM_JOB_ID:-}"
SLURM_JOB_NAME="\${SLURM_JOB_NAME:-}"
SLURM_JOB_NODELIST="\${SLURM_JOB_NODELIST:-}"
SLURMD_NODENAME="\${SLURMD_NODENAME:-}"
READYEND
        echo "[INFO] Server info file updated: \$SERVER_INFO_FILE"
        echo
        echo "========== LLADA $SERVER_TYPE_DISPLAY API SERVER STARTED - CONNECTION INFO =========="
        echo
        echo "----------[ 1. LOCAL TERMINAL: Create SSH Tunnel ]----------"
        echo "Run this command on your LOCAL machine. It will seem to hang, which is normal."
        echo
        echo "   ssh -N -L $PORT:\$COMPUTE_NODE:$PORT \${USER}@your_cluster_login_node"
        echo
        echo "   (Replace 'your_cluster_login_node' with your cluster's login node address)"
        echo "------------------------------------------------------------" 
        echo
        echo "----------[ 2. ACCESS THE API: Use Local URLs ]----------"
        echo "After setting up the SSH tunnel, use these URLs on your LOCAL machine:"
        echo
        echo "   API Base URL:    http://localhost:$PORT"
        echo "   Health Check:    http://localhost:$PORT/health"
        echo "   API Documentation: http://localhost:$PORT/docs"
        if [[ "$SERVER_MODE" == "batch" ]]; then
            echo "   Batch Statistics: http://localhost:$PORT/batch/stats"
        fi
        echo
        echo "----------[ 3. TEST THE API ]----------"
        echo "Test with curl:"
        echo "   curl http://localhost:$PORT/health"
        if [[ "$SERVER_MODE" == "batch" ]]; then
            echo "   curl http://localhost:$PORT/batch/stats"
        fi
        echo
        echo "Or test with the test script:"
        if [[ "$SERVER_MODE" == "batch" ]]; then
            echo "   python xp/llada_api/test_batch_server.py"
        else
            echo "   python xp/llada_api/examples/llada_api_client.py"
        fi
        echo "=============================================================="
        echo
    fi
done
EOF
)

    # Build container mounts - start with project directory
    CONTAINER_MOUNTS="$PROJECT_DIR:$PROJECT_DIR"

    if [[ "$SERVER_INFO_FILE" != "$PROJECT_DIR"* ]]; then
        SERVER_INFO_DIR_ABS="$(dirname "$SERVER_INFO_FILE")"
        CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$SERVER_INFO_DIR_ABS:$SERVER_INFO_DIR_ABS"
        print_status "Auto-mounting server info directory: $SERVER_INFO_DIR_ABS"
        print_status "Server info file path (for container): $SERVER_INFO_FILE"
        
        # Verify the directory exists on the host
        if [[ ! -d "$SERVER_INFO_DIR_ABS" ]]; then
            print_warning "Server info directory does not exist on host: $SERVER_INFO_DIR_ABS"
            print_status "Creating directory on host..."
            mkdir -p "$SERVER_INFO_DIR_ABS" || {
                print_error "Failed to create server info directory"
                exit 1
            }
        fi
    fi
    
    # Auto-mount model path if it's a local directory
    if [[ -n "$MODEL_PATH" ]] && [[ -d "$MODEL_PATH" || -f "$MODEL_PATH" ]]; then
        # Get absolute path to ensure proper mounting
        MODEL_ABS_PATH=$(realpath "$MODEL_PATH")
        CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$MODEL_ABS_PATH:$MODEL_ABS_PATH"
        print_status "Auto-mounting model path: $MODEL_ABS_PATH"
    fi
    
    # Auto-mount DCP path if it's a local directory
    if [[ -n "$DCP_PATH" ]] && [[ -d "$DCP_PATH" ]]; then
        # Get absolute path to ensure proper mounting
        DCP_ABS_PATH=$(realpath "$DCP_PATH")
        CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$DCP_ABS_PATH:$DCP_ABS_PATH"
        print_status "Auto-mounting DCP path: $DCP_ABS_PATH"
    fi
    
    # Auto-mount temp directory if using DCP (in case it's outside project dir)
    if [[ -n "$DCP_PATH" ]] && [[ "$TEMP_DIR" != "/tmp/"* ]]; then
        # Only mount if temp dir is not in /tmp (which is usually available in containers)
        TEMP_ABS_PATH=$(realpath -m "$TEMP_DIR")  # -m creates path if it doesn't exist
        if [[ "$TEMP_ABS_PATH" != "$PROJECT_DIR"* ]]; then
            CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$TEMP_ABS_PATH:$TEMP_ABS_PATH"
            print_status "Auto-mounting temp directory: $TEMP_ABS_PATH"
        fi
    fi

    # Submit the SLURM job
    print_status "Submitting SLURM job..."
    print_status "Container mounts: $CONTAINER_MOUNTS"
    
    srun --job-name="$JOB_NAME" \
         --time="$TIME" \
         --gpus-per-node="$GPUS_PER_NODE" \
         --cpus-per-task="$CPUS_PER_TASK" \
         --mem="$MEM" \
         --partition="$PARTITION" \
         --account="$ACCOUNT" \
         --no-container-mount-home \
         --container-image="$CONTAINER_IMAGE" \
         --container-workdir="$PROJECT_DIR" \
         --container-mounts="$CONTAINER_MOUNTS" \
         --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"benchmarking","description":"DiffusionLLM benchmarking script which has periods of low GPU activity due to evaluation metrics calculating during runtime"}}' \
         bash -c "$COMMAND_BLOCK"
}

# Show startup info
echo "=============================================================="
if [[ "$SERVER_MODE" == "batch" ]]; then
    print_batch_info "LLaDA BATCH Server Startup Script"
    echo "  🚀 Launching BATCH server for high-throughput inference"
    if [[ -n "$ENGINE" ]]; then
        echo "  ⚡ Engine: $ENGINE"
        if [[ "$ENGINE" == "dinfer" ]]; then
            echo "     (dInfer: 10x+ faster than Fast-dLLM)"
        elif [[ "$ENGINE" == "fast-dllm" ]]; then
            echo "     (Fast-dLLM: Optimized LLaDA inference)"
        fi
    else
        echo "  ⚡ Engine: (auto-detected from model type)"
    fi
    if [[ -n "$ALGORITHM" ]]; then
        echo "  🔧 Algorithm: $ALGORITHM"
    else
        echo "  🔧 Algorithm: (using engine default)"
    fi
    echo "  📊 Batch size: $BATCH_SIZE | Max wait: ${MAX_WAIT_TIME}s"
else
    print_status "LLaDA STREAMING Server Startup Script"
    echo "  🌊 Launching STREAMING server for real-time responses"
    if [[ -n "$ENGINE" ]]; then
        echo "  ⚡ Engine: $ENGINE"
    else
        echo "  ⚡ Engine: (auto-detected from model type)"
    fi
    if [[ -n "$ALGORITHM" ]]; then
        echo "  🔧 Algorithm: $ALGORITHM"
    else
        echo "  🔧 Algorithm: (using engine default)"
    fi
fi

if [[ "$LOCAL_MODE" == true ]]; then
    echo "  💻 Execution mode: LOCAL"
else
    echo "  🖥️  Execution mode: SLURM"
fi

if [[ "$VERBOSE" == true ]]; then
    echo "  🔍 Verbose logging: ENABLED (detailed debugging)"
fi

echo "=============================================================="
echo

# Check if multi-GPU mode should be enabled (automatically when GPUs > 1)
if [[ "$GPUS_PER_NODE" -gt 1 ]]; then
    # Redirect to multi-GPU scripts
    print_status "Multi-GPU mode enabled (${GPUS_PER_NODE} GPUs) - delegating to multi-GPU launcher"
    
    # Build arguments to pass through
    MULTI_GPU_ARGS=""
    
    if [[ -n "$MODEL_PATH" ]]; then
        MULTI_GPU_ARGS="$MULTI_GPU_ARGS --model-path '$MODEL_PATH'"
    fi
    
    if [[ -n "$DCP_PATH" ]]; then
        MULTI_GPU_ARGS="$MULTI_GPU_ARGS --dcp-path '$DCP_PATH'"
    fi
    
    if [[ -n "$BASE_MODEL" ]]; then
        MULTI_GPU_ARGS="$MULTI_GPU_ARGS --base-model '$BASE_MODEL'"
    fi
    
    if [[ -n "$ENGINE" ]]; then
        MULTI_GPU_ARGS="$MULTI_GPU_ARGS --engine '$ENGINE'"
    fi
    
    if [[ -n "$ALGORITHM" ]]; then
        MULTI_GPU_ARGS="$MULTI_GPU_ARGS --algorithm '$ALGORITHM'"
    fi
    
    # Pass through server info file path
    if [[ -n "$SERVER_INFO_FILE" ]]; then
        MULTI_GPU_ARGS="$MULTI_GPU_ARGS --server-info-file '$SERVER_INFO_FILE'"
    fi
    
    MULTI_GPU_ARGS="$MULTI_GPU_ARGS --num-gpus $GPUS_PER_NODE"
    MULTI_GPU_ARGS="$MULTI_GPU_ARGS --port $PORT"
    MULTI_GPU_ARGS="$MULTI_GPU_ARGS --batch-size $BATCH_SIZE"
    MULTI_GPU_ARGS="$MULTI_GPU_ARGS --max-wait-time $MAX_WAIT_TIME"
    
    if [[ "$VERBOSE" == true ]]; then
        MULTI_GPU_ARGS="$MULTI_GPU_ARGS --verbose"
    fi
    
    if [[ "$NO_CHAT_TEMPLATE" == true ]]; then
        MULTI_GPU_ARGS="$MULTI_GPU_ARGS --no-chat-template"
    fi
    
    # Determine which multi-GPU script to use
    if [[ "$LOCAL_MODE" == true ]]; then
        # Local multi-GPU mode
        MULTI_GPU_SCRIPT="$SCRIPT_DIR/start_multi_gpu_server.sh"
        print_status "Launching local multi-GPU server with $GPUS_PER_NODE GPUs"
        eval "$MULTI_GPU_SCRIPT $MULTI_GPU_ARGS"
    else
        # SLURM multi-GPU mode
        MULTI_GPU_SCRIPT="$SCRIPT_DIR/start_multi_gpu_slurm.sh"
        
        # Add SLURM-specific args
        MULTI_GPU_ARGS="$MULTI_GPU_ARGS --job-name '$JOB_NAME'"
        MULTI_GPU_ARGS="$MULTI_GPU_ARGS --time '$TIME'"
        MULTI_GPU_ARGS="$MULTI_GPU_ARGS --cpus $CPUS_PER_TASK"
        MULTI_GPU_ARGS="$MULTI_GPU_ARGS --mem '$MEM'"
        MULTI_GPU_ARGS="$MULTI_GPU_ARGS --partition '$PARTITION'"
        MULTI_GPU_ARGS="$MULTI_GPU_ARGS --container '$CONTAINER_IMAGE'"
        
        print_status "Launching SLURM multi-GPU server with $GPUS_PER_NODE GPUs"
        eval "$MULTI_GPU_SCRIPT $MULTI_GPU_ARGS"
    fi
    
    exit 0
fi

# Execute based on mode (single GPU)
if [[ "$LOCAL_MODE" == true ]]; then
    run_local
else
    run_slurm
fi
