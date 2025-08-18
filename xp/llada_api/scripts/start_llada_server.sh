#!/bin/bash

# LLaDA OpenAI API Server Startup Script
# This script provides easy ways to start the LLaDA server with different configurations.
# Can run locally (--local) or as a SLURM job (default).

set -e

# Default values - LLaDA Server
HOST="0.0.0.0"
PORT=8000
MODEL_PATH=""
DCP_PATH=""
BASE_MODEL="GSAI-ML/LLaDA-8B-Instruct"
TEMP_DIR="/tmp/llada_hf_converted"

# Default values - SLURM
LOCAL_MODE=false
JOB_NAME="llada-api-server"
TIME="4:00:00"
GPUS_PER_NODE=1
CPUS_PER_TASK=16
MEM="64G"
PARTITION="interactive"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/mfathi/containers/nemo_rl_base.sqsh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
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
    echo "Execution Mode:"
    echo "  --local                 Run locally (default: run as SLURM job)"
    echo ""
    echo "SLURM Job Options (ignored with --local):"
    echo "  --job-name NAME         SLURM job name (default: $JOB_NAME)"
    echo "  --time TIME             Job time limit (default: $TIME)"
    echo "  --gpus NUM              GPUs per node (default: $GPUS_PER_NODE)"
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
    echo "  # Local with HuggingFace model (local path)"
    echo "  $0 --local --model-path /path/to/llada-8b-instruct"
    echo ""
    echo "  # Local with HuggingFace model (from Hub)"
    echo "  $0 --local --model-path GSAI-ML/LLaDA-8B-Instruct"
    echo ""
    echo "  # SLURM job with DCP checkpoint"
    echo "  export ACCOUNT=your_account"
    echo "  $0 --dcp-path /path/to/checkpoint.dcp --base-model GSAI-ML/LLaDA-8B-Instruct"
    echo ""
    echo "  # SLURM job with HuggingFace model from Hub"
    echo "  $0 --model-path GSAI-ML/LLaDA-8B-Instruct --gpus 2 --mem 128G --time 8:00:00"
    echo ""
    echo "  # Local with custom host/port"
    echo "  $0 --local --model-path /path/to/model --host 127.0.0.1 --port 8080"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
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
        # Execution mode
        --local)
            LOCAL_MODE=true
            shift 1
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
    
    # Note: Logs will be output to stdout instead of files
    print_status "Logs will be output directly to stdout for real-time viewing"
fi

# Get the absolute path to the Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLADA_API_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_PATH="$LLADA_API_DIR/llada_openai_server.py"
PROJECT_DIR="$(cd "$LLADA_API_DIR/../.." && pwd)"  # NeMo-RL project root

if [[ ! -f "$SCRIPT_PATH" ]]; then
    print_error "LLaDA server script not found: $SCRIPT_PATH"
    exit 1
fi

# Build LLaDA server command arguments
LLADA_ARGS="--host '$HOST' --port '$PORT'"

if [[ -n "$MODEL_PATH" ]]; then
    # Check if it's a local path or HuggingFace model name
    if [[ -d "$MODEL_PATH" ]] || [[ -f "$MODEL_PATH" ]]; then
        print_status "Using local HuggingFace model: $MODEL_PATH"
        MODEL_TYPE="local path"
    else
        print_status "Using HuggingFace model name: $MODEL_PATH"
        MODEL_TYPE="HuggingFace Hub"
        # For HuggingFace model names, we don't need to check if the path exists locally
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
        print_warning "Make sure you have run: uv sync --locked --extra vllm --no-install-project"
        print_warning "For easier local testing, consider using: --model-path GSAI-ML/LLaDA-8B-Instruct"
    fi
    
    LLADA_ARGS="$LLADA_ARGS --dcp-path '$DCP_PATH' --base-model '$BASE_MODEL' --temp-dir '$TEMP_DIR'"
fi

# Function to run locally
run_local() {
    print_status "Running LLaDA server locally"
    
    # Check for required Python packages
    print_status "Checking Python dependencies..."
    export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
    
    python3 -c "import fastapi, uvicorn, torch, transformers" 2>/dev/null || {
        print_error "Missing required Python packages. Please install using uv:"
        echo "  uv sync --locked --extra vllm --no-install-project  # Sync from uv.lock"
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

    # Show server info
    print_status "Server configuration:"
    echo "  • Host: $HOST"
    echo "  • Port: $PORT"
    echo "  • API Base URL: http://$HOST:$PORT"
    echo "  • Health check: http://$HOST:$PORT/health"
    echo "  • API docs: http://$HOST:$PORT/docs"

    # Set up PYTHONPATH for local execution to include NeMo-RL
    export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
    print_status "Set PYTHONPATH to include NeMo-RL project: $PROJECT_DIR"

    # Build and execute command
    CMD="python3 '$SCRIPT_PATH' $LLADA_ARGS"
    print_status "Starting LLaDA OpenAI API server..."
    echo "Command: $CMD"
    echo ""
    print_status "Press Ctrl+C to stop the server"
    echo "=================================================="

    eval "$CMD"
}

# Function to run on SLURM
run_slurm() {
    print_status "Submitting LLaDA server as SLURM job"
    
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
    print_status "Server configuration:"
    echo "  • Host: $HOST"
    echo "  • Port: $PORT"
    echo "  • Compute node URL: http://\$SLURMD_NODENAME:$PORT (from within job)"
    echo "  • Local access: http://localhost:$PORT (via SSH tunnel)"
    
    print_status "Connection setup:"
    echo "  • The server will display SSH tunnel commands when it starts"
    echo "  • Connection instructions will appear in the terminal output"
    echo "  • Server logs will be shown directly in stdout"

    # Create the command block that will run inside the container
    COMMAND_BLOCK=$(cat <<EOF
# Unset UV_CACHE_DIR to prevent conflicts with host cache
unset UV_CACHE_DIR

# Environment setup
export PATH="/root/.local/bin:\$PATH"
VENV_DIR="/opt/nemo_rl_venv"

echo "===================================================================="
echo "LLaDA OpenAI API Server starting on compute node: \$(hostname)"
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

# Step 2: Prepare environment from uv.lock (+ vllm extra)
echo "[2/3] Syncing dependencies to container environment from uv.lock (+ vllm extra)..."
uv sync --locked --extra vllm --no-install-project
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

echo "[3/3] Starting LLaDA OpenAI API server..."
COMPUTE_NODE=\$(hostname)
echo "Server starting on compute node: \$COMPUTE_NODE"
echo "=================================================="

# Start the server and provide connection info when it starts
\$VENV_DIR/bin/python "$SCRIPT_PATH" $LLADA_ARGS 2>&1 | while IFS= read -r line; do
    echo "\$line"
    
    # When we see the server startup message, show connection instructions  
    if [[ "\$line" =~ "Uvicorn running on".*":$PORT" ]]; then
        echo
        echo "========== LLADA API SERVER STARTED - CONNECTION INFO =========="
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
        echo
        echo "----------[ 3. TEST THE API ]----------"
        echo "Test with curl:"
        echo "   curl http://localhost:$PORT/health"
        echo
        echo "Or use the Python client:"
        echo "   python xp/llada_api/examples/llada_api_client.py"
        echo "=============================================================="
        echo
    fi
done
EOF
)

    # Submit the SLURM job
    print_status "Submitting SLURM job..."
    
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
         --container-mounts="$PROJECT_DIR:$PROJECT_DIR" \
         bash -c "$COMMAND_BLOCK"
}

# Execute based on mode
if [[ "$LOCAL_MODE" == true ]]; then
    run_local
else
    run_slurm
fi
