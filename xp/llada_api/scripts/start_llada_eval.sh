#!/bin/bash

set -euo pipefail

# LLaDA Evaluation SLURM Launcher
# Launches xp/nemo-skills/eval_llada.py inside the NeMo-RL container
# using SLURM (or locally with --local). Designed to pair with
# start_llada_batch_server.sh by reusing the generated server info file.

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

show_usage() {
    cat <<USAGE
Usage: $0 [OPTIONS] [-- EVAL_ARGS...]

Launch the LLaDA evaluation script (xp/nemo-skills/eval_llada.py) on SLURM
inside the NeMo-RL container. Script-specific options must appear before --.
Arguments after -- are passed directly to eval_llada.py.

Core options:
  -h, --help                 Show this help message and exit
  --local                    Run evaluation locally instead of via SLURM
  --server-address URL       Override server address (default: read from info file)
  --server-info-file PATH    Path to server info file written by start_llada_batch_server.sh
                             (default: xp/llada_api/.llada_server_info)
  --no-wait-for-server       Do not poll the server /health endpoint before running evaluation

SLURM options (ignored with --local):
  --job-name NAME            SLURM job name (default: llada-eval)
  --time TIME                Wall clock time limit (default: 2:00:00)
  --cpus NUM                 CPUs per task (default: 32)
  --mem SIZE                 Memory per node (default: 64G)
  --partition PART           SLURM partition (default: batch)
  --container IMAGE          Container image path (default matches server launcher)
  --account ACCOUNT          SLURM account (defaults to \$ACCOUNT environment variable)

Examples:
  # Launch evaluation with defaults, reading server info from generated file
  $0 -- --benchmark gsm8k:4 --quick-test

  # Explicit server info file and custom output directory
  $0 --server-info-file /path/to/info.env -- --output-dir results/gsm8k

  # Local dry run (no SLURM)
  $0 --local -- --dry-run --quick-test
USAGE
}

# Default configuration
LOCAL_MODE=false
JOB_NAME="llada-eval"
TIME="2:00:00"
CPUS_PER_TASK=32
MEM="64G"
PARTITION="batch"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/mfathi/containers/nemo_rl_base.sqsh"
SERVER_ADDRESS=""
SERVER_INFO_FILE=""
USER_PROVIDED_SERVER_ADDRESS=""
WAIT_FOR_SERVER=true
ACCOUNT_VALUE="${ACCOUNT:-}"
VERBOSE=false

EVAL_ARGS=()

# Parse arguments (script options before --, eval args after --)
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;
        --local)
            LOCAL_MODE=true
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
        --account)
            ACCOUNT_VALUE="$2"
            shift 2
            ;;
        --server-address)
            SERVER_ADDRESS="$2"
            USER_PROVIDED_SERVER_ADDRESS="$2"
            shift 2
            ;;
        --server-info-file)
            SERVER_INFO_FILE="$2"
            shift 2
            ;;
        --no-wait-for-server)
            WAIT_FOR_SERVER=false
            shift 1
            ;;
        --verbose)
            VERBOSE=true
            shift 1
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                EVAL_ARGS+=("$1")
                shift
            done
            ;;
        *)
            # Treat unknown options as eval_llada.py arguments
            EVAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Resolve directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLADA_API_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$LLADA_API_DIR/../.." && pwd)"
EVAL_SCRIPT="$PROJECT_DIR/xp/nemo-skills/eval_llada.py"

if [[ ! -f "$EVAL_SCRIPT" ]]; then
    print_error "Evaluation script not found at $EVAL_SCRIPT"
    exit 1
fi

if [[ -z "$SERVER_INFO_FILE" ]]; then
    SERVER_INFO_FILE="$LLADA_API_DIR/.llada_server_info"
fi

SERVER_INFO_FILE="$(realpath -m "$SERVER_INFO_FILE")"

if [[ ! -f "$SERVER_INFO_FILE" ]]; then
    if [[ -z "$USER_PROVIDED_SERVER_ADDRESS" ]]; then
        print_error "Server info file not found: $SERVER_INFO_FILE"
        print_error "Run start_llada_batch_server.sh first, or provide --server-address explicitly."
        exit 1
    else
        print_warning "Server info file not found: $SERVER_INFO_FILE (using --server-address override)"
    fi
fi

# Load server info if available
if [[ -f "$SERVER_INFO_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$SERVER_INFO_FILE"
fi

# Wait for SLURM job to publish usable server details unless overridden
if [[ -z "$USER_PROVIDED_SERVER_ADDRESS" && -f "$SERVER_INFO_FILE" ]]; then
    NEEDS_INFO_WAIT=false
    if [[ "${SERVER_INFO_SOURCE:-}" == "slurm" ]]; then
        if [[ "${SERVER_STATUS:-}" != "running" || -z "${SERVER_ADDRESS:-}" || "${SERVER_ADDRESS:-}" == *"0.0.0.0"* || -z "${SERVER_CLIENT_HOST:-}" || "${SERVER_CLIENT_HOST:-}" == "0.0.0.0" ]]; then
            NEEDS_INFO_WAIT=true
        fi
    elif [[ -z "${SERVER_ADDRESS:-}" ]]; then
        NEEDS_INFO_WAIT=true
    fi

    if [[ "$NEEDS_INFO_WAIT" == true ]]; then
        print_status "Waiting for SLURM server job to publish connection details..."
        INFO_WAIT_INTERVAL=5
        INFO_WAIT_TIMEOUT=900
        INFO_WAIT_ELAPSED=0
        INFO_WAIT_ITERATIONS=0
        while true; do
            sleep "$INFO_WAIT_INTERVAL"
            INFO_WAIT_ELAPSED=$((INFO_WAIT_ELAPSED + INFO_WAIT_INTERVAL))
            INFO_WAIT_ITERATIONS=$((INFO_WAIT_ITERATIONS + 1))
            # shellcheck disable=SC1090
            source "$SERVER_INFO_FILE"
            if [[ "${SERVER_STATUS:-}" == "running" && -n "${SERVER_ADDRESS:-}" && "${SERVER_ADDRESS:-}" != *"0.0.0.0"* ]]; then
                if [[ -n "${SERVER_CLIENT_HOST:-}" && "${SERVER_CLIENT_HOST:-}" != "0.0.0.0" ]]; then
                    break
                fi
            fi
            # Print progress every minute (12 iterations)
            if (( INFO_WAIT_ITERATIONS == 1 || INFO_WAIT_ITERATIONS % 12 == 0 )); then
                print_status "Still waiting... (${INFO_WAIT_ELAPSED}s elapsed, status=${SERVER_STATUS:-unset}, client_host=${SERVER_CLIENT_HOST:-unset})"
            fi
            if (( INFO_WAIT_ELAPSED >= INFO_WAIT_TIMEOUT )); then
                print_error "Timed out waiting for server info update in $SERVER_INFO_FILE"
                print_error "Last status: SERVER_STATUS=${SERVER_STATUS:-unset}, SERVER_ADDRESS=${SERVER_ADDRESS:-unset}, SERVER_CLIENT_HOST=${SERVER_CLIENT_HOST:-unset}"
                exit 1
            fi
        done
        print_status "Server info file updated (status: ${SERVER_STATUS:-unknown})."
    fi
fi

# Preserve explicit server address if provided
if [[ -n "$USER_PROVIDED_SERVER_ADDRESS" ]]; then
    SERVER_ADDRESS="$USER_PROVIDED_SERVER_ADDRESS"
fi

if [[ -z "${SERVER_ADDRESS:-}" ]]; then
    if [[ -n "${SERVER_BASE_URL:-}" ]]; then
        SERVER_ADDRESS="${SERVER_BASE_URL}/v1"
    else
        print_error "Server address could not be determined. Provide --server-address or update the server info file."
        exit 1
    fi
fi

SERVER_BASE_URL_FROM_ADDRESS="${SERVER_ADDRESS%/v1}"
if [[ "$SERVER_BASE_URL_FROM_ADDRESS" == "$SERVER_ADDRESS" ]]; then
    SERVER_BASE_URL_FROM_ADDRESS="$SERVER_ADDRESS"
fi

if [[ -z "${SERVER_HEALTH_URL:-}" ]]; then
    SERVER_HEALTH_URL="${SERVER_BASE_URL_FROM_ADDRESS}/health"
fi

if [[ -z "${SERVER_MODE:-}" ]]; then
    SERVER_MODE="batch"
fi

if [[ -z "${SERVER_PORT:-}" ]]; then
    # Attempt to extract port from address
    SERVER_PORT="$(echo "$SERVER_ADDRESS" | sed -E 's#.*:([0-9]+)/?.*#\1#')"
fi

print_status "Server address: $SERVER_ADDRESS"
print_status "Server health endpoint: $SERVER_HEALTH_URL"

if [[ -f "$SERVER_INFO_FILE" ]]; then
    print_status "Using server info file: $SERVER_INFO_FILE"
fi

# Append --server-address to evaluation args if not already provided
SERVER_ADDRESS_ARG_PRESENT=false
for ((i=0; i<${#EVAL_ARGS[@]}; i++)); do
    if [[ "${EVAL_ARGS[$i]}" == "--server-address" ]]; then
        SERVER_ADDRESS_ARG_PRESENT=true
        break
    fi
done

if [[ "$SERVER_ADDRESS_ARG_PRESENT" == false ]]; then
    EVAL_ARGS+=("--server-address" "$SERVER_ADDRESS")
fi

# Prepare evaluation arguments string for logging/execution
EVAL_ARGS_SERIALIZED=""
for arg in "${EVAL_ARGS[@]}"; do
    EVAL_ARGS_SERIALIZED+=" $(printf "%q" "$arg")"
done

# Detect --output-dir for container mounts
EVAL_OUTPUT_DIR=""
for ((i=0; i<${#EVAL_ARGS[@]}; i++)); do
    if [[ "${EVAL_ARGS[$i]}" == "--output-dir" ]]; then
        if (( i + 1 < ${#EVAL_ARGS[@]} )); then
            EVAL_OUTPUT_DIR="${EVAL_ARGS[$((i + 1))]}"
        fi
        break
    fi
done

if [[ -n "$EVAL_OUTPUT_DIR" ]]; then
    # Resolve relative paths against current working directory
    if [[ "$EVAL_OUTPUT_DIR" != /* ]]; then
        EVAL_OUTPUT_DIR="$(realpath -m "$EVAL_OUTPUT_DIR")"
    else
        EVAL_OUTPUT_DIR="$(realpath -m "$EVAL_OUTPUT_DIR")"
    fi
    print_status "Evaluation outputs will be written to: $EVAL_OUTPUT_DIR"
fi

if [[ "$LOCAL_MODE" == true ]]; then
    print_status "Running evaluation locally (no SLURM)"
    export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
    print_status "Checking Python dependencies..."
    uv sync --locked --no-install-project --extra eval
    if [[ "$WAIT_FOR_SERVER" == true ]]; then
        if command -v curl >/dev/null 2>&1; then
            print_status "Waiting for server health at $SERVER_HEALTH_URL ..."
            ATTEMPT=0
            TIMEOUT=600
            INTERVAL=5
            until curl -sf --connect-timeout 2 "$SERVER_HEALTH_URL" >/dev/null; do
                ATTEMPT=$((ATTEMPT + 1))
                ELAPSED=$((ATTEMPT * INTERVAL))
                if (( ELAPSED >= TIMEOUT )); then
                    print_error "Server did not respond at $SERVER_HEALTH_URL within $TIMEOUT seconds."
                    exit 1
                fi
                sleep "$INTERVAL"
            done
            print_status "Server responded successfully."
        else
            print_warning "curl not installed; skipping health check."
        fi
    fi
    CMD="python3 '$EVAL_SCRIPT'$EVAL_ARGS_SERIALIZED"
    print_status "Executing: $CMD"
    eval "$CMD"
    exit 0
fi

# SLURM execution path
if [[ -z "$ACCOUNT_VALUE" ]]; then
    print_error "ACCOUNT environment variable is not set. Provide --account or export ACCOUNT."
    exit 1
fi

CONTAINER_MOUNTS="$PROJECT_DIR:$PROJECT_DIR"

# Mount output directory if it exists and is outside the project tree
if [[ -n "$EVAL_OUTPUT_DIR" ]]; then
    if [[ -d "$EVAL_OUTPUT_DIR" || ! -e "$EVAL_OUTPUT_DIR" ]]; then
        mkdir -p "$EVAL_OUTPUT_DIR"
    fi
    if [[ "$EVAL_OUTPUT_DIR" != "$PROJECT_DIR"* ]]; then
        CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$EVAL_OUTPUT_DIR:$EVAL_OUTPUT_DIR"
        print_status "Mounted output directory for container: $EVAL_OUTPUT_DIR"
    fi
fi

print_status "Submitting SLURM evaluation job"
echo "  • Job name: $JOB_NAME"
echo "  • Time limit: $TIME"
echo "  • CPUs per task: $CPUS_PER_TASK"
echo "  • Memory: $MEM"
echo "  • Partition: $PARTITION"
echo "  • Account: $ACCOUNT_VALUE"
echo "  • Container: $CONTAINER_IMAGE"
echo "  • Server: $SERVER_ADDRESS"

if [[ "$WAIT_FOR_SERVER" == true ]]; then
    print_status "Server readiness check enabled (health URL: $SERVER_HEALTH_URL)"
else
    print_warning "Server readiness check disabled (--no-wait-for-server)"
fi

COMMAND_BLOCK=$(cat <<EOF
set -euo pipefail

unset UV_CACHE_DIR
export PATH="/root/.local/bin:\$PATH"
export PYTHONPATH="$PROJECT_DIR:\${PYTHONPATH:-}"
VENV_DIR="/opt/nemo_rl_venv"

echo "===================================================================="
echo "LLaDA Evaluation job starting on node: \$(hostname)"
echo "Using Python environment at: \${VENV_DIR}"
echo "===================================================================="

if [ -f "\$VENV_DIR/bin/activate" ]; then
    source "\$VENV_DIR/bin/activate"
else
    echo "[WARN] No activation script at \$VENV_DIR/bin/activate - using default environment"
fi

echo "[1/3] Syncing dependencies from uv.lock..."
uv sync --locked --no-install-project --extra eval
echo "[2/3] Dependencies ready."

SERVER_HEALTH_URL="$SERVER_HEALTH_URL"
WAIT_FOR_SERVER="$WAIT_FOR_SERVER"
if [[ "\$WAIT_FOR_SERVER" == "true" ]]; then
    if command -v curl >/dev/null 2>&1; then
        echo "[Eval] Waiting for server health at \$SERVER_HEALTH_URL ..."
        ATTEMPT=0
        TIMEOUT=900
        INTERVAL=5
        until curl -sf --connect-timeout 2 "\$SERVER_HEALTH_URL" >/dev/null; do
            ATTEMPT=\$((ATTEMPT + 1))
            ELAPSED=\$((ATTEMPT * INTERVAL))
            if [[ \$ELAPSED -ge \$TIMEOUT ]]; then
                echo "[Eval] ERROR: Server did not respond within \$TIMEOUT seconds."
                exit 1
            fi
            sleep \$INTERVAL
        done
        echo "[Eval] Server is ready."
    else
        echo "[Eval] WARN: curl not available; skipping server health check."
    fi
else
    echo "[Eval] Skipping server health check (--no-wait-for-server)."
fi

echo "[3/3] Running evaluation script..."
CMD="\$VENV_DIR/bin/python '$EVAL_SCRIPT'$EVAL_ARGS_SERIALIZED"
echo "[Eval] Command: \$CMD"
eval "\$CMD"
EOF
)

srun --job-name="$JOB_NAME" \
     --time="$TIME" \
     --cpus-per-task="$CPUS_PER_TASK" \
     --mem="$MEM" \
     --partition="$PARTITION" \
     --account="$ACCOUNT_VALUE" \
     --container-image="$CONTAINER_IMAGE" \
     --container-workdir="$PROJECT_DIR" \
     --container-mounts="$CONTAINER_MOUNTS" \
     bash -lc "$COMMAND_BLOCK"


