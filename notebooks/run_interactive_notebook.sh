#!/bin/bash

# ===================================================================================
# Interactive Jupyter Notebook Launcher for SLURM
#
# This script uses srun to launch an interactive Jupyter Lab session inside a
# container. It handles virtual environment creation, dependency installation,
# and provides clear connection instructions.
#
# Usage:
#   1. Ensure $ACCOUNT and $LOG environment variables are set.
#   2. Run from your terminal: ./notebooks/run_interactive_notebook.sh
# ===================================================================================

# --- Job Configuration ---
JOB_NAME="interactive-notebook"
TIME="4:00:00"
GPUS_PER_NODE=1
CPUS_PER_TASK=16
MEM="64G"
PARTITION="interactive"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/mfathi/containers/nemo_rl_base.sqsh"
PROJECT_DIR=$(pwd) # Capture the current working directory
KERNEL_NAME="slurm-job-kernel-mfathi"
VENV_DIR="/opt/nemo_rl_venv"

# --- Validate Environment Variables ---
if [ -z "$ACCOUNT" ] || [ -z "$LOG" ]; then
    echo "Error: Please ensure the \$ACCOUNT and \$LOG environment variables are set."
    exit 1
fi
LOG_DIR="$LOG/notebooks"
mkdir -p "$LOG_DIR"


# --- srun Command Block ---
# This block defines the commands that will be executed on the compute node
# inside the container after the resources are allocated.
COMMAND_BLOCK=$(cat <<'EOF'
# Unset UV_CACHE_DIR to prevent conflicts with host cache
unset UV_CACHE_DIR

# --- Environment Setup on the Compute Node ---
export PATH="/root/.local/bin:$PATH"
VENV_DIR="/opt/nemo_rl_venv"
KERNEL_NAME="slurm-job-kernel-mfathi"

echo "===================================================================="
echo "Job running on compute node: $(hostname)"
echo "Using container's Python environment: ${VENV_DIR}"
echo "===================================================================="

# Step 1: Activate the container's existing Python environment
echo
echo "[1/3] Activating container's Python environment..."
if [ -f "$VENV_DIR/bin/activate" ]; then
    source $VENV_DIR/bin/activate
    echo "Container Python environment activated."
else
    echo "Warning: No activation script found at $VENV_DIR/bin/activate"
    echo "Proceeding with container's default Python environment..."
fi
echo

# Step 2: Prepare environment from uv.lock (+ vllm extra)
echo "[2/3] Syncing dependencies to container environment from uv.lock (+ vllm extra)..."
uv sync --locked --extra vllm --no-install-project
if [ $? -ne 0 ]; then
    echo "Error: Failed to sync dependencies from uv.lock. Exiting."
    exit 1
fi

# Notebook-only deps not in uv.lock
uv pip install jupyterlab ipykernel sentencepiece pandas matplotlib
if [ $? -ne 0 ]; then
    echo "Error: Failed to install notebook dependencies. Exiting."
    exit 1
fi
echo "Dependencies installed successfully."
echo

# Step 3: Register the container environment as a Jupyter kernel
echo "[3/3] Registering container environment as a Jupyter kernel..."
# Use the specific Python executable from the container environment
$VENV_DIR/bin/python -m ipykernel install --user --name="$KERNEL_NAME" --display-name="SLURM Job Kernel ($USER)"
echo "Jupyter kernel '$KERNEL_NAME' registered with container Python."

# Verify that the kernel is correctly pointing to our container environment
echo "Kernel Python executable: $($VENV_DIR/bin/python --version)"
echo "Kernel Python path: $($VENV_DIR/bin/python -c 'import sys; print(sys.executable)')"

# Verify key dependencies are available in the kernel
echo "Verifying key dependencies in kernel..."
$VENV_DIR/bin/python -c "
import sys
print(f'Python executable: {sys.executable}')
try:
    import torch; print(f'✓ PyTorch {torch.__version__}')
except ImportError as e: print(f'✗ PyTorch not found: {e}')
try:
    import vllm; print(f'✓ vLLM {vllm.__version__}')
except ImportError as e: print(f'✗ vLLM not found: {e}')
try:
    import transformers; print(f'✓ Transformers {transformers.__version__}')
except ImportError as e: print(f'✗ Transformers not found: {e}')
"
echo

# Step 4: Prepare and start Jupyter Lab
echo "Starting Jupyter Lab server..."
REQUESTED_PORT=$(shuf -i 8000-9999 -n 1)
TOKEN=$(openssl rand -hex 16)
COMPUTE_NODE=$(hostname)

echo "Requested port: ${REQUESTED_PORT}"
echo "Starting Jupyter Lab (it may choose a different port if ${REQUESTED_PORT} is busy)..."
echo

# Start Jupyter Lab and capture its output to detect the actual port
$VENV_DIR/bin/jupyter lab --no-browser --port=${REQUESTED_PORT} --ip=0.0.0.0 --NotebookApp.token=${TOKEN} --allow-root 2>&1 | while IFS= read -r line; do
    echo "$line"
    
    # Check if this line contains the actual server URL
    if [[ "$line" =~ "Jupyter Server".*"is running at:" ]]; then
        echo
        echo "========== JUPYTER LAB STARTED - UPDATING CONNECTION INFO =========="
    fi
    
    # Extract the actual port from the server URL line
    if [[ "$line" =~ "http://${COMPUTE_NODE}:"([0-9]+)"/lab" ]]; then
        ACTUAL_PORT="${BASH_REMATCH[1]}"
        echo
        echo "==================== UPDATED CONNECTION INSTRUCTIONS ===================="
        echo
        echo "----------[ 1. LOCAL TERMINAL: Create SSH Tunnel ]----------"
        echo "Run this command on your LOCAL machine. It will seem to hang, which is normal."
        echo
        echo "   ssh -N -L ${ACTUAL_PORT}:${COMPUTE_NODE}:${ACTUAL_PORT} ${USER}@your_cluster_login_node"
        echo
        echo "   (Replace 'your_cluster_login_node' with your cluster's SSH address)"
        echo "------------------------------------------------------------"
        echo
        echo "----------[ 2. VS CODE / BROWSER: Connect to Server ]----------"
        echo "Use this URL to connect in your browser or in VS Code:"
        echo "   (Ctrl+Shift+P -> 'Jupyter: Specify Jupyter server...' -> Paste URL)"
        echo
        echo "   http://localhost:${ACTUAL_PORT}/lab?token=${TOKEN}"
        echo
        echo "Once connected, select the kernel: 'SLURM Job Kernel ($USER)'"
        echo "================================================================"
        echo
    fi
done
EOF
)


# --- Launch the Interactive Job ---
echo "Requesting interactive job allocation from SLURM..."

srun --job-name=${JOB_NAME} \
     --time=${TIME} \
     --gpus-per-node=${GPUS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_TASK} \
     --mem=${MEM} \
     --partition=${PARTITION} \
     --account=${ACCOUNT} \
     --no-container-mount-home \
     --container-image=${CONTAINER_IMAGE} \
     --container-workdir=${PROJECT_DIR} \
     --container-mounts=${PROJECT_DIR}:${PROJECT_DIR} \
     --output="${LOG_DIR}/notebook_job_%j.log" \
     bash -c "$COMMAND_BLOCK"

echo "Interactive job finished."
