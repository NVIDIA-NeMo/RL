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
TIME="01:00:00"
GPUS_PER_NODE=1
CPUS_PER_TASK=16
MEM="64G"
PARTITION="interactive"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/mfathi/containers/nemo_rl_base.sqsh"
PROJECT_DIR=$(pwd) # Capture the current working directory
KERNEL_NAME="slurm-job-kernel-mfathi"
VENV_DIR=".venv"

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
VENV_DIR=".venv"
KERNEL_NAME="slurm-job-kernel-mfathi"

# Clean up previous environment to ensure a fresh start
echo "--> Removing old virtual environment..."
rm -rf "$VENV_DIR"

echo "===================================================================="
echo "Job running on compute node: $(hostname)"
echo "Virtual Environment will be set up in: $(pwd)/${VENV_DIR}"
echo "===================================================================="

# Step 1: Set up Python virtual environment using uv
echo
echo "[1/4] Setting up Python virtual environment with uv..."
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating new virtual environment with uv..."
    uv venv $VENV_DIR
fi
source $VENV_DIR/bin/activate
echo "Virtual environment activated."
echo

# Step 2: Prepare environment from uv.lock (+ vllm extra)
echo "[2/4] Preparing Python environment from uv.lock (+ vllm extra)..."
VIRTUAL_ENV="$VENV_DIR" uv sync --locked --extra vllm --no-install-project
if [ $? -ne 0 ]; then
    echo "Error: Failed to sync dependencies from uv.lock. Exiting."
    exit 1
fi

# Notebook-only deps not in uv.lock
VIRTUAL_ENV="$VENV_DIR" uv pip install jupyterlab ipykernel sentencepiece pandas matplotlib
if [ $? -ne 0 ]; then
    echo "Error: Failed to install notebook dependencies. Exiting."
    exit 1
fi
echo "Dependencies installed successfully."
echo

# Step 3: Register the virtual environment as a Jupyter kernel
echo "[3/4] Registering virtual environment as a Jupyter kernel..."
python -m ipykernel install --user --name="$KERNEL_NAME" --display-name="SLURM Job Kernel ($USER)"
echo "Jupyter kernel '$KERNEL_NAME' registered."
echo

# Step 4: Prepare and start Jupyter Lab
echo "[4/4] Starting Jupyter Lab server..."
PORT=$(shuf -i 8000-9999 -n 1)
TOKEN=$(openssl rand -hex 16)
COMPUTE_NODE=$(hostname)

echo
echo "==================== CONNECTION INSTRUCTIONS ===================="
echo
echo "----------[ 1. LOCAL TERMINAL: Create SSH Tunnel ]----------"
echo "Run this command on your LOCAL machine. It will seem to hang, which is normal."
echo
echo "   ssh -N -L ${PORT}:${COMPUTE_NODE}:${PORT} ${USER}@your_cluster_login_node"
echo
echo "   (Replace 'your_cluster_login_node' with your cluster's SSH address)"
echo "------------------------------------------------------------"
echo
echo "----------[ 2. VS CODE / BROWSER: Connect to Server ]----------"
echo "Use this URL to connect in your browser or in VS Code:"
echo "   (Ctrl+Shift+P -> 'Jupyter: Specify Jupyter server...' -> Paste URL)"
echo
echo "   http://localhost:${PORT}/lab?token=${TOKEN}"
echo
echo "Once connected, select the kernel: 'SLURM Job Kernel ($USER)'"
echo "================================================================"
echo

# Start Jupyter Lab, allowing it to run in the foreground
jupyter lab --no-browser --port=${PORT} --ip=0.0.0.0 --NotebookApp.token=${TOKEN} --allow-root
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
