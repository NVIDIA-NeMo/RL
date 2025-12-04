#!/bin/bash
#SBATCH -A coreai_dlalgo_nemorl
#SBATCH -J vllm-offline-bench
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node 1
#SBATCH -o /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/vllm_standalone_perf_exp/offline/%j-logs/slurm-%j.out
#SBATCH -e /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/vllm_standalone_perf_exp/offline/%j-logs/slurm-%j.err

set -ex

# ============================================================
# HuggingFace Configuration
# ============================================================
export HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}
export HF_TOKEN=${HF_TOKEN:-}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}

# ============================================================
# Configuration (can be overridden via environment variables)
# ============================================================

# Model path - can be HuggingFace model name or local path
# Examples:
#   MODEL_PATH=Qwen/Qwen2.5-32B-Instruct          # Dense model
#   MODEL_PATH=Qwen/Qwen3-30B-A3B                 # MoE model (30B params, 3B active)
#   MODEL_PATH=Qwen/Qwen3-235B-A22B               # Large MoE model
#   MODEL_PATH=deepseek-ai/DeepSeek-V3           # DeepSeek MoE
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-32B-Instruct}

# Cluster configuration
# GB200: 4 GPUs/node, H100: 8 GPUs/node
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
GPU_MODEL=${GPU_MODEL:-"unknown"}

# Parallelism
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
EXPERT_PARALLEL_SIZE=${EXPERT_PARALLEL_SIZE:-1}

# MoE settings (for DeepSeek, Mixtral, Qwen3-MoE, etc.)
# Set ENABLE_EXPERT_PARALLEL=1 for MoE models
ENABLE_EXPERT_PARALLEL=${ENABLE_EXPERT_PARALLEL:-0}

# GRPO-style batch configuration
NUM_PROMPTS=${NUM_PROMPTS:-64}
NUM_GENERATIONS=${NUM_GENERATIONS:-32}

# Generation parameters
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_TOKENS=${MAX_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-1.0}

# Prompts file (prepared by prepare_prompts.py)
PROMPTS_FILE=${PROMPTS_FILE:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/prompts.json}

# If no prompts file, use random input
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-150}

# Container and paths
# Script directory and parent directory
SCRIPT_DIR=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/vllm_benchmark
BASE_DIR=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl
CONTAINER_IMAGE=${CONTAINER_IMAGE:-vllm/vllm-openai:nightly-aarch64}
CONTAINER_NAME=vllm-offline-bench
JOB_CACHE_DIR=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/job_cache

# Log directory (separated by benchmark type)
EXPERIMENT_DIR=${EXPERIMENT_DIR:-$BASE_DIR/vllm_standalone_perf_exp}
LOGS_DIR=${LOGS_DIR:-$EXPERIMENT_DIR/offline/${SLURM_JOB_ID}-logs}

# venv for custom installations (persisted across jobs)
VLLM_VENV_DIR=${VLLM_VENV_DIR:-$BASE_DIR/vllm-benchmark-venv}

# vLLM installation options
# Set VLLM_INSTALL_FROM_SOURCE=1 to install from GitHub
# Set VLLM_LOCAL_PATH to install from local directory (editable mode)
# Set DO_BUILD=1 to rebuild the venv (run once, then reuse)
VLLM_INSTALL_FROM_SOURCE=${VLLM_INSTALL_FROM_SOURCE:-0}
VLLM_LOCAL_PATH=${VLLM_LOCAL_PATH:-}
DO_BUILD=${DO_BUILD:-0}
VLLM_GIT_REPO=${VLLM_GIT_REPO:-https://github.com/vllm-project/vllm.git}
VLLM_GIT_BRANCH=${VLLM_GIT_BRANCH:-main}

# ============================================================
# Setup
# ============================================================

mkdir -p $EXPERIMENT_DIR
mkdir -p $LOGS_DIR
mkdir -p $JOB_CACHE_DIR

# ============================================================
# Build venv (Duncan-style DO_BUILD)
# Run once with DO_BUILD=1 to create persistent venv
# ============================================================
if [ "$DO_BUILD" == "1" ]; then
    echo "============================================================"
    echo "Building venv with custom vLLM installation..."
    echo "============================================================"
    
    srun -N 1 --container-image="$CONTAINER_IMAGE" \
        --container-mounts="/lustre:/lustre,$JOB_CACHE_DIR:/root/.cache" \
        bash -c "
set -ex
# Create venv with system site-packages (includes vLLM from container)
python3 -m venv --system-site-packages $VLLM_VENV_DIR
source $VLLM_VENV_DIR/bin/activate

# Install huggingface-hub to temp directory (avoid affecting login node)
# Note: huggingface-hub must be <1.0 for compatibility with transformers in this container
HF_FIX_DIR=/tmp/hf_fix_\$\$
pip install --no-cache-dir --target=\$HF_FIX_DIR huggingface-hub==0.34
export PYTHONPATH=\$HF_FIX_DIR:\$PYTHONPATH


# Install custom vLLM if requested
if [ -n '$VLLM_LOCAL_PATH' ] && [ -d '$VLLM_LOCAL_PATH' ]; then
    echo 'Installing vLLM from local path (editable): $VLLM_LOCAL_PATH'
    pip install -e '$VLLM_LOCAL_PATH'
elif [ '$VLLM_INSTALL_FROM_SOURCE' == '1' ]; then
    echo 'Installing vLLM from: $VLLM_GIT_REPO@$VLLM_GIT_BRANCH'
    pip install git+$VLLM_GIT_REPO@$VLLM_GIT_BRANCH
fi

echo 'venv created at: $VLLM_VENV_DIR'
echo ''
echo '============================================================'
echo 'Installed package versions:'
echo '============================================================'
pip show vllm transformers huggingface-hub torch | grep -E '^(Name|Version):'
echo ''
echo '============================================================'
python -c 'import vllm; print(f\"vLLM version: {vllm.__version__}\")'
python -c 'import transformers; print(f\"Transformers version: {transformers.__version__}\")'
python -c 'import huggingface_hub; print(f\"HuggingFace Hub version: {huggingface_hub.__version__}\")'
python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\")'
"
    echo "Build complete! Now run without DO_BUILD=1"
    exit 0
fi

NUM_GPUS_PER_NODE=4
if [[ `hostname` == *"eos"* ]]; then
  NUM_GPUS_PER_NODE=8
fi

NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
NUM_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))

# Calculate Data Parallelism degree
# DP = Total GPUs / (TP × PP)
# Note: EP (Expert Parallelism) works within TP, doesn't require additional GPUs
GPUS_PER_INSTANCE=$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))
DATA_PARALLEL_SIZE=$((NUM_GPUS / GPUS_PER_INSTANCE))

# Validate DP calculation
if [ $DATA_PARALLEL_SIZE -lt 1 ]; then
    echo "ERROR: Not enough GPUs for TP=$TENSOR_PARALLEL_SIZE × PP=$PIPELINE_PARALLEL_SIZE"
    echo "Total GPUs: $NUM_GPUS, Required: $GPUS_PER_INSTANCE"
    exit 1
fi

if [ $((DATA_PARALLEL_SIZE * GPUS_PER_INSTANCE)) -ne $NUM_GPUS ]; then
    echo "WARNING: GPU count ($NUM_GPUS) not evenly divisible by TP×PP ($GPUS_PER_INSTANCE)"
    echo "Some GPUs may be unused. DP=$DATA_PARALLEL_SIZE will use $((DATA_PARALLEL_SIZE * GPUS_PER_INSTANCE)) GPUs"
fi

echo "============================================================"
echo "vLLM Offline Throughput Benchmark (Duncan-style)"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Nodes: $NUM_NODES"
echo "GPUs per node: $NUM_GPUS_PER_NODE"
echo "Total GPUs: $NUM_GPUS"
if [ $EXPERT_PARALLEL_SIZE -gt 1 ]; then
    echo "Parallelism: TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE, DP=$DATA_PARALLEL_SIZE, EP=$EXPERT_PARALLEL_SIZE (MoE)"
else
    echo "Parallelism: TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE, DP=$DATA_PARALLEL_SIZE"
fi
echo "GPUs per vLLM instance: $GPUS_PER_INSTANCE"
echo "GRPO-style: $NUM_PROMPTS prompts × $NUM_GENERATIONS generations"
echo "Logs: $LOGS_DIR"
echo "============================================================"

# Get head node IP
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Handle IPv6
if [[ "$head_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<< "$head_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_ip=${ADDR[1]}
  else
    head_ip=${ADDR[0]}
  fi
fi

export RAY_ADDRESS=$head_ip:6379

echo "Head node: $head_node ($head_ip)"

# ============================================================
# vLLM Environment Variables 
# ============================================================
export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# MoE-specific environment variables (for DeepSeek, Mixtral, Qwen3-MoE, etc.)
if [ "$ENABLE_EXPERT_PARALLEL" == "1" ] || [ $EXPERT_PARALLEL_SIZE -gt 1 ]; then
    echo "MoE mode enabled (EP=$EXPERT_PARALLEL_SIZE)"
    export VLLM_FLASHINFER_MOE_BACKEND=${VLLM_FLASHINFER_MOE_BACKEND:-throughput}
    # For FP8 MoE models
    # export VLLM_USE_FLASHINFER_MOE_FP8=1
    # For FP4 MoE models  
    # export VLLM_USE_FLASHINFER_MOE_FP4=1
fi

# ============================================================
# Start Ray workers on non-head nodes
# ============================================================
worker_nodes=("${nodes_array[@]:1}")

if [ ${#worker_nodes[@]} -gt 0 ]; then
    echo "Starting Ray workers on: ${worker_nodes[*]}"
    
    srun --nodes=$((NUM_NODES - 1)) --ntasks-per-node=1 \
        --container-image="$CONTAINER_IMAGE" \
        --container-mounts="/lustre:/lustre,$JOB_CACHE_DIR:/root/.cache" \
        --container-name="$CONTAINER_NAME" \
        --nodelist="${worker_nodes[*]}" \
        --mpi=pmix \
        bash -c "
set -ex

# Install huggingface-hub to temp directory (avoid affecting login node)
HF_FIX_DIR=/tmp/hf_fix_\$\$
pip install --quiet --no-cache-dir --target=\$HF_FIX_DIR huggingface-hub==0.34
export PYTHONPATH=\$HF_FIX_DIR:\$PYTHONPATH

# Activate venv if it exists (from DO_BUILD)
if [ -d '$VLLM_VENV_DIR' ]; then
    source $VLLM_VENV_DIR/bin/activate
    echo 'Using venv: $VLLM_VENV_DIR'
fi

# Wait for head node Ray to start first (container image download + ray start)
echo 'Waiting for head node Ray to start...'
sleep 60

# Retry connecting to head node
for i in 1 2 3 4 5; do
    echo \"Attempt \$i: Connecting to Ray head at $head_ip:6379...\"
    ray start --address '$head_ip:6379' --min-worker-port=20000 --max-worker-port=29999 && break
    echo \"Failed, retrying in 15 seconds...\"
    sleep 15
done

sleep infinity
" &
    worker_pid=$!
    
    trap "kill $worker_pid 2>/dev/null || true" EXIT
    sleep 20
fi

# ============================================================
# Run benchmark on head node
# ============================================================
srun --nodes=1 --ntasks-per-node=1 -w $head_node \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="/lustre:/lustre,$JOB_CACHE_DIR:/root/.cache" \
    --container-name="$CONTAINER_NAME" \
    --mpi=pmix \
    bash -c "
set -ex

# Install huggingface-hub to temp directory (avoid affecting login node)
# Container has 1.1.4, but transformers needs <1.0
echo 'Fixing huggingface-hub version...'
HF_FIX_DIR=/tmp/hf_fix_\$\$
pip install --quiet --no-cache-dir --target=\$HF_FIX_DIR huggingface-hub==0.34
export PYTHONPATH=\$HF_FIX_DIR:\$PYTHONPATH

# Activate venv if it exists (from DO_BUILD) - optional, for custom vLLM builds
if [ -d '$VLLM_VENV_DIR' ]; then
    source $VLLM_VENV_DIR/bin/activate
    echo 'Using venv: $VLLM_VENV_DIR'
fi

# Start Ray head only for multi-node (like Duncan's script)
# Single node: vLLM uses multiprocessing internally, no Ray needed
if [ $NUM_NODES -gt 1 ]; then
    echo 'Starting Ray head (NUM_NODES=$NUM_NODES)...'
    ray start --head --node-ip-address='$head_ip' --port=6379 \
        --min-worker-port=20000 --max-worker-port=29999
    sleep 15
    ray status
else
    echo 'Single node mode - skipping Ray (vLLM will use multiprocessing)'
    # Unset RAY_ADDRESS to prevent vLLM from trying to connect to Ray
    unset RAY_ADDRESS
fi

# Check GPU availability
echo '============================================================'
echo 'GPU Check:'
echo '============================================================'
nvidia-smi -L || echo 'WARNING: nvidia-smi failed!'
python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU count: {torch.cuda.device_count()}\")'

# Auto-detect GPU model if not specified
if [ '$GPU_MODEL' == 'unknown' ]; then
    DETECTED_GPU=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
    if echo \"\$DETECTED_GPU\" | grep -qi 'GB200'; then
        GPU_MODEL='GB200'
    elif echo \"\$DETECTED_GPU\" | grep -qi 'H100'; then
        GPU_MODEL='H100'
    elif echo \"\$DETECTED_GPU\" | grep -qi 'H200'; then
        GPU_MODEL='H200'
    elif echo \"\$DETECTED_GPU\" | grep -qi 'A100'; then
        GPU_MODEL='A100'
    else
        GPU_MODEL=\"\$DETECTED_GPU\"
    fi
    echo \"Detected GPU: \$GPU_MODEL\"
fi
echo '============================================================'

# Check package versions (use python3 in vLLM container)
echo 'Package versions:'
echo '============================================================'
python3 -c 'import vllm; print(f\"vLLM: {vllm.__version__}\")'
python3 -c 'import transformers; print(f\"Transformers: {transformers.__version__}\")'
python3 -c 'import huggingface_hub; print(f\"HuggingFace Hub: {huggingface_hub.__version__}\")'
python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'
echo '============================================================'

# Build prompts file argument
PROMPTS_ARG=''
if [ -f '$PROMPTS_FILE' ]; then
    PROMPTS_ARG='--prompts-file $PROMPTS_FILE'
else
    PROMPTS_ARG='--random-input-len $RANDOM_INPUT_LEN'
fi

# Run benchmark (use python3 in vLLM container)
python3 $SCRIPT_DIR/benchmark_vllm_offline.py \
    --model '$MODEL_PATH' \
    --tp $TENSOR_PARALLEL_SIZE \
    --pp $PIPELINE_PARALLEL_SIZE \
    --dp $DATA_PARALLEL_SIZE \
    --ep $EXPERT_PARALLEL_SIZE \
    --num-nodes $NUM_NODES \
    --gpus-per-node $GPUS_PER_NODE \
    --gpu-model \"\$GPU_MODEL\" \
    --num-prompts $NUM_PROMPTS \
    --num-generations $NUM_GENERATIONS \
    --max-model-len $MAX_MODEL_LEN \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    \$PROMPTS_ARG \
    --output-file '$LOGS_DIR/results.json'

# Stop Ray (only if multi-node)
if [ $NUM_NODES -gt 1 ]; then
    ray stop
fi
"

echo "============================================================"
echo "Benchmark complete! Results in: $LOGS_DIR"
echo "============================================================"

