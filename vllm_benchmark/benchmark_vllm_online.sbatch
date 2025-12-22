#!/bin/bash
# Note: -A (account), -p (partition), and -J (job-name) are passed via command line
# from run_vllm_benchmark.sh to support different clusters and descriptive job names
#SBATCH -t 04:00:00
#SBATCH --ntasks-per-node 1
# Note: stdout/stderr are redirected to LOGS_DIR after directory creation

set -ex

# ============================================================
# Paths Configuration (passed from run_vllm_benchmark.sh)
# ============================================================
# SCRIPT_DIR, EXPERIMENT_DIR, MODEL_SHORT_NAME, PARALLELISM_DIR are passed as environment variables
SCRIPT_DIR=${SCRIPT_DIR:-$(dirname "$(readlink -f "$0")")}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-$SCRIPT_DIR/../vllm_standalone_perf_exp}

# Model and parallelism info for directory structure
MODEL_SHORT_NAME=${MODEL_SHORT_NAME:-unknown}
PARALLELISM_DIR=${PARALLELISM_DIR:-1n4t1p}

# Log directory structure: <benchmark_type>/<model>/<parallelism>/<job_id>-logs/
LOGS_DIR=${LOGS_DIR:-$EXPERIMENT_DIR/online/${MODEL_SHORT_NAME}/${PARALLELISM_DIR}/${SLURM_JOB_ID}-logs}

# Create log directory and redirect all output there
mkdir -p "$LOGS_DIR"
exec > >(tee -a "$LOGS_DIR/slurm-${SLURM_JOB_ID}.out") 2>&1

echo "============================================================"
echo "SLURM Job Information"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Log Directory: $LOGS_DIR"
echo "Start Time: $(date)"
echo "============================================================"

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
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-32B-Instruct}

# Parallelism
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
EXPERT_PARALLEL_SIZE=${EXPERT_PARALLEL_SIZE:-1}

# Online serving parameters
MAX_CONCURRENCY=${MAX_CONCURRENCY:-64}
NUM_PROMPTS=${NUM_PROMPTS:-320}  # MAX_CONCURRENCY * 5

# Input/Output sequence lengths to sweep
MAX_ISL=${MAX_ISL:-150}
MAX_OSL=${MAX_OSL:-"1000 2000"}  # Space-separated list

# Generation parameters
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

# Container and paths
# Note: SCRIPT_DIR and EXPERIMENT_DIR are passed from run_vllm_benchmark.sh
BASE_DIR=$(dirname "$SCRIPT_DIR")
CONTAINER_IMAGE=${CONTAINER_IMAGE:-vllm/vllm-openai:nightly-aarch64}
CONTAINER_NAME=vllm-online-bench
# JOB_CACHE_DIR is passed from run_vllm_benchmark.sh (cluster-specific)
# Fallback to a safe default if not set
JOB_CACHE_DIR=${JOB_CACHE_DIR:-/lustre/fsw/coreai_dlalgo_llm/users/sna/job_cache}

# Server startup time (seconds)
SERVER_STARTUP_TIME=${SERVER_STARTUP_TIME:-120}

# ============================================================
# Setup
# ============================================================

mkdir -p "$EXPERIMENT_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "$JOB_CACHE_DIR"

# GPUs per node (can be overridden via env var)
NUM_GPUS_PER_NODE=${GPUS_PER_NODE:-4}
if [[ `hostname` == *"eos"* ]]; then
  NUM_GPUS_PER_NODE=8
fi

# GPU model (will be auto-detected in container if not set)
GPU_MODEL_NAME=${GPU_MODEL:-"unknown"}

NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
NUM_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))

# Calculate Data Parallelism degree
# Note: For online serving, we typically use DP=1 and scale with replicas
GPUS_PER_INSTANCE=$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))
DATA_PARALLEL_SIZE=$((NUM_GPUS / GPUS_PER_INSTANCE))

# Calculate max sequence length from ISL + OSL
LARGEST_ISL=$(echo $MAX_ISL | tr ' ' '\n' | sort -n | tail -n 1)
LARGEST_OSL=$(echo $MAX_OSL | tr ' ' '\n' | sort -n | tail -n 1)
COMPUTED_MAX_LEN=$((LARGEST_ISL + LARGEST_OSL))
if [ $COMPUTED_MAX_LEN -gt $MAX_MODEL_LEN ]; then
    MAX_MODEL_LEN=$COMPUTED_MAX_LEN
fi

echo "============================================================"
echo "vLLM Online Serving Benchmark (Duncan-style)"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Nodes: $NUM_NODES"
echo "GPUs per node: $NUM_GPUS_PER_NODE"
echo "Total GPUs: $NUM_GPUS"
if [ $EXPERT_PARALLEL_SIZE -gt 1 ]; then
    echo "Parallelism: TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE, EP=$EXPERT_PARALLEL_SIZE (MoE)"
else
    echo "Parallelism: TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE"
fi
echo "Max concurrency: $MAX_CONCURRENCY"
echo "Num prompts: $NUM_PROMPTS"
echo "ISL (Input Sequence Length): $MAX_ISL"
echo "OSL (Output Sequence Length): $MAX_OSL"
echo "Max model length: $MAX_MODEL_LEN"
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
# Core vLLM settings
export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# NeMo-RL performance settings (always enabled)
export VLLM_SKIP_P2P_CHECK=1              # Skip P2P check, rely on driver
export VLLM_ENABLE_V1_MULTIPROCESSING=0   # Avoid Ray conflict
export NCCL_CUMEM_ENABLE=1                # Fix PyNCCL P2P init error

# Cross-node model parallelism NCCL settings
if [ $NUM_NODES -gt 1 ]; then
    GPUS_PER_INSTANCE=$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))
    if [ $GPUS_PER_INSTANCE -gt $NUM_GPUS_PER_NODE ]; then
        export NCCL_NVLS_ENABLE=0         # Fix NCCL error for cross-node MP
        echo "Cross-node parallelism detected, NCCL_NVLS_ENABLE=0"
    fi
fi

# MoE-specific environment variables
if [ "$EXPERT_PARALLEL_SIZE" -gt 1 ]; then
    echo "MoE mode enabled (EP=$EXPERT_PARALLEL_SIZE)"
    # Choose MOE backend based on concurrency
    if [ ${MAX_CONCURRENCY:-64} -lt 64 ]; then
        export VLLM_FLASHINFER_MOE_BACKEND=${VLLM_FLASHINFER_MOE_BACKEND:-latency}
    else
        export VLLM_FLASHINFER_MOE_BACKEND=${VLLM_FLASHINFER_MOE_BACKEND:-throughput}
    fi
    echo "VLLM_FLASHINFER_MOE_BACKEND=$VLLM_FLASHINFER_MOE_BACKEND"
fi

# Data Parallelism settings
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))
DATA_PARALLEL_SIZE=$((TOTAL_GPUS / (TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE)))
if [ $DATA_PARALLEL_SIZE -gt 1 ]; then
    export VLLM_DP_MASTER_IP=$head_ip
    export VLLM_DP_SIZE=$DATA_PARALLEL_SIZE
    echo "Data Parallelism enabled: DP=$DATA_PARALLEL_SIZE"
fi

# ============================================================
# Build vllm serve flags
# ============================================================
# Data parallelism: run DP instances across all GPUs
DP_FLAG=""
if [ $DATA_PARALLEL_SIZE -gt 1 ]; then
    DP_FLAG="--data-parallel-size $DATA_PARALLEL_SIZE"
    echo "Data Parallelism: DP=$DATA_PARALLEL_SIZE (Total GPUs = TP*PP*DP = $TENSOR_PARALLEL_SIZE*$PIPELINE_PARALLEL_SIZE*$DATA_PARALLEL_SIZE)"
fi

DISTRIBUTED_FLAG=""
if [ $NUM_NODES -gt 1 ] || [ $DATA_PARALLEL_SIZE -gt 1 ]; then
    DISTRIBUTED_FLAG="--distributed-executor-backend ray"
fi

TP_FLAG=""
if [ $TENSOR_PARALLEL_SIZE -gt 1 ]; then
    TP_FLAG="--tensor-parallel-size $TENSOR_PARALLEL_SIZE"
fi

PP_FLAG=""
if [ $PIPELINE_PARALLEL_SIZE -gt 1 ]; then
    PP_FLAG="--pipeline-parallel-size $PIPELINE_PARALLEL_SIZE"
fi

EP_FLAG=""
if [ $EXPERT_PARALLEL_SIZE -gt 1 ]; then
    EP_FLAG="--enable-expert-parallel"
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

# Wait for head node Ray to start
echo 'Waiting for head node Ray to start...'
sleep 60

# Retry connecting to head node
for i in 1 2 3 4 5; do
    echo \"Attempt \$i: Connecting to Ray head at $head_ip:6379...\"
    ray start --address '$head_ip:6379' && break
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
HF_FIX_DIR=/tmp/hf_fix_\$\$
pip install --quiet --no-cache-dir --target=\$HF_FIX_DIR huggingface-hub==0.34
export PYTHONPATH=\$HF_FIX_DIR:\$PYTHONPATH

# Start Ray head if multi-node OR data parallelism
if [ $NUM_NODES -gt 1 ] || [ $DATA_PARALLEL_SIZE -gt 1 ]; then
    echo 'Starting Ray head (NUM_NODES=$NUM_NODES, DP=$DATA_PARALLEL_SIZE)...'
    ray start --head --node-ip-address='$head_ip' --port=6379
    sleep 15
    ray status
fi

# Check GPU availability
echo '============================================================'
echo 'GPU Check:'
nvidia-smi -L || echo 'WARNING: nvidia-smi failed!'
python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}\")'

# Auto-detect GPU model if not specified
if [ '$GPU_MODEL_NAME' == 'unknown' ]; then
    DETECTED_GPU=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
    if echo \"\$DETECTED_GPU\" | grep -qi 'GB200'; then
        GPU_MODEL_NAME='GB200'
    elif echo \"\$DETECTED_GPU\" | grep -qi 'H100'; then
        GPU_MODEL_NAME='H100'
    elif echo \"\$DETECTED_GPU\" | grep -qi 'H200'; then
        GPU_MODEL_NAME='H200'
    elif echo \"\$DETECTED_GPU\" | grep -qi 'A100'; then
        GPU_MODEL_NAME='A100'
    else
        GPU_MODEL_NAME=\"\$DETECTED_GPU\"
    fi
    echo \"Detected GPU: \$GPU_MODEL_NAME\"
fi
echo '============================================================'

# Check vLLM version
python3 -c 'import vllm; print(f\"vLLM: {vllm.__version__}\")'

# ============================================================
# Start vLLM server in background
# ============================================================
echo ''
echo '============================================================'
echo 'Starting vLLM server...'
echo '============================================================'

vllm serve '$MODEL_PATH' \
    $DISTRIBUTED_FLAG \
    $TP_FLAG \
    $PP_FLAG \
    $DP_FLAG \
    $EP_FLAG \
    --trust-remote-code \
    --disable-log-requests \
    --max-num-seqs $MAX_CONCURRENCY \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --no-enable-prefix-caching \
    &

SERVER_PID=\$!

# Monitor GPU in background
nvidia-smi -l 120 > '$LOGS_DIR/nvidia-smi.log' 2>&1 &
NVIDIA_PID=\$!

# Wait for server to start
echo 'Waiting for server to start (${SERVER_STARTUP_TIME}s)...'
sleep $SERVER_STARTUP_TIME

# Check if server is running
if ! ps -p \$SERVER_PID > /dev/null; then
    echo 'ERROR: vLLM server died!'
    exit 1
fi

echo 'Server is ready!'

# ============================================================
# Run benchmarks with different ISL/OSL combinations
# ============================================================
echo ''
echo '============================================================'
echo 'Running benchmarks...'
echo '============================================================'

RESULTS_SUMMARY='$LOGS_DIR/results_summary.txt'
echo 'GPU,Nodes,GPUsPerNode,TotalGPUs,TP,PP,ISL,OSL,Concurrency,NumPrompts,RequestThroughput,TokenThroughput,MeanTTFT,MeanITL' > \$RESULTS_SUMMARY

for ISL in $MAX_ISL; do
for OSL in $MAX_OSL; do
    echo ''
    echo \">>> Benchmarking ISL=\$ISL, OSL=\$OSL, Concurrency=$MAX_CONCURRENCY\"
    
    RESULT_FILE='$LOGS_DIR/result_ISL\${ISL}_OSL\${OSL}.txt'
    
    # Retry loop
    for attempt in 1 2 3; do
        if vllm bench serve \
            --backend vllm \
            --model '$MODEL_PATH' \
            --dataset-name random \
            --max-concurrency $MAX_CONCURRENCY \
            --num-prompts $NUM_PROMPTS \
            --ignore-eos \
            --random-input-len \$ISL \
            --random-output-len \$OSL > \$RESULT_FILE 2>&1; then
            echo \"Benchmark completed for ISL=\$ISL, OSL=\$OSL\"
            cat \$RESULT_FILE
            break
        else
            # Check if server died
            if ! ps -p \$SERVER_PID > /dev/null; then
                echo 'ERROR: vLLM server died during benchmark!'
                cat \$RESULT_FILE
                exit 1
            fi
            echo \"Attempt \$attempt failed, retrying...\"
            cat \$RESULT_FILE
            sleep 10
        fi
    done
    
    # Extract metrics and append to summary using Python script
    if [ -f \$RESULT_FILE ]; then
        python3 $SCRIPT_DIR/parse_online_result.py \$RESULT_FILE \$ISL \$OSL $MAX_CONCURRENCY $NUM_PROMPTS \"\$GPU_MODEL_NAME\" $NUM_NODES $NUM_GPUS_PER_NODE $NUM_GPUS $TENSOR_PARALLEL_SIZE $PIPELINE_PARALLEL_SIZE >> \$RESULTS_SUMMARY
    fi
done
done

# ============================================================
# Cleanup
# ============================================================
echo ''
echo '============================================================'
echo 'Benchmark complete!'
echo '============================================================'

# Print summary
echo ''
echo 'Results Summary:'
cat \$RESULTS_SUMMARY

# Stop server
kill \$SERVER_PID 2>/dev/null || true
kill \$NVIDIA_PID 2>/dev/null || true

# Stop Ray
if [ $NUM_NODES -gt 1 ]; then
    ray stop
fi
"

echo "============================================================"
echo "Benchmark complete! Results in: $LOGS_DIR"
echo "============================================================"

# ============================================================
# Upload to wandb (if enabled)
# ============================================================
WANDB_UPLOAD=${WANDB_UPLOAD:-1}
WANDB_PROJECT=${WANDB_PROJECT:-vllm-benchmark}

if [ "$WANDB_UPLOAD" == "1" ] && [ -f "$LOGS_DIR/results.json" ]; then
    echo ""
    echo "Uploading results to wandb (project: $WANDB_PROJECT)..."
    
    python3 $SCRIPT_DIR/upload_to_wandb.py \
        "$LOGS_DIR/results.json" \
        --project "$WANDB_PROJECT" \
        --benchmark-type online \
        --tags "online" "$GPU_MODEL" || echo "Warning: wandb upload failed (wandb may not be installed)"
fi

echo ""
echo "End time: $(date)"
echo "============================================================"

