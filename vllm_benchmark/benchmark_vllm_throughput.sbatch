#!/bin/bash
#SBATCH -A coreai_dlalgo_nemorl
#SBATCH -J vllm-throughput-bench
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node 1
#SBATCH -o vllm_standalone_perf_exp/throughput/%j-logs/slurm-%j.out
#SBATCH -e vllm_standalone_perf_exp/throughput/%j-logs/slurm-%j.err

set -ex

# ============================================================
# vLLM Official Throughput Benchmark (vllm bench throughput)
# ============================================================
# This uses vLLM's built-in benchmark tool for measuring
# maximum throughput with fixed input/output lengths.
#
# Usage:
#   ./run_vllm_benchmark.sh run-throughput
#
# The benchmark sweeps through multiple ISL/OSL combinations
# and reports tokens/sec for each configuration.
# ============================================================

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
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-32B-Instruct}

# Cluster configuration
# GB200: 4 GPUs/node, H100: 8 GPUs/node
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
GPU_MODEL=${GPU_MODEL:-"unknown"}

# Parallelism
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
EXPERT_PARALLEL_SIZE=${EXPERT_PARALLEL_SIZE:-1}

# MoE settings
ENABLE_EXPERT_PARALLEL=${ENABLE_EXPERT_PARALLEL:-0}

# ============================================================
# Throughput benchmark parameters
# ============================================================
# Reference: https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/throughput.py
#
# Without --dataset, vllm bench throughput uses RandomDataset:
#   - Input: EXACTLY input-len tokens (synthetic random tokens)
#   - Output: EXACTLY output-len tokens (ignore_eos=True is HARDCODED in vLLM!)
#
# Key vLLM source code facts:
#   1. ignore_eos=True is ALREADY HARDCODED in SamplingParams
#   2. --random-range-ratio controls input length variance (default: 0.0 = exact)
# ============================================================

# ISL (Input Sequence Length) - space-separated list
# Each prompt will have EXACTLY this many input tokens (RandomDataset)
INPUT_LENS=${INPUT_LENS:-"128 256 512 1024"}

# OSL (Output Sequence Length) - space-separated list  
OUTPUT_LENS=${OUTPUT_LENS:-"128 256 512 1024 2048"}

# Number of prompts to generate per ISL/OSL combination
NUM_PROMPTS=${NUM_PROMPTS:-1000}

# Input length variance ratio (default: 0.0 = exact length)
# 0.0 = EXACT input-len tokens
# 0.2 = input-len * [0.8, 1.2] range
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.0}

# Model configuration
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

# Container and paths
# Script directory and parent directory
SCRIPT_DIR=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/vllm_benchmark
BASE_DIR=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl
CONTAINER_IMAGE=${CONTAINER_IMAGE:-vllm/vllm-openai:nightly-aarch64}
CONTAINER_NAME=vllm-throughput-bench
JOB_CACHE_DIR=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/job_cache

# Log directory
EXPERIMENT_DIR=${EXPERIMENT_DIR:-$BASE_DIR/vllm_standalone_perf_exp}
LOGS_DIR=${LOGS_DIR:-$EXPERIMENT_DIR/throughput/${SLURM_JOB_ID}-logs}

# venv for custom installations
VLLM_VENV_DIR=${VLLM_VENV_DIR:-$BASE_DIR/vllm-benchmark-venv}

# ============================================================
# Setup
# ============================================================

mkdir -p $EXPERIMENT_DIR
mkdir -p $LOGS_DIR
mkdir -p $JOB_CACHE_DIR

NUM_GPUS_PER_NODE=4
if [[ `hostname` == *"eos"* ]]; then
  NUM_GPUS_PER_NODE=8
fi

NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
NUM_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))

echo "============================================================"
echo "vLLM Official Throughput Benchmark"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Nodes: $NUM_NODES"
echo "GPUs per node: $NUM_GPUS_PER_NODE"
echo "Total GPUs: $NUM_GPUS"
echo "Parallelism: TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE"
if [ $EXPERT_PARALLEL_SIZE -gt 1 ]; then
    echo "Expert Parallelism: EP=$EXPERT_PARALLEL_SIZE (MoE)"
fi
echo "Input lengths: $INPUT_LENS"
echo "Output lengths: $OUTPUT_LENS"
echo "Num prompts per config: $NUM_PROMPTS"
echo "Random range ratio: $RANDOM_RANGE_RATIO"
if [ "$RANDOM_RANGE_RATIO" == "0.0" ] || [ "$RANDOM_RANGE_RATIO" == "0" ]; then
    echo "Length mode: EXACT (input=exact, output=exact via hardcoded ignore_eos)"
else
    echo "Length mode: VARIABLE input (±${RANDOM_RANGE_RATIO}), EXACT output"
fi
echo "Note: vLLM hardcodes ignore_eos=True, so output is always exact"
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

# MoE-specific environment variables
if [ "$ENABLE_EXPERT_PARALLEL" == "1" ] || [ $EXPERT_PARALLEL_SIZE -gt 1 ]; then
    echo "MoE mode enabled (EP=$EXPERT_PARALLEL_SIZE)"
    export VLLM_FLASHINFER_MOE_BACKEND=${VLLM_FLASHINFER_MOE_BACKEND:-throughput}
fi

# ============================================================
# Start Ray workers on non-head nodes
# ============================================================
worker_nodes=("${nodes_array[@]:1}")

if [ ${#worker_nodes[@]} -gt 0 ]; then
    echo "Starting Ray workers on: ${worker_nodes[*]}"
    for node in "${worker_nodes[@]}"; do
        srun --nodes=1 --ntasks=1 -w "$node" --mpi=pmix \
            --container-image="$CONTAINER_IMAGE" \
            --container-mounts="/lustre:/lustre,$JOB_CACHE_DIR:/root/.cache" \
            --container-name=$CONTAINER_NAME-worker-$node \
            bash -c "
set -ex
nvidia-smi -L
if [ -d '$VLLM_VENV_DIR' ]; then
    source $VLLM_VENV_DIR/bin/activate
fi

# Install huggingface-hub fix
HF_FIX_DIR=/tmp/hf_fix_\$\$
mkdir -p \$HF_FIX_DIR
pip install --quiet --no-cache-dir --target=\$HF_FIX_DIR huggingface-hub==0.34
export PYTHONPATH=\$HF_FIX_DIR:\$PYTHONPATH

export RAY_ADDRESS=$head_ip:6379
ray start --address='$head_ip:6379'
sleep infinity
" &
    done
    
    sleep 10
fi

# ============================================================
# Run Throughput Benchmark on head node
# ============================================================
echo "Running throughput benchmark on head node..."

srun --nodes=1 --ntasks=1 -w "$head_node" --mpi=pmix \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="/lustre:/lustre,$JOB_CACHE_DIR:/root/.cache" \
    --container-name=$CONTAINER_NAME-head \
    bash -c "
set -ex

# GPU check
nvidia-smi -L
echo 'Checking PyTorch CUDA access...'
python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}\")'

# Activate venv if exists
if [ -d '$VLLM_VENV_DIR' ]; then
    echo 'Activating venv: $VLLM_VENV_DIR'
    source $VLLM_VENV_DIR/bin/activate
fi

# Install huggingface-hub fix
HF_FIX_DIR=/tmp/hf_fix_\$\$
mkdir -p \$HF_FIX_DIR
pip install --quiet --no-cache-dir --target=\$HF_FIX_DIR huggingface-hub==0.34
export PYTHONPATH=\$HF_FIX_DIR:\$PYTHONPATH

# Set vLLM environment
export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Ray setup
if [ $NUM_NODES -gt 1 ]; then
    echo 'Starting Ray head (NUM_NODES=$NUM_NODES)...'
    ray start --head --node-ip-address='$head_ip' --port=6379 \
        --min-worker-port=20000 --max-worker-port=29999
    sleep 15
    ray status
else
    echo 'Single node mode - skipping Ray'
    unset RAY_ADDRESS
fi

# Create results file
RESULTS_FILE=$LOGS_DIR/results.json
RESULTS_SUMMARY=$LOGS_DIR/results_summary.txt

# Write header
echo 'Model,GPU,Nodes,GPUs/Node,TP,PP,EP,InputLen,OutputLen,NumPrompts,Throughput(req/s),Throughput(tok/s),TotalTime(s)' > \$RESULTS_SUMMARY

# Build common vLLM args
VLLM_ARGS=\"--model $MODEL_PATH\"
VLLM_ARGS=\"\$VLLM_ARGS --tensor-parallel-size $TENSOR_PARALLEL_SIZE\"
VLLM_ARGS=\"\$VLLM_ARGS --pipeline-parallel-size $PIPELINE_PARALLEL_SIZE\"
VLLM_ARGS=\"\$VLLM_ARGS --max-model-len $MAX_MODEL_LEN\"
VLLM_ARGS=\"\$VLLM_ARGS --num-prompts $NUM_PROMPTS\"
VLLM_ARGS=\"\$VLLM_ARGS --trust-remote-code\"

# Input length variance (0.0 = exact length)
# Note: ignore_eos=True is HARDCODED in vLLM's throughput.py, no flag needed
VLLM_ARGS=\"\$VLLM_ARGS --random-range-ratio $RANDOM_RANGE_RATIO\"

# Add expert parallelism for MoE models
if [ $EXPERT_PARALLEL_SIZE -gt 1 ]; then
    VLLM_ARGS=\"\$VLLM_ARGS --enable-expert-parallel\"
fi

# Add Ray backend for multi-GPU
if [ $NUM_NODES -gt 1 ]; then
    VLLM_ARGS=\"\$VLLM_ARGS --distributed-executor-backend ray\"
fi

# Get GPU model name (extract short name like GB200, H100, A100, etc.)
GPU_MODEL_RAW=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
# Extract GPU model: "NVIDIA GB200" -> "GB200", "NVIDIA H100" -> "H100"
# Note: Use grep -E instead of grep -oP for better compatibility (aarch64)
GPU_MODEL=\$(echo \"\$GPU_MODEL_RAW\" | grep -oE 'GB200|H100|H200|A100|A10|L40|RTX[0-9]+' | head -1 || true)
if [ -z \"\$GPU_MODEL\" ]; then
    # Fallback: remove "NVIDIA " prefix and take first word
    GPU_MODEL=\$(echo \"\$GPU_MODEL_RAW\" | sed 's/NVIDIA //g' | awk '{print \$1}' | cut -c1-10)
fi
echo \"Detected GPU: \$GPU_MODEL_RAW -> \$GPU_MODEL\"

# Extract model short name
MODEL_SHORT=\$(basename $MODEL_PATH)

echo ''
echo '============================================================'
echo 'Starting throughput sweep...'
echo '============================================================'

# Initialize JSON results
echo '[' > \$RESULTS_FILE

FIRST_RESULT=1

# Sweep through ISL/OSL combinations
for ISL in $INPUT_LENS; do
    for OSL in $OUTPUT_LENS; do
        echo ''
        echo '============================================================'
        echo \"Running: ISL=\$ISL, OSL=\$OSL\"
        echo '============================================================'
        
        OUTPUT_FILE=$LOGS_DIR/result_ISL\${ISL}_OSL\${OSL}.txt
        
        # Run vllm bench throughput
        vllm bench throughput \
            \$VLLM_ARGS \
            --input-len \$ISL \
            --output-len \$OSL \
            2>&1 | tee \$OUTPUT_FILE
        
        # Parse results from output
        # Look for lines like:
        # Throughput: 123.45 requests/s, 12345.67 total tokens/s, 1234.56 output tokens/s
        # Note: Use sed instead of grep -oP for better compatibility (aarch64)
        
        THROUGHPUT_REQ=\$(grep 'Throughput:' \$OUTPUT_FILE | sed 's/.*Throughput: *\([0-9.]*\) *requests.*/\1/' | head -1 || echo '0')
        THROUGHPUT_TOK=\$(grep 'total tokens/s' \$OUTPUT_FILE | sed 's/.*[, ] *\([0-9.]*\) *total tokens.*/\1/' | head -1 || echo '0')
        TOTAL_TIME=\$(grep 'Total time:' \$OUTPUT_FILE | sed 's/.*Total time: *\([0-9.]*\).*/\1/' | head -1 || echo '0')
        
        # If parsing failed, try alternate format
        if [ -z \"\$THROUGHPUT_REQ\" ] || [ \"\$THROUGHPUT_REQ\" == '0' ]; then
            THROUGHPUT_REQ=\$(grep -i 'throughput' \$OUTPUT_FILE | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo '0')
        fi
        if [ -z \"\$THROUGHPUT_TOK\" ] || [ \"\$THROUGHPUT_TOK\" == '0' ]; then
            THROUGHPUT_TOK=\$(grep -i 'tokens' \$OUTPUT_FILE | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo '0')
        fi
        
        # Ensure values are valid numbers (default to 0 if empty)
        THROUGHPUT_REQ=\${THROUGHPUT_REQ:-0}
        THROUGHPUT_TOK=\${THROUGHPUT_TOK:-0}
        TOTAL_TIME=\${TOTAL_TIME:-0}
        
        echo \"Parsed: req/s=\$THROUGHPUT_REQ, tok/s=\$THROUGHPUT_TOK, time=\$TOTAL_TIME\"
        
        # Append to CSV
        echo \"\$MODEL_SHORT,\$GPU_MODEL,$NUM_NODES,$NUM_GPUS_PER_NODE,$TENSOR_PARALLEL_SIZE,$PIPELINE_PARALLEL_SIZE,$EXPERT_PARALLEL_SIZE,\$ISL,\$OSL,$NUM_PROMPTS,\$THROUGHPUT_REQ,\$THROUGHPUT_TOK,\$TOTAL_TIME\" >> \$RESULTS_SUMMARY
        
        # Append to JSON
        if [ \$FIRST_RESULT -eq 0 ]; then
            echo ',' >> \$RESULTS_FILE
        fi
        FIRST_RESULT=0
        
        cat >> \$RESULTS_FILE << JSONEOF
{
  \"model\": \"$MODEL_PATH\",
  \"gpu_model\": \"\$GPU_MODEL\",
  \"num_nodes\": $NUM_NODES,
  \"gpus_per_node\": $NUM_GPUS_PER_NODE,
  \"total_gpus\": $NUM_GPUS,
  \"tp_size\": $TENSOR_PARALLEL_SIZE,
  \"pp_size\": $PIPELINE_PARALLEL_SIZE,
  \"ep_size\": $EXPERT_PARALLEL_SIZE,
  \"input_len\": \$ISL,
  \"output_len\": \$OSL,
  \"num_prompts\": $NUM_PROMPTS,
  \"throughput_requests_per_sec\": \$THROUGHPUT_REQ,
  \"throughput_tokens_per_sec\": \$THROUGHPUT_TOK,
  \"total_time_sec\": \$TOTAL_TIME,
  \"benchmark_type\": \"throughput\"
}
JSONEOF
        
    done
done

echo ']' >> \$RESULTS_FILE

echo ''
echo '============================================================'
echo 'Benchmark complete!'
echo '============================================================'
echo 'Results summary:'
cat \$RESULTS_SUMMARY
echo ''
echo 'JSON results: \$RESULTS_FILE'
echo 'CSV summary: \$RESULTS_SUMMARY'
"

echo ""
echo "============================================================"
echo "Throughput benchmark complete!"
echo "============================================================"
echo "Results:"
echo "  - Summary: $LOGS_DIR/results_summary.txt"
echo "  - JSON: $LOGS_DIR/results.json"
echo "  - Individual results: $LOGS_DIR/result_ISL*_OSL*.txt"
echo "============================================================"

