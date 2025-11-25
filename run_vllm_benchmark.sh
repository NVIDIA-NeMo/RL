#!/bin/bash
# Run vLLM Offline Benchmark
#
# Usage:
#   # Step 1: Prepare prompts (submits a SLURM job)
#   ./run_vllm_benchmark.sh prepare
#
#   # Step 2: Run benchmark (after prepare job completes)
#   ./run_vllm_benchmark.sh run
#
#   # Or run with random prompts (no preparation needed)
#   ./run_vllm_benchmark.sh run-random

set -e

# ============================================================
# Configuration
# ============================================================
NUM_NODES=${NUM_NODES:-4}
GPUS_PER_NODE=4

# Model (use local path for faster loading)
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-32B-Instruct}

# Parallelism
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}

# DP (Data Parallelism) is automatically calculated:
# DP = (NUM_NODES × GPUS_PER_NODE) / (TP × PP)
# For example: 4 nodes × 4 GPUs = 16 GPUs, TP=4, PP=1 → DP=4

# GRPO-style batch
NUM_PROMPTS=${NUM_PROMPTS:-64}
NUM_GENERATIONS=${NUM_GENERATIONS:-32}

# Seed for dataset (must match grpo.seed in your GRPO config for fair comparison)
# Default 42 matches grpo_math_1B.yaml (and inherited configs like grpo_math_qwen30ba3b_megatron.yaml)
SEED=${SEED:-42}

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPTS_FILE="$SCRIPT_DIR/prompts.json"

# Container configuration:
# - prepare: uses NeMo-RL container (needs NeMo-RL dataset dependencies)
# - run/build: uses vLLM container (set in benchmark_vllm_offline.sbatch)
NEMO_RL_CONTAINER=${NEMO_RL_CONTAINER:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/nemo_rl.sqsh}
# vLLM container is configured in benchmark_vllm_offline.sbatch:
#   CONTAINER_IMAGE=${CONTAINER_IMAGE:-vllm/vllm-openai:nightly}
HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}

# Generation parameters
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_TOKENS=${MAX_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-1.0}

# ============================================================
# Commands
# ============================================================

case "${1:-help}" in
    prepare)
        echo "============================================================"
        echo "Submitting prompt preparation job (NeMo-RL container)"
        echo "============================================================"
        echo "Model: $MODEL_PATH"
        echo "Num prompts: $NUM_PROMPTS"
        echo "SCRIPT_DIR: $SCRIPT_DIR"
        echo "Output: $PROMPTS_FILE"
        echo "============================================================"
        
        cd "$SCRIPT_DIR"
        
        # Submit a short job to prepare prompts using NeMo-RL container
        # Use ray.sub style: pass env vars directly, not via export
        CONTAINER="$NEMO_RL_CONTAINER" \
        HF_HOME="$HF_HOME" \
        HF_DATASETS_CACHE="$HF_HOME/cache" \
        MOUNTS="/lustre:/lustre" \
        COMMAND="/opt/nemo_rl_venv/bin/python $SCRIPT_DIR/prepare_prompts.py \
            --model $MODEL_PATH \
            --output $PROMPTS_FILE \
            --num-prompts $NUM_PROMPTS \
            --seed $SEED" \
        sbatch --nodes=1 \
            --gres=gpu:4 \
            --time=00:30:00 \
            --account=coreai_dlalgo_nemorl \
            --partition=batch \
            --job-name=prepare-prompts \
            --output="$SCRIPT_DIR/prepare_prompts_%j.log" \
            ray.sub
        
        echo ""
        echo "Job submitted! Check status with: squeue -u sna"
        echo "After job completes, run: $0 run"
        ;;
    
    run)
        # Calculate DP for display
        TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
        GPUS_PER_INSTANCE=$((TP_SIZE * PP_SIZE))
        DP_SIZE=$((TOTAL_GPUS / GPUS_PER_INSTANCE))
        
        echo "============================================================"
        echo "Submitting vLLM Offline Benchmark"
        echo "============================================================"
        echo "Nodes: $NUM_NODES (${GPUS_PER_NODE} GPUs each)"
        echo "Total GPUs: $TOTAL_GPUS"
        echo "Model: $MODEL_PATH"
        echo "Parallelism: TP=$TP_SIZE, PP=$PP_SIZE, DP=$DP_SIZE (auto-calculated)"
        echo "Prompts: $NUM_PROMPTS × $NUM_GENERATIONS"
        echo "Prompts file: $PROMPTS_FILE"
        echo "============================================================"
        
        if [ ! -f "$PROMPTS_FILE" ]; then
            echo "Warning: Prompts file not found. Run '$0 prepare' first or use 'run-random'."
            echo "Proceeding with random prompts..."
        fi
        
        cd "$SCRIPT_DIR"
        
        MODEL_PATH="$MODEL_PATH" \
        TENSOR_PARALLEL_SIZE=$TP_SIZE \
        PIPELINE_PARALLEL_SIZE=$PP_SIZE \
        NUM_PROMPTS=$NUM_PROMPTS \
        NUM_GENERATIONS=$NUM_GENERATIONS \
        MAX_MODEL_LEN=$MAX_MODEL_LEN \
        MAX_TOKENS=$MAX_TOKENS \
        TEMPERATURE=$TEMPERATURE \
        PROMPTS_FILE="$PROMPTS_FILE" \
        sbatch \
            --nodes=$NUM_NODES \
            --gres=gpu:$GPUS_PER_NODE \
            benchmark_vllm_offline.sbatch
        
        echo ""
        echo "Logs will be in: vllm_standalone_perf_exp/\$SLURM_JOB_ID-logs/"
        echo "  - slurm-*.out    (main output)"
        echo "  - slurm-*.err    (errors)"
        echo "  - results.json   (benchmark results)"
        ;;
    
    run-random)
        # Calculate DP for display
        TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
        GPUS_PER_INSTANCE=$((TP_SIZE * PP_SIZE))
        DP_SIZE=$((TOTAL_GPUS / GPUS_PER_INSTANCE))
        
        echo "============================================================"
        echo "Submitting vLLM Offline Benchmark (Random Prompts)"
        echo "============================================================"
        echo "Nodes: $NUM_NODES (${GPUS_PER_NODE} GPUs each)"
        echo "Total GPUs: $TOTAL_GPUS"
        echo "Model: $MODEL_PATH"
        echo "Parallelism: TP=$TP_SIZE, PP=$PP_SIZE, DP=$DP_SIZE (auto-calculated)"
        echo "Prompts: $NUM_PROMPTS × $NUM_GENERATIONS (random)"
        echo "============================================================"
        
        cd "$SCRIPT_DIR"
        
        MODEL_PATH="$MODEL_PATH" \
        TENSOR_PARALLEL_SIZE=$TP_SIZE \
        PIPELINE_PARALLEL_SIZE=$PP_SIZE \
        NUM_PROMPTS=$NUM_PROMPTS \
        NUM_GENERATIONS=$NUM_GENERATIONS \
        MAX_MODEL_LEN=$MAX_MODEL_LEN \
        MAX_TOKENS=$MAX_TOKENS \
        TEMPERATURE=$TEMPERATURE \
        PROMPTS_FILE="" \
        RANDOM_INPUT_LEN=150 \
        sbatch \
            --nodes=$NUM_NODES \
            --gres=gpu:$GPUS_PER_NODE \
            benchmark_vllm_offline.sbatch
        
        echo ""
        echo "Logs will be in: vllm_standalone_perf_exp/\$SLURM_JOB_ID-logs/"
        ;;
    
    build)
        echo "============================================================"
        echo "Building venv for vLLM benchmark (one-time setup)"
        echo "============================================================"
        echo "This creates a persistent venv that will be reused."
        echo "============================================================"
        
        cd "$SCRIPT_DIR"
        
        DO_BUILD=1 \
        VLLM_INSTALL_FROM_SOURCE="${VLLM_INSTALL_FROM_SOURCE:-0}" \
        VLLM_GIT_REPO="${VLLM_GIT_REPO:-https://github.com/vllm-project/vllm.git}" \
        VLLM_GIT_BRANCH="${VLLM_GIT_BRANCH:-main}" \
        sbatch \
            --nodes=1 \
            --gres=gpu:$GPUS_PER_NODE \
            --time=01:00:00 \
            --job-name=vllm-build \
            benchmark_vllm_offline.sbatch
        
        echo ""
        echo "Build job submitted! After it completes, run: $0 run"
        ;;
    
    build-custom)
        echo "============================================================"
        echo "Building venv with Custom vLLM from source"
        echo "============================================================"
        echo "vLLM repo: ${VLLM_GIT_REPO:-https://github.com/vllm-project/vllm.git}"
        echo "vLLM branch: ${VLLM_GIT_BRANCH:-main}"
        echo "============================================================"
        
        cd "$SCRIPT_DIR"
        
        DO_BUILD=1 \
        VLLM_INSTALL_FROM_SOURCE=1 \
        VLLM_GIT_REPO="${VLLM_GIT_REPO:-https://github.com/vllm-project/vllm.git}" \
        VLLM_GIT_BRANCH="${VLLM_GIT_BRANCH:-main}" \
        sbatch \
            --nodes=1 \
            --gres=gpu:$GPUS_PER_NODE \
            --time=01:00:00 \
            --job-name=vllm-build-custom \
            benchmark_vllm_offline.sbatch
        
        echo ""
        echo "Build job submitted! After it completes, run: $0 run"
        ;;
    
    build-local)
        if [ -z "$VLLM_LOCAL_PATH" ]; then
            echo "ERROR: VLLM_LOCAL_PATH is required"
            echo ""
            echo "Usage:"
            echo "  VLLM_LOCAL_PATH=/path/to/your/vllm $0 build-local"
            echo ""
            echo "Example:"
            echo "  # Clone and modify vLLM"
            echo "  git clone https://github.com/vllm-project/vllm.git ~/vllm"
            echo "  cd ~/vllm && vim vllm/engine/llm_engine.py  # make changes"
            echo ""
            echo "  # Build with local vLLM"
            echo "  VLLM_LOCAL_PATH=~/vllm $0 build-local"
            exit 1
        fi
        
        echo "============================================================"
        echo "Building venv with Local vLLM (editable install)"
        echo "============================================================"
        echo "vLLM local path: $VLLM_LOCAL_PATH"
        echo "============================================================"
        
        if [ ! -d "$VLLM_LOCAL_PATH" ]; then
            echo "ERROR: Directory not found: $VLLM_LOCAL_PATH"
            exit 1
        fi
        
        cd "$SCRIPT_DIR"
        
        DO_BUILD=1 \
        VLLM_LOCAL_PATH="$VLLM_LOCAL_PATH" \
        sbatch \
            --nodes=1 \
            --gres=gpu:$GPUS_PER_NODE \
            --time=01:00:00 \
            --job-name=vllm-build-local \
            benchmark_vllm_offline.sbatch
        
        echo ""
        echo "Build job submitted! After it completes, run: $0 run"
        echo ""
        echo "TIP: After modifying vLLM code, re-run this command to rebuild."
        ;;
    
    run-custom-vllm)
        echo "============================================================"
        echo "Submitting vLLM Benchmark (uses pre-built venv)"
        echo "============================================================"
        echo "Make sure you ran '$0 build-custom' first!"
        echo "============================================================"
        
        cd "$SCRIPT_DIR"
        
        MODEL_PATH="$MODEL_PATH" \
        TENSOR_PARALLEL_SIZE=$TP_SIZE \
        PIPELINE_PARALLEL_SIZE=$PP_SIZE \
        NUM_PROMPTS=$NUM_PROMPTS \
        NUM_GENERATIONS=$NUM_GENERATIONS \
        PROMPTS_FILE="$PROMPTS_FILE" \
        sbatch \
            --nodes=$NUM_NODES \
            --gres=gpu:$GPUS_PER_NODE \
            benchmark_vllm_offline.sbatch
        ;;
    
    help|*)
        echo "vLLM Offline Benchmark Runner"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  prepare         Prepare prompts from NeMo-RL dataset (submits SLURM job)"
        echo "  build           Build venv for benchmark (one-time, uses container vLLM)"
        echo "  build-custom    Build venv with custom vLLM from GitHub"
        echo "  build-local     Build venv with local vLLM source (for development)"
        echo "  run             Run benchmark with prepared prompts"
        echo "  run-random      Run benchmark with random prompts (no prepare needed)"
        echo "  run-custom-vllm Run benchmark (after build-custom or build-local)"
        echo ""
        echo "Environment variables:"
        echo "  NUM_NODES       Number of nodes (default: 4)"
        echo "  MODEL_PATH      Model path or HuggingFace name"
        echo "  TP_SIZE         Tensor parallel size (default: 4)"
        echo "  PP_SIZE         Pipeline parallel size (default: 1)"
        echo "  # DP is auto-calculated: DP = (NUM_NODES × GPUS_PER_NODE) / (TP × PP)"
        echo "  NUM_PROMPTS     Number of unique prompts (default: 64)"
        echo "  NUM_GENERATIONS Generations per prompt (default: 32)"
        echo "  SEED            Random seed for dataset (default: 42, must match grpo.seed)"
        echo "  VLLM_GIT_REPO   Custom vLLM git repo (for build-custom)"
        echo "  VLLM_GIT_BRANCH Custom vLLM git branch (for build-custom)"
        echo "  VLLM_LOCAL_PATH Local vLLM source directory (for build-local)"
        echo ""
        echo "Parallelism example (4 nodes × 4 GPUs = 16 GPUs):"
        echo "  TP=4, PP=1 → DP=4 (4 vLLM instances, each using 4 GPUs)"
        echo "  TP=8, PP=1 → DP=2 (2 vLLM instances, each using 8 GPUs)"
        echo "  TP=4, PP=2 → DP=2 (2 vLLM instances, each using 8 GPUs)"
        echo ""
        echo "Examples:"
        echo "  # Quick start (random prompts)"
        echo "  $0 run-random"
        echo ""
        echo "  # With NeMo-RL dataset prompts"
        echo "  $0 prepare           # Submit job to prepare prompts"
        echo "  squeue -u \$USER      # Wait for job to complete"
        echo "  $0 run               # Run benchmark"
        echo ""
        echo "  # Test custom vLLM branch (from GitHub)"
        echo "  VLLM_GIT_REPO=https://github.com/myuser/vllm.git \\"
        echo "  VLLM_GIT_BRANCH=my-feature \\"
        echo "  $0 build-custom      # Build venv with custom vLLM (one-time)"
        echo "  $0 run-custom-vllm   # Run benchmark"
        echo ""
        echo "  # Test local vLLM modifications (RECOMMENDED for development)"
        echo "  git clone https://github.com/vllm-project/vllm.git ~/vllm"
        echo "  # Edit ~/vllm/vllm/... as needed"
        echo "  VLLM_LOCAL_PATH=~/vllm $0 build-local"
        echo "  $0 run-custom-vllm   # Run benchmark"
        echo "  # After more edits, rebuild:"
        echo "  VLLM_LOCAL_PATH=~/vllm $0 build-local"
        ;;
esac

