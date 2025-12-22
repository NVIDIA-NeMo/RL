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
# Model Presets (use MODEL=<name> for easy configuration)
# ============================================================
# Usage: MODEL=qwen32b ./run_vllm_benchmark.sh run-random
#
# Available presets:
#   qwen32b      - Qwen2.5-32B-Instruct (Dense, 32B)
#   qwen3-30b    - Qwen3-30B-A3B (MoE, 30B total, 3B active)
#   qwen3-235b   - Qwen3-235B-A22B (MoE, 235B total, 22B active)
#   llama70b     - Llama-3.1-70B-Instruct (Dense, 70B)
#   llama405b    - Llama-3.1-405B-Instruct (Dense, 405B)
#   deepseek-v3  - DeepSeek-V3 (MoE, 671B total, 37B active)
#
# Or use MODEL_PATH directly for custom models

apply_model_preset() {
    case "${MODEL:-}" in
        # Qwen Models - Dense
        qwen32b|qwen2.5-32b)
            MODEL_PATH="Qwen/Qwen2.5-32B-Instruct"
            : ${TP_SIZE:=4}
            : ${NUM_NODES:=1}
            : ${MAX_MODEL_LEN:=32768}  # Max context: 32K
            ;;
        qwen3-32b)
            MODEL_PATH="Qwen/Qwen3-32B"
            : ${TP_SIZE:=4}
            : ${NUM_NODES:=1}
            : ${MAX_MODEL_LEN:=32768}  # Max context: 32K
            ;;
        
        # Qwen Models - MoE
        qwen3-30b|qwen30b)
            MODEL_PATH="Qwen/Qwen3-30B-A3B"
            : ${TP_SIZE:=4}
            : ${EP_SIZE:=4}
            : ${NUM_NODES:=1}
            : ${MAX_MODEL_LEN:=32768}  # Max context: 32K
            IS_MOE=1
            ;;
        qwen3-235b|qwen235b)
            MODEL_PATH="Qwen/Qwen3-235B-A22B"
            : ${TP_SIZE:=8}
            : ${EP_SIZE:=8}
            : ${NUM_NODES:=2}
            : ${MAX_MODEL_LEN:=32768}  # Max context: 32K
            IS_MOE=1
            ;;
        
        # LLaMA Models
        llama8b|llama3-8b)
            MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
            : ${TP_SIZE:=1}
            : ${NUM_NODES:=1}
            : ${MAX_MODEL_LEN:=131072}  # Max context: 128K
            ;;
        llama70b|llama3-70b)
            MODEL_PATH="meta-llama/Llama-3.1-70B-Instruct"
            : ${TP_SIZE:=4}
            : ${NUM_NODES:=1}
            : ${MAX_MODEL_LEN:=131072}  # Max context: 128K
            ;;
        llama405b|llama3-405b)
            MODEL_PATH="meta-llama/Llama-3.1-405B-Instruct"
            : ${TP_SIZE:=8}
            : ${PP_SIZE:=2}
            : ${NUM_NODES:=4}
            : ${MAX_MODEL_LEN:=131072}  # Max context: 128K
            ;;
        
        # DeepSeek Models
        deepseek-v3|deepseekv3)
            MODEL_PATH="deepseek-ai/DeepSeek-V3"
            : ${TP_SIZE:=8}
            : ${EP_SIZE:=8}
            : ${NUM_NODES:=8}
            : ${MAX_MODEL_LEN:=65536}  # Max context: 64K
            IS_MOE=1
            ;;
        
        "")
            # No preset, use MODEL_PATH directly
            ;;
        *)
            echo "Unknown model preset: $MODEL"
            echo "Available presets:"
            echo "  Qwen Dense: qwen32b (Qwen2.5), qwen3-32b (Qwen3)"
            echo "  Qwen MoE:   qwen3-30b, qwen3-235b"
            echo "  LLaMA:      llama8b, llama70b, llama405b"
            echo "  DeepSeek:   deepseek-v3"
            echo "Or use MODEL_PATH=<path> directly"
            exit 1
            ;;
    esac
}

# ============================================================
# Cluster Detection and Configuration
# ============================================================
# Automatically detect cluster based on hostname or use CLUSTER env var
# Supported clusters:
#   lyris   - GB200 cluster (partition=gb200, account=coreai_dlalgo_llm, no gres)
#   oci-hsg - OCI HSG cluster (partition=batch, account=coreai_dlalgo_nemorl, gres=gpu:N)
#
# Override with: CLUSTER=lyris ./run_vllm_benchmark.sh run-random

detect_cluster() {
    if [ -n "${CLUSTER:-}" ]; then
        echo "$CLUSTER"
        return
    fi
    
    local hostname=$(hostname)
    
    # Detect based on hostname patterns
    if [[ "$hostname" == *"lyris"* ]] || [[ "$hostname" == *"gb200"* ]]; then
        echo "lyris"
    elif [[ "$hostname" == *"eos"* ]] || [[ "$hostname" == *"hsg"* ]]; then
        echo "oci-hsg"
    else
        # Default to lyris if running on login node (check SLURM partition availability)
        if sinfo -p gb200 &>/dev/null 2>&1; then
            echo "lyris"
        else
            echo "oci-hsg"
        fi
    fi
}

apply_cluster_config() {
    local cluster=$(detect_cluster)
    
    case "$cluster" in
        lyris)
            # Lyris GB200 cluster
            : ${SLURM_PARTITION:=gb200}
            : ${SLURM_ACCOUNT:=coreai_dlalgo_llm}
            : ${USE_GRES:=0}  # No --gres option
            : ${GPUS_PER_NODE:=4}
            # Use Lyris-accessible path for cache
            : ${JOB_CACHE_DIR:=/lustre/fsw/coreai_dlalgo_llm/users/sna/job_cache}
            ;;
        oci-hsg|*)
            # OCI HSG cluster (default)
            : ${SLURM_PARTITION:=batch}
            : ${SLURM_ACCOUNT:=coreai_dlalgo_nemorl}
            : ${USE_GRES:=1}  # Use --gres=gpu:N
            : ${GPUS_PER_NODE:=4}
            # OCI path for cache
            : ${JOB_CACHE_DIR:=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/job_cache}
            ;;
    esac
    
    echo "Detected cluster: $cluster"
}

# Apply cluster configuration
apply_cluster_config

# Build GRES option (empty for Lyris, --gres=gpu:N for others)
get_gres_option() {
    if [ "${USE_GRES:-1}" == "1" ]; then
        echo "--gres=gpu:$GPUS_PER_NODE"
    else
        echo ""
    fi
}

# ============================================================
# Configuration
# ============================================================

# Apply model preset first (sets defaults)
apply_model_preset

# Override with environment variables if set
NUM_NODES=${NUM_NODES:-1}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-32B-Instruct}

# Parallelism (may be set by preset)
TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}

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
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}  # Default for most models
MAX_TOKENS=${MAX_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-1.0}

# ============================================================
# Experiment directory (for logs and results)
# ============================================================
# Results go to: $EXPERIMENT_DIR/<bench_type>/<job_id>-logs/
# This is relative to SCRIPT_DIR so collect_vLLM_results.py can find them
EXPERIMENT_DIR=${EXPERIMENT_DIR:-$SCRIPT_DIR/../vllm_standalone_perf_exp}

# ============================================================
# Weights & Biases (wandb) Configuration
# ============================================================
# Set WANDB_UPLOAD=0 to disable automatic upload
# Set WANDB_PROJECT to change the project name (default: vllm-benchmark)
# Set WANDB_ENTITY to change the team/username
WANDB_UPLOAD=${WANDB_UPLOAD:-1}
WANDB_PROJECT=${WANDB_PROJECT:-vllm-benchmark}
WANDB_ENTITY=${WANDB_ENTITY:-}

# ============================================================
# Job naming helpers
# ============================================================
get_model_short_name() {
    local model_path="$1"
    local short_name
    
    # Extract model name from path (e.g., "Qwen/Qwen3-32B" -> "Qwen3-32B")
    short_name="${model_path##*/}"
    
    # Further shorten common patterns
    short_name="${short_name//-Instruct/}"
    short_name="${short_name//Llama-3.1-/Llama3-}"
    short_name="${short_name//Llama-3-/Llama3-}"
    short_name="${short_name//meta-llama/}"
    
    # Truncate to max 15 chars to leave room for parallelism info
    if [ ${#short_name} -gt 15 ]; then
        short_name="${short_name:0:15}"
    fi
    
    echo "$short_name"
}

# Build job name with model and parallelism info
# Format: <bench_type>-<model>-<nodes>n<tp>t<pp>p
# Example: off-Qwen3-32B-4n1t1p (offline, Qwen3-32B, 4 nodes, TP=1, PP=1)
build_job_name() {
    local bench_type="$1"  # off, tput, online
    local model_short="$2"
    local nodes="$3"
    local tp="$4"
    local pp="$5"
    
    # Build parallelism suffix: <nodes>n<tp>t<pp>p
    local para_suffix="${nodes}n${tp}t${pp}p"
    
    echo "${bench_type}-${model_short}-${para_suffix}"
}

# Build parallelism directory name
# Format: <nodes>n<tp>t<pp>p (e.g., 4n1t1p)
build_parallelism_dir() {
    local nodes="$1"
    local tp="$2"
    local pp="$3"
    
    echo "${nodes}n${tp}t${pp}p"
}

# Get model short name for job naming
MODEL_SHORT_NAME=$(get_model_short_name "$MODEL_PATH")

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
        GRES_OPT=$(get_gres_option)
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
            $GRES_OPT \
            --time=00:30:00 \
            --account=$SLURM_ACCOUNT \
            --partition=$SLURM_PARTITION \
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
        if [ "$EP_SIZE" -gt 1 ]; then
            echo "Parallelism: TP=$TP_SIZE, PP=$PP_SIZE, EP=$EP_SIZE, DP=$DP_SIZE (MoE, auto-calculated)"
        else
            echo "Parallelism: TP=$TP_SIZE, PP=$PP_SIZE, DP=$DP_SIZE (auto-calculated)"
        fi
        echo "Prompts: $NUM_PROMPTS × $NUM_GENERATIONS"
        echo "Prompts file: $PROMPTS_FILE"
        echo "============================================================"
        
        if [ ! -f "$PROMPTS_FILE" ]; then
            echo "Warning: Prompts file not found. Run '$0 prepare' first or use 'run-random'."
            echo "Proceeding with random prompts..."
        fi
        
        cd "$SCRIPT_DIR"
        
        GRES_OPT=$(get_gres_option)
        JOB_NAME=$(build_job_name "off" "$MODEL_SHORT_NAME" "$NUM_NODES" "$TP_SIZE" "$PP_SIZE")
        PARALLELISM_DIR=$(build_parallelism_dir "$NUM_NODES" "$TP_SIZE" "$PP_SIZE")
        MODEL_PATH="$MODEL_PATH" \
        TENSOR_PARALLEL_SIZE=$TP_SIZE \
        PIPELINE_PARALLEL_SIZE=$PP_SIZE \
        EXPERT_PARALLEL_SIZE=$EP_SIZE \
        GPUS_PER_NODE=$GPUS_PER_NODE \
        GPU_MODEL="${GPU_MODEL:-unknown}" \
        NUM_PROMPTS=$NUM_PROMPTS \
        NUM_GENERATIONS=$NUM_GENERATIONS \
        MAX_MODEL_LEN=$MAX_MODEL_LEN \
        MAX_TOKENS=$MAX_TOKENS \
        TEMPERATURE=$TEMPERATURE \
        PROMPTS_FILE="$PROMPTS_FILE" \
        EXPERIMENT_DIR="$EXPERIMENT_DIR" \
        SCRIPT_DIR="$SCRIPT_DIR" \
        MODEL_SHORT_NAME="$MODEL_SHORT_NAME" \
        PARALLELISM_DIR="$PARALLELISM_DIR" \
        WANDB_UPLOAD="$WANDB_UPLOAD" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        WANDB_ENTITY="$WANDB_ENTITY" \
        JOB_CACHE_DIR="$JOB_CACHE_DIR" \
        sbatch \
            --nodes=$NUM_NODES \
            $GRES_OPT \
            --partition=$SLURM_PARTITION \
            --account=$SLURM_ACCOUNT \
            --job-name="$JOB_NAME" \
            -o /dev/null -e /dev/null \
            benchmark_vllm_offline.sbatch
        
        echo ""
        echo "Logs will be in: $EXPERIMENT_DIR/offline/$MODEL_SHORT_NAME/$PARALLELISM_DIR/<JOB_ID>-logs/"
        echo "  - slurm-<JOB_ID>.out  (SLURM + job output combined)"
        echo "  - results.json        (benchmark results)"
        [ "$WANDB_UPLOAD" == "1" ] && echo "  - wandb: $WANDB_PROJECT (auto-upload enabled)"
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
        if [ "$EP_SIZE" -gt 1 ]; then
            echo "Parallelism: TP=$TP_SIZE, PP=$PP_SIZE, EP=$EP_SIZE, DP=$DP_SIZE (MoE, auto-calculated)"
        else
            echo "Parallelism: TP=$TP_SIZE, PP=$PP_SIZE, DP=$DP_SIZE (auto-calculated)"
        fi
        echo "Prompts: $NUM_PROMPTS × $NUM_GENERATIONS (random)"
        echo "============================================================"
        
        cd "$SCRIPT_DIR"
        
        GRES_OPT=$(get_gres_option)
        JOB_NAME=$(build_job_name "off" "$MODEL_SHORT_NAME" "$NUM_NODES" "$TP_SIZE" "$PP_SIZE")
        PARALLELISM_DIR=$(build_parallelism_dir "$NUM_NODES" "$TP_SIZE" "$PP_SIZE")
        MODEL_PATH="$MODEL_PATH" \
        TENSOR_PARALLEL_SIZE=$TP_SIZE \
        PIPELINE_PARALLEL_SIZE=$PP_SIZE \
        EXPERT_PARALLEL_SIZE=$EP_SIZE \
        GPUS_PER_NODE=$GPUS_PER_NODE \
        GPU_MODEL="${GPU_MODEL:-unknown}" \
        NUM_PROMPTS=$NUM_PROMPTS \
        NUM_GENERATIONS=$NUM_GENERATIONS \
        MAX_MODEL_LEN=$MAX_MODEL_LEN \
        MAX_TOKENS=$MAX_TOKENS \
        TEMPERATURE=$TEMPERATURE \
        PROMPTS_FILE="" \
        RANDOM_INPUT_LEN=150 \
        EXPERIMENT_DIR="$EXPERIMENT_DIR" \
        SCRIPT_DIR="$SCRIPT_DIR" \
        MODEL_SHORT_NAME="$MODEL_SHORT_NAME" \
        PARALLELISM_DIR="$PARALLELISM_DIR" \
        WANDB_UPLOAD="$WANDB_UPLOAD" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        WANDB_ENTITY="$WANDB_ENTITY" \
        JOB_CACHE_DIR="$JOB_CACHE_DIR" \
        sbatch \
            --nodes=$NUM_NODES \
            $GRES_OPT \
            --partition=$SLURM_PARTITION \
            --account=$SLURM_ACCOUNT \
            --job-name="$JOB_NAME" \
            -o /dev/null -e /dev/null \
            benchmark_vllm_offline.sbatch
        
        echo ""
        echo "Logs will be in: $EXPERIMENT_DIR/offline/$MODEL_SHORT_NAME/$PARALLELISM_DIR/<JOB_ID>-logs/"
        [ "$WANDB_UPLOAD" == "1" ] && echo "  - wandb: $WANDB_PROJECT (auto-upload enabled)"
        ;;
    
    run-throughput)
        # ============================================================
        # AUTO-CALCULATE required nodes for throughput benchmark
        # vllm bench throughput does NOT support DP, only TP*PP GPUs are used
        # ============================================================
        REQUIRED_GPUS=$((TP_SIZE * PP_SIZE))
        REQUIRED_NODES=$(( (REQUIRED_GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))  # ceil division
        
        if [ "$NUM_NODES" -gt "$REQUIRED_NODES" ]; then
            echo "⚠️  WARNING: You requested $NUM_NODES nodes ($((NUM_NODES * GPUS_PER_NODE)) GPUs)"
            echo "   But vllm bench throughput only uses TP×PP = $REQUIRED_GPUS GPUs"
            echo "   Auto-adjusting to $REQUIRED_NODES node(s) to avoid GPU waste."
            NUM_NODES=$REQUIRED_NODES
        fi
        
        TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
        GPUS_PER_INSTANCE=$((TP_SIZE * PP_SIZE))
        
        # Throughput benchmark parameters (vllm bench throughput)
        # Reference: https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/throughput.py
        INPUT_LENS=${INPUT_LENS:-"64 100 150"}
        OUTPUT_LENS=${OUTPUT_LENS:-"2048 4096"}
        THROUGHPUT_NUM_PROMPTS=${THROUGHPUT_NUM_PROMPTS:-64}
        # NeMo-RL: num_generations_per_prompt (n parameter for vLLM)
        THROUGHPUT_NUM_GENERATIONS=${THROUGHPUT_NUM_GENERATIONS:-32}
        # RANDOM_RANGE_RATIO: Input length variance (0.0 = exact length)
        # Note: ignore_eos=True is HARDCODED in vLLM, output is always exact
        RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.0}
        
        echo "============================================================"
        echo "Submitting vLLM Official Throughput Benchmark"
        echo "============================================================"
        echo "Nodes: $NUM_NODES (${GPUS_PER_NODE} GPUs each)"
        echo "Total GPUs: $TOTAL_GPUS"
        echo "Model: $MODEL_PATH"
        if [ "$EP_SIZE" -gt 1 ]; then
            echo "Parallelism: TP=$TP_SIZE, PP=$PP_SIZE, EP=$EP_SIZE (MoE)"
        else
            echo "Parallelism: TP=$TP_SIZE, PP=$PP_SIZE"
        fi
        echo "Input lengths: $INPUT_LENS"
        echo "Output lengths: $OUTPUT_LENS"
        echo "Num prompts per config: $THROUGHPUT_NUM_PROMPTS"
        if [ "$RANDOM_RANGE_RATIO" == "0.0" ] || [ "$RANDOM_RANGE_RATIO" == "0" ]; then
            echo "Length mode: EXACT input + EXACT output"
        else
            echo "Length mode: VARIABLE input (±${RANDOM_RANGE_RATIO}) + EXACT output"
        fi
        echo "(Note: vLLM hardcodes ignore_eos=True, output is always exact)"
        echo "============================================================"
        
        cd "$SCRIPT_DIR"
        
        GRES_OPT=$(get_gres_option)
        JOB_NAME=$(build_job_name "tput" "$MODEL_SHORT_NAME" "$NUM_NODES" "$TP_SIZE" "$PP_SIZE")
        PARALLELISM_DIR=$(build_parallelism_dir "$NUM_NODES" "$TP_SIZE" "$PP_SIZE")
        MODEL_PATH="$MODEL_PATH" \
        TENSOR_PARALLEL_SIZE=$TP_SIZE \
        PIPELINE_PARALLEL_SIZE=$PP_SIZE \
        EXPERT_PARALLEL_SIZE=$EP_SIZE \
        GPUS_PER_NODE=$GPUS_PER_NODE \
        GPU_MODEL="${GPU_MODEL:-unknown}" \
        INPUT_LENS="$INPUT_LENS" \
        OUTPUT_LENS="$OUTPUT_LENS" \
        NUM_PROMPTS=$THROUGHPUT_NUM_PROMPTS \
        NUM_GENERATIONS_PER_PROMPT=$THROUGHPUT_NUM_GENERATIONS \
        RANDOM_RANGE_RATIO=$RANDOM_RANGE_RATIO \
        MAX_MODEL_LEN=$MAX_MODEL_LEN \
        EXPERIMENT_DIR="$EXPERIMENT_DIR" \
        SCRIPT_DIR="$SCRIPT_DIR" \
        MODEL_SHORT_NAME="$MODEL_SHORT_NAME" \
        PARALLELISM_DIR="$PARALLELISM_DIR" \
        WANDB_UPLOAD="$WANDB_UPLOAD" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        WANDB_ENTITY="$WANDB_ENTITY" \
        JOB_CACHE_DIR="$JOB_CACHE_DIR" \
        sbatch \
            --nodes=$NUM_NODES \
            $GRES_OPT \
            --partition=$SLURM_PARTITION \
            --account=$SLURM_ACCOUNT \
            --job-name="$JOB_NAME" \
            -o /dev/null -e /dev/null \
            benchmark_vllm_throughput.sbatch
        
        echo ""
        echo "Logs will be in: $EXPERIMENT_DIR/throughput/$MODEL_SHORT_NAME/$PARALLELISM_DIR/<JOB_ID>-logs/"
        echo "  - result_ISL*_OSL*.txt   (per-config results)"
        echo "  - results_summary.txt    (summary CSV)"
        echo "  - results.json           (JSON results)"
        [ "$WANDB_UPLOAD" == "1" ] && echo "  - wandb: $WANDB_PROJECT (auto-upload enabled)"
        ;;
    
    run-online)
        # Calculate DP for display
        TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
        GPUS_PER_INSTANCE=$((TP_SIZE * PP_SIZE))
        DP_SIZE=$((TOTAL_GPUS / GPUS_PER_INSTANCE))
        
        # Online serving parameters
        MAX_CONCURRENCY=${MAX_CONCURRENCY:-64}
        ONLINE_NUM_PROMPTS=${ONLINE_NUM_PROMPTS:-$((MAX_CONCURRENCY * 5))}
        MAX_ISL=${MAX_ISL:-150}
        MAX_OSL=${MAX_OSL:-"1000 2000"}
        
        echo "============================================================"
        echo "Submitting vLLM Online Serving Benchmark"
        echo "============================================================"
        echo "Nodes: $NUM_NODES (${GPUS_PER_NODE} GPUs each)"
        echo "Total GPUs: $TOTAL_GPUS"
        echo "Model: $MODEL_PATH"
        if [ "$EP_SIZE" -gt 1 ]; then
            echo "Parallelism: TP=$TP_SIZE, PP=$PP_SIZE, EP=$EP_SIZE (MoE)"
        else
            echo "Parallelism: TP=$TP_SIZE, PP=$PP_SIZE"
        fi
        echo "Max concurrency: $MAX_CONCURRENCY"
        echo "Num prompts: $ONLINE_NUM_PROMPTS"
        echo "ISL: $MAX_ISL, OSL: $MAX_OSL"
        echo "============================================================"
        
        cd "$SCRIPT_DIR"
        
        GRES_OPT=$(get_gres_option)
        JOB_NAME=$(build_job_name "online" "$MODEL_SHORT_NAME" "$NUM_NODES" "$TP_SIZE" "$PP_SIZE")
        PARALLELISM_DIR=$(build_parallelism_dir "$NUM_NODES" "$TP_SIZE" "$PP_SIZE")
        MODEL_PATH="$MODEL_PATH" \
        TENSOR_PARALLEL_SIZE=$TP_SIZE \
        PIPELINE_PARALLEL_SIZE=$PP_SIZE \
        EXPERT_PARALLEL_SIZE=$EP_SIZE \
        GPUS_PER_NODE=$GPUS_PER_NODE \
        GPU_MODEL="${GPU_MODEL:-unknown}" \
        MAX_CONCURRENCY=$MAX_CONCURRENCY \
        NUM_PROMPTS=$ONLINE_NUM_PROMPTS \
        MAX_ISL="$MAX_ISL" \
        MAX_OSL="$MAX_OSL" \
        MAX_MODEL_LEN=$MAX_MODEL_LEN \
        EXPERIMENT_DIR="$EXPERIMENT_DIR" \
        SCRIPT_DIR="$SCRIPT_DIR" \
        MODEL_SHORT_NAME="$MODEL_SHORT_NAME" \
        PARALLELISM_DIR="$PARALLELISM_DIR" \
        WANDB_UPLOAD="$WANDB_UPLOAD" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        WANDB_ENTITY="$WANDB_ENTITY" \
        JOB_CACHE_DIR="$JOB_CACHE_DIR" \
        sbatch \
            --nodes=$NUM_NODES \
            $GRES_OPT \
            --partition=$SLURM_PARTITION \
            --account=$SLURM_ACCOUNT \
            --job-name="$JOB_NAME" \
            -o /dev/null -e /dev/null \
            benchmark_vllm_online.sbatch
        
        echo ""
        echo "Logs will be in: $EXPERIMENT_DIR/online/$MODEL_SHORT_NAME/$PARALLELISM_DIR/<JOB_ID>-logs/"
        echo "  - result_ISL*_OSL*.txt   (per-config results)"
        echo "  - results_summary.txt    (summary CSV)"
        [ "$WANDB_UPLOAD" == "1" ] && echo "  - wandb: $WANDB_PROJECT (auto-upload enabled)"
        ;;
    
    build)
        echo "============================================================"
        echo "Building venv for vLLM benchmark (one-time setup)"
        echo "============================================================"
        echo "This creates a persistent venv that will be reused."
        echo "============================================================"
        
        cd "$SCRIPT_DIR"
        
        GRES_OPT=$(get_gres_option)
        DO_BUILD=1 \
        VLLM_INSTALL_FROM_SOURCE="${VLLM_INSTALL_FROM_SOURCE:-0}" \
        VLLM_GIT_REPO="${VLLM_GIT_REPO:-https://github.com/vllm-project/vllm.git}" \
        VLLM_GIT_BRANCH="${VLLM_GIT_BRANCH:-main}" \
        EXPERIMENT_DIR="$EXPERIMENT_DIR" \
        SCRIPT_DIR="$SCRIPT_DIR" \
        JOB_CACHE_DIR="$JOB_CACHE_DIR" \
        sbatch \
            --nodes=1 \
            $GRES_OPT \
            --time=01:00:00 \
            --job-name=vllm-build \
            --partition=$SLURM_PARTITION \
            --account=$SLURM_ACCOUNT \
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
        
        GRES_OPT=$(get_gres_option)
        DO_BUILD=1 \
        VLLM_INSTALL_FROM_SOURCE=1 \
        VLLM_GIT_REPO="${VLLM_GIT_REPO:-https://github.com/vllm-project/vllm.git}" \
        VLLM_GIT_BRANCH="${VLLM_GIT_BRANCH:-main}" \
        EXPERIMENT_DIR="$EXPERIMENT_DIR" \
        SCRIPT_DIR="$SCRIPT_DIR" \
        JOB_CACHE_DIR="$JOB_CACHE_DIR" \
        sbatch \
            --nodes=1 \
            $GRES_OPT \
            --time=01:00:00 \
            --job-name=vllm-build-custom \
            --partition=$SLURM_PARTITION \
            --account=$SLURM_ACCOUNT \
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
        
        GRES_OPT=$(get_gres_option)
        DO_BUILD=1 \
        VLLM_LOCAL_PATH="$VLLM_LOCAL_PATH" \
        EXPERIMENT_DIR="$EXPERIMENT_DIR" \
        SCRIPT_DIR="$SCRIPT_DIR" \
        JOB_CACHE_DIR="$JOB_CACHE_DIR" \
        sbatch \
            --nodes=1 \
            $GRES_OPT \
            --time=01:00:00 \
            --job-name=vllm-build-local \
            --partition=$SLURM_PARTITION \
            --account=$SLURM_ACCOUNT \
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
        
        GRES_OPT=$(get_gres_option)
        JOB_NAME=$(build_job_name "custom" "$MODEL_SHORT_NAME" "$NUM_NODES" "$TP_SIZE" "$PP_SIZE")
        PARALLELISM_DIR=$(build_parallelism_dir "$NUM_NODES" "$TP_SIZE" "$PP_SIZE")
        MODEL_PATH="$MODEL_PATH" \
        TENSOR_PARALLEL_SIZE=$TP_SIZE \
        PIPELINE_PARALLEL_SIZE=$PP_SIZE \
        EXPERT_PARALLEL_SIZE=$EP_SIZE \
        NUM_PROMPTS=$NUM_PROMPTS \
        NUM_GENERATIONS=$NUM_GENERATIONS \
        EXPERIMENT_DIR="$EXPERIMENT_DIR" \
        SCRIPT_DIR="$SCRIPT_DIR" \
        MODEL_SHORT_NAME="$MODEL_SHORT_NAME" \
        PARALLELISM_DIR="$PARALLELISM_DIR" \
        PROMPTS_FILE="$PROMPTS_FILE" \
        JOB_CACHE_DIR="$JOB_CACHE_DIR" \
        sbatch \
            --nodes=$NUM_NODES \
            $GRES_OPT \
            --partition=$SLURM_PARTITION \
            --account=$SLURM_ACCOUNT \
            --job-name="$JOB_NAME" \
            -o /dev/null -e /dev/null \
            benchmark_vllm_offline.sbatch
        ;;
    
    help|*)
        echo "vLLM Benchmark Runner (Offline, Online & Throughput)"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Offline Benchmark Commands (batch processing, max throughput):"
        echo "  prepare         Prepare prompts from NeMo-RL dataset (submits SLURM job)"
        echo "  run             Run offline benchmark with prepared prompts"
        echo "  run-random      Run offline benchmark with random prompts"
        echo ""
        echo "Throughput Benchmark Commands (vllm bench throughput - fixed ISL/OSL):"
        echo "  run-throughput  Run official vLLM throughput benchmark (sweeps ISL/OSL)"
        echo ""
        echo "Online Benchmark Commands (server mode, latency measurement):"
        echo "  run-online      Run online serving benchmark (vllm serve + vllm bench)"
        echo ""
        echo "Build Commands:"
        echo "  build           Build venv for benchmark (one-time, uses container vLLM)"
        echo "  build-custom    Build venv with custom vLLM from GitHub"
        echo "  build-local     Build venv with local vLLM source (for development)"
        echo "  run-custom-vllm Run benchmark (after build-custom or build-local)"
        echo ""
        echo "Cluster configuration (auto-detected or set via CLUSTER env var):"
        echo "  CLUSTER         Cluster name: lyris, oci-hsg (auto-detected from hostname)"
        echo "  SLURM_PARTITION Partition name (lyris: gb200, oci-hsg: batch)"
        echo "  SLURM_ACCOUNT   Account name (lyris: coreai_dlalgo_llm, oci-hsg: coreai_dlalgo_nemorl)"
        echo "  USE_GRES        Use --gres option: 0=no, 1=yes (lyris: 0, oci-hsg: 1)"
        echo ""
        echo "Environment variables:"
        echo "  MODEL           Model preset name (qwen32b, qwen3-30b, etc.)"
        echo "  MODEL_PATH      Model path or HuggingFace name (overrides MODEL)"
        echo "  NUM_NODES       Number of nodes (auto-set by preset)"
        echo "  TP_SIZE         Tensor parallel size (auto-set by preset)"
        echo "  PP_SIZE         Pipeline parallel size (default: 1)"
        echo "  EP_SIZE         Expert parallel size for MoE (auto-set by preset)"
        echo "  # DP is auto-calculated: DP = (NUM_NODES × GPUS_PER_NODE) / (TP × PP)"
        echo ""
        echo "HuggingFace variables:"
        echo "  HF_HOME         HuggingFace home directory (for model cache)"
        echo "  HF_TOKEN        HuggingFace token (for gated models)"
        echo ""
        echo "Offline-specific variables:"
        echo "  NUM_PROMPTS     Number of unique prompts (default: 64)"
        echo "  NUM_GENERATIONS Generations per prompt (default: 32)"
        echo ""
        echo "Throughput-specific variables (run-throughput):"
        echo "  INPUT_LENS      Input sequence lengths, space-separated (default: '128 256 512 1024')"
        echo "  OUTPUT_LENS     Output sequence lengths, space-separated (default: '128 256 512 1024 2048')"
        echo "  THROUGHPUT_NUM_PROMPTS  Prompts per config (default: 1000)"
        echo "  RANDOM_RANGE_RATIO  Input length variance (default: 0.0 = exact)"
        echo "                  0.0 = EXACT input-len tokens"
        echo "                  0.2 = input-len * [0.8, 1.2] range"
        echo "  (Note: ignore_eos=True is hardcoded in vLLM, output is always exact)"
        echo ""
        echo "Online-specific variables (run-online):"
        echo "  MAX_CONCURRENCY Max concurrent requests (default: 64)"
        echo "  MAX_ISL         Input sequence length (default: 150)"
        echo "  MAX_OSL         Output sequence lengths, space-separated (default: '1000 2000')"
        echo ""
        echo "Weights & Biases (wandb) integration:"
        echo "  WANDB_UPLOAD    Enable auto-upload: 1=yes (default), 0=no"
        echo "  WANDB_PROJECT   wandb project name (default: vllm-benchmark)"
        echo "  WANDB_ENTITY    wandb team/username (optional)"
        echo ""
        echo "Parallelism example (4 nodes × 4 GPUs = 16 GPUs):"
        echo "  TP=4, PP=1 → DP=4 (4 vLLM instances, each using 4 GPUs)"
        echo "  TP=8, PP=1 → DP=2 (2 vLLM instances, each using 8 GPUs)"
        echo ""
        echo "Examples:"
        echo "  # Offline benchmark (using model preset)"
        echo "  MODEL=qwen32b $0 run-random"
        echo ""
        echo "  # Official throughput benchmark (fixed ISL/OSL sweep)"
        echo "  MODEL=qwen32b $0 run-throughput"
        echo ""
        echo "  # Throughput with custom ISL/OSL ranges"
        echo "  MODEL=qwen32b INPUT_LENS='256 512' OUTPUT_LENS='512 1024 2048' $0 run-throughput"
        echo ""
        echo "  # Online benchmark (server mode)"
        echo "  MODEL=qwen32b $0 run-online"
        echo ""
        echo "  # Online with custom ISL/OSL"
        echo "  MODEL=qwen3-30b MAX_ISL=200 MAX_OSL='500 1000 2000' $0 run-online"
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
        echo ""
        echo "========================================================"
        echo "Model Presets (Easy Mode!):"
        echo "========================================================"
        echo ""
        echo "  # Just use MODEL=<preset> - settings are auto-configured!"
        echo ""
        echo "  MODEL=qwen32b $0 run-random      # Qwen2.5-32B (1 node, TP=4)"
        echo "  MODEL=qwen3-30b $0 run-random    # Qwen3-30B MoE (1 node, TP=4, EP=4)"
        echo "  MODEL=qwen3-235b $0 run-random   # Qwen3-235B MoE (2 nodes, TP=8, EP=8)"
        echo "  MODEL=llama70b $0 run-random     # Llama-3.1-70B (1 node, TP=4)"
        echo "  MODEL=deepseek-v3 $0 run-random  # DeepSeek-V3 (8 nodes, TP=8, EP=8)"
        echo ""
        echo "  # Override preset defaults if needed:"
        echo "  MODEL=qwen3-30b NUM_NODES=2 $0 run-random"
        echo ""
        echo "  # Or use custom model path:"
        echo "  MODEL_PATH=/path/to/model TP_SIZE=4 $0 run-random"
        echo ""
        echo "Available presets:"
        echo "  qwen32b     - Qwen2.5-32B-Instruct (32B Dense)"
        echo "  qwen3-30b   - Qwen3-30B-A3B (30B MoE, 3B active)"
        echo "  qwen3-235b  - Qwen3-235B-A22B (235B MoE, 22B active)"
        echo "  llama70b    - Llama-3.1-70B-Instruct (70B Dense)"
        echo "  llama405b   - Llama-3.1-405B-Instruct (405B Dense)"
        echo "  deepseek-v3 - DeepSeek-V3 (671B MoE, 37B active)"
        echo ""
        ;;
esac

