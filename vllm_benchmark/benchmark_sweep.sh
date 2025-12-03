#!/bin/bash
# ============================================================
# vLLM Throughput Benchmark Sweep Script
# ============================================================
# This script runs vllm bench throughput for multiple models
# with configurable parallelism settings.
#
# Usage:
#   ./benchmark_sweep.sh                    # Run all configured benchmarks
#   ./benchmark_sweep.sh --dry-run          # Show what would be run without submitting
#   ./benchmark_sweep.sh --model qwen32b    # Run only specific model
#   ./benchmark_sweep.sh --list             # List all available configurations
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# Default Settings
# ============================================================
DRY_RUN=0
FILTER_MODEL=""
GPUS_PER_NODE=4  # GB200: 4, H100: 8

# ISL/OSL configurations for throughput benchmark
DEFAULT_INPUT_LENS="512 1024"
DEFAULT_OUTPUT_LENS="512 1024"
DEFAULT_NUM_PROMPTS=1000

# ============================================================
# Model Configurations
# ============================================================
# Format: MODEL_NAME|HF_PATH|TP|PP|EP|NODES|DESCRIPTION
#
# Notes:
# - DP is auto-calculated: DP = (NODES × GPUS_PER_NODE) / (TP × PP)
# - EP (Expert Parallelism) is for MoE models only
# - For MoE: typically EP = TP (experts distributed within TP group)

declare -a CONFIGS=(
    # Dense Models - LLaMA
    "llama3-8b|meta-llama/Llama-3.1-8B-Instruct|1|1|1|1|LLaMA3 8B (1 GPU)"
    "llama3-70b|meta-llama/Llama-3.1-70B-Instruct|4|1|1|1|LLaMA3 70B (4 GPUs, TP=4)"
    "llama3-70b-2n|meta-llama/Llama-3.1-70B-Instruct|4|1|1|2|LLaMA3 70B (8 GPUs, TP=4, DP=2)"
    
    # Dense Models - Qwen3
    "qwen3-32b|Qwen/Qwen3-32B|4|1|1|1|Qwen3 32B (4 GPUs, TP=4)"
    "qwen3-32b-2n|Qwen/Qwen3-32B|4|1|1|2|Qwen3 32B (8 GPUs, TP=4, DP=2)"
    
    # MoE Models - Qwen3
    "qwen3-30b|Qwen/Qwen3-30B-A3B|4|1|4|1|Qwen3 30B MoE (4 GPUs, TP=4, EP=4)"
    "qwen3-30b-2n|Qwen/Qwen3-30B-A3B|4|1|4|2|Qwen3 30B MoE (8 GPUs, TP=4, EP=4, DP=2)"
    "qwen3-235b|Qwen/Qwen3-235B-A22B|8|1|8|2|Qwen3 235B MoE (8 GPUs, TP=8, EP=8)"
    "qwen3-235b-4n|Qwen/Qwen3-235B-A22B|8|1|8|4|Qwen3 235B MoE (16 GPUs, TP=8, EP=8, DP=2)"
    
    # MoE Models - DeepSeek
    "deepseek-v3|deepseek-ai/DeepSeek-V3|8|1|8|8|DeepSeek V3 (32 GPUs, TP=8, EP=8, DP=4)"
    "deepseek-v3-16n|deepseek-ai/DeepSeek-V3|8|1|8|16|DeepSeek V3 (64 GPUs, TP=8, EP=8, DP=8)"
)

# ============================================================
# Parse Arguments
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run|-n)
            DRY_RUN=1
            shift
            ;;
        --model|-m)
            FILTER_MODEL="$2"
            shift 2
            ;;
        --list|-l)
            echo "Available model configurations:"
            echo ""
            printf "%-20s %-8s %-8s %-8s %-8s %s\n" "NAME" "TP" "PP" "EP" "NODES" "DESCRIPTION"
            echo "─────────────────────────────────────────────────────────────────────────────────────────"
            for config in "${CONFIGS[@]}"; do
                IFS='|' read -r name path tp pp ep nodes desc <<< "$config"
                total_gpus=$((nodes * GPUS_PER_NODE))
                gpus_per_inst=$((tp * pp))
                dp=$((total_gpus / gpus_per_inst))
                printf "%-20s TP=%-4s PP=%-4s EP=%-4s N=%-4s %s (DP=%d)\n" "$name" "$tp" "$pp" "$ep" "$nodes" "$desc" "$dp"
            done
            exit 0
            ;;
        --help|-h)
            echo "vLLM Throughput Benchmark Sweep"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run, -n     Show what would be run without submitting jobs"
            echo "  --model, -m NAME  Run only specific model configuration"
            echo "  --list, -l        List all available configurations"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  INPUT_LENS        Override input sequence lengths (default: '$DEFAULT_INPUT_LENS')"
            echo "  OUTPUT_LENS       Override output sequence lengths (default: '$DEFAULT_OUTPUT_LENS')"
            echo "  NUM_PROMPTS       Override number of prompts (default: $DEFAULT_NUM_PROMPTS)"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run all benchmarks"
            echo "  $0 --dry-run                 # Preview without submitting"
            echo "  $0 --model qwen32b           # Run only qwen32b"
            echo "  $0 --model llama3            # Run all llama3 variants"
            echo "  INPUT_LENS='512 1024' $0     # Custom ISL range"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================
# Run Benchmarks
# ============================================================
INPUT_LENS="${INPUT_LENS:-$DEFAULT_INPUT_LENS}"
OUTPUT_LENS="${OUTPUT_LENS:-$DEFAULT_OUTPUT_LENS}"
NUM_PROMPTS="${NUM_PROMPTS:-$DEFAULT_NUM_PROMPTS}"

echo "============================================================"
echo "vLLM Throughput Benchmark Sweep"
echo "============================================================"
echo "Input lengths:  $INPUT_LENS"
echo "Output lengths: $OUTPUT_LENS"
echo "Num prompts:    $NUM_PROMPTS"
echo "GPUs per node:  $GPUS_PER_NODE"
if [ -n "$FILTER_MODEL" ]; then
    echo "Filter:         $FILTER_MODEL"
fi
if [ $DRY_RUN -eq 1 ]; then
    echo "Mode:           DRY RUN (no jobs will be submitted)"
fi
echo "============================================================"
echo ""

submitted=0
skipped=0

for config in "${CONFIGS[@]}"; do
    IFS='|' read -r name path tp pp ep nodes desc <<< "$config"
    
    # Filter by model name if specified
    if [ -n "$FILTER_MODEL" ] && [[ ! "$name" == *"$FILTER_MODEL"* ]]; then
        continue
    fi
    
    # Calculate derived values
    total_gpus=$((nodes * GPUS_PER_NODE))
    gpus_per_inst=$((tp * pp))
    dp=$((total_gpus / gpus_per_inst))
    
    echo "────────────────────────────────────────────────────────────"
    echo "Model: $name"
    echo "  Path: $path"
    echo "  Config: ${nodes}N × ${GPUS_PER_NODE}G = ${total_gpus} GPUs"
    echo "  Parallelism: TP=$tp, PP=$pp, EP=$ep, DP=$dp"
    echo "  Description: $desc"
    
    if [ $DRY_RUN -eq 1 ]; then
        echo "  [DRY RUN] Would submit job"
        skipped=$((skipped + 1))
    else
        # Submit the job
        MODEL_PATH="$path" \
        TP_SIZE=$tp \
        PP_SIZE=$pp \
        EP_SIZE=$ep \
        NUM_NODES=$nodes \
        INPUT_LENS="$INPUT_LENS" \
        OUTPUT_LENS="$OUTPUT_LENS" \
        THROUGHPUT_NUM_PROMPTS=$NUM_PROMPTS \
        ./run_vllm_benchmark.sh run-throughput 2>&1 | grep -E "(Submitted|Job ID|Logs)" || true
        
        submitted=$((submitted + 1))
        
        # Small delay to avoid overwhelming SLURM
        sleep 1
    fi
    echo ""
done

echo "============================================================"
if [ $DRY_RUN -eq 1 ]; then
    echo "DRY RUN complete. Would have submitted $skipped jobs."
else
    echo "Submitted $submitted jobs."
    echo ""
    echo "Monitor progress:"
    echo "  squeue -u \$USER"
    echo ""
    echo "View results:"
    echo "  python collect_results.py --throughput"
fi
echo "============================================================"

