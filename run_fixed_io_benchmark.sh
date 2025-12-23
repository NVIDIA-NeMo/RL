#!/bin/bash
# ============================================================
# NeMo-RL Fixed Input/Output Length Benchmark
# ============================================================
# Runs GRPO training with fixed input/output lengths for fair
# performance comparison with:
#   - vLLM Standalone (generation throughput)
#   - Megatron-Bridge (training throughput)
#
# Configuration based on OpenMathInstruct-2 dataset characteristics:
#   - Input (problem): ~50-150 tokens (using 128)
#   - Output (solution): ~200-500+ tokens with CoT (using 3968)
#   - Total: 4096 (matches Megatron-Bridge seq_length)
#
# Reference: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2
#
# Usage:
#   ./run_fixed_io_benchmark.sh                    # Run all models
#   ./run_fixed_io_benchmark.sh --model qwen32b    # Run specific model
#   ./run_fixed_io_benchmark.sh --dry-run          # Show commands only
#   ./run_fixed_io_benchmark.sh --list             # List available models
#
# Custom I/O lengths:
#   INPUT_LEN=256 OUTPUT_LEN=3840 ./run_fixed_io_benchmark.sh
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# Default Configuration (OpenMathInstruct-2 optimized)
# ============================================================
INPUT_LEN=${INPUT_LEN:-128}
OUTPUT_LEN=${OUTPUT_LEN:-3968}
TOTAL_LEN=$((INPUT_LEN + OUTPUT_LEN))
WANDB_PROJECT=${WANDB_PROJECT:-sync-grpo-gb200-benchmark-fixedRLconfig}
# Models to benchmark (matching Megatron-Bridge configs)
DEFAULT_PRESETS="llama8b llama70b qwen30b qwen32b"

# ============================================================
# Parse Arguments
# ============================================================
DRY_RUN=""
FILTER_MODEL=""
LIST_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run|-n)
            DRY_RUN="--dry-run"
            echo "=== DRY RUN MODE ==="
            shift
            ;;
        --model|-m)
            FILTER_MODEL="$2"
            shift 2
            ;;
        --list|-l)
            LIST_ONLY=true
            shift
            ;;
        --input-length|-i)
            INPUT_LEN="$2"
            shift 2
            ;;
        --output-length|-o)
            OUTPUT_LEN="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run, -n          Show commands without executing"
            echo "  --model, -m MODEL      Run only specific model preset"
            echo "  --list, -l             List available model presets"
            echo "  --input-length, -i N   Set input length (default: 128)"
            echo "  --output-length, -o N  Set output length (default: 3968)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  INPUT_LEN=N            Set input length"
            echo "  OUTPUT_LEN=N           Set output length"
            echo ""
            echo "Available model presets:"
            echo "  llama8b    - LLaMA3.1-8B-Instruct (8 GPUs)"
            echo "  llama70b   - LLaMA3.1-70B-Instruct (16 GPUs)"
            echo "  qwen30b    - Qwen3-30B-A3B MoE (16 GPUs)"
            echo "  qwen32b    - Qwen3-32B (16 GPUs)"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run all models with default I/O"
            echo "  $0 --model qwen32b           # Run only Qwen3-32B"
            echo "  $0 --dry-run                 # Show commands without executing"
            echo "  INPUT_LEN=256 OUTPUT_LEN=3840 $0  # Custom I/O lengths"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Recalculate total after potential overrides
TOTAL_LEN=$((INPUT_LEN + OUTPUT_LEN))

# ============================================================
# List Available Presets
# ============================================================
if [[ "$LIST_ONLY" == "true" ]]; then
    echo ""
    echo "Available model presets for fixed I/O benchmark:"
    echo ""
    echo "  Preset      Model                      GPUs   Megatron-Bridge Config"
    echo "  --------    -------------------------  -----  ----------------------"
    echo "  llama8b     meta-llama/Llama-3.1-8B    8      llama31_8b_nemorl"
    echo "  llama70b    meta-llama/Llama-3.1-70B   16     llama31_70b_nemorl"
    echo "  qwen30b     Qwen/Qwen3-30B-A3B (MoE)   16     qwen3_30b_ep8"
    echo "  qwen32b     Qwen/Qwen3-32B             16     qwen3_32b_tp4"
    echo ""
    echo "Default I/O configuration (OpenMathInstruct-2):"
    echo "  Input:  ${INPUT_LEN} tokens"
    echo "  Output: ${OUTPUT_LEN} tokens"
    echo "  Total:  ${TOTAL_LEN} tokens (matches Megatron-Bridge seq_length)"
    echo ""
    exit 0
fi

# ============================================================
# Build Preset List
# ============================================================
if [[ -n "$FILTER_MODEL" ]]; then
    PRESETS="$FILTER_MODEL"
else
    PRESETS="$DEFAULT_PRESETS"
fi

# ============================================================
# Print Configuration Summary
# ============================================================
echo ""
echo "============================================================"
echo "NeMo-RL Fixed Input/Output Length Benchmark"
echo "============================================================"
echo ""
echo "Configuration (OpenMathInstruct-2 optimized):"
echo "  Input Length:  ${INPUT_LEN} tokens"
echo "  Output Length: ${OUTPUT_LEN} tokens"
echo "  Total Length:  ${TOTAL_LEN} tokens"
echo "  Sequence Packing: DISABLED (for fair comparison)"
echo ""
echo "Comparison targets:"
echo "  - vLLM Standalone: INPUT=${INPUT_LEN}, OUTPUT=${OUTPUT_LEN}"
echo "  - Megatron-Bridge: seq_length=${TOTAL_LEN}"
echo ""
echo "Models to run: ${PRESETS}"
echo ""
echo "Start time: $(date)"
echo ""

if [[ "$TOTAL_LEN" -ne 4096 ]]; then
    echo "⚠️  WARNING: Total length (${TOTAL_LEN}) != 4096"
    echo "    This may not match Megatron-Bridge default seq_length!"
    echo ""
fi

# ============================================================
# Run Benchmarks
# ============================================================
LAUNCHED=0
FAILED=0

for preset in $PRESETS; do
    echo "============================================================"
    echo "[$((LAUNCHED + FAILED + 1))] Launching: ${preset}"
    echo "    Config: Input=${INPUT_LEN}, Output=${OUTPUT_LEN}, Total=${TOTAL_LEN}"
    echo "============================================================"
    
    # Set variant for specific models (optional)
    VARIANT_ARG=""
    if [[ "$preset" == "qwen32b" ]]; then
        VARIANT_ARG="--variant gb200_tp2"
    fi
    
    CMD="python3 launch_grpo.py --preset ${preset} \
        ${VARIANT_ARG} \
        --use-random-dataset \
        --input-length ${INPUT_LEN} \
        --output-length ${OUTPUT_LEN} \
        --disable-sequence-packing \
        --wandb-project ${WANDB_PROJECT} \
        ${DRY_RUN}"
    
    echo "Command: $CMD"
    echo ""
    
    if eval $CMD; then
        echo "✅ ${preset}: Submitted successfully"
        LAUNCHED=$((LAUNCHED + 1))
    else
        echo "❌ ${preset}: Failed to submit"
        FAILED=$((FAILED + 1))
    fi
    
    echo ""
    sleep 2
done

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "Benchmark Summary"
echo "============================================================"
echo ""
echo "  Launched: ${LAUNCHED}"
echo "  Failed:   ${FAILED}"
echo ""
echo "End time: $(date)"
echo ""
echo "Check job status with: squeue -u \$USER"
echo ""
echo "Results will be tagged with: _I${INPUT_LEN}O${OUTPUT_LEN}_nopack"
echo ""
echo "Compare with:"
echo "  - vLLM:           cd vllm_benchmark && ./run_all_benchmarks.sh"
echo "  - Megatron-Bridge: cd /path/to/Megatron-Bridge && ./run_nemorl_reference_sweep.sh"
echo "============================================================"

