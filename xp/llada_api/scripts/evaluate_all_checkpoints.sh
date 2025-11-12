#!/bin/bash

# Evaluate All Checkpoints Script
# Iterates over all step_* directories in a results directory
# and evaluates each checkpoint on a specified benchmark

set -e

# Default values
RESULTS_DIR=""
BENCHMARK="gsm8k:1"
OUTPUT_BASE_DIR=""
BASE_MODEL="nvidia/Nemotron-Diffusion-Research-4B-v0"
MODEL_NAME="nemotron-4b"
GENERATION_ALGORITHM="nemotron"
THRESHOLD=0.9
TOKENS_TO_GENERATE=512
STEPS=512
BLOCK_LENGTH=32
BATCH_SIZE=1
SERVER_GPUS=8
VERBOSE=false
PORT_BASE=8000

# SLURM options
USE_SLURM=false
SERVER_PARTITION="interactive"
EVAL_PARTITION="interactive"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/mfathi/containers/nemo_rl_base.sqsh"

# Checkpoint selection
STEP_PATTERN="step_*"  # Pattern to match checkpoint directories
SKIP_EXISTING=false    # Skip checkpoints that already have results

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

show_usage() {
    cat <<EOF
Usage: $0 RESULTS_DIR BENCHMARK [OPTIONS]

Evaluates all step_* checkpoints in a directory on a given benchmark.

Positional Arguments:
  RESULTS_DIR                 Directory containing step_* subdirectories
  BENCHMARK                   Benchmark to evaluate (e.g., gsm8k:1, math:2)

Model Options:
  --base-model MODEL          Base model for DCP (default: $BASE_MODEL)
  --model-name NAME           Model name for eval (default: $MODEL_NAME)
  --output-base-dir DIR       Base output directory (default: RESULTS_DIR/eval_results)
  
Inference Options:
  --generation-algorithm ALGO Algorithm (default: $GENERATION_ALGORITHM)
  --threshold VAL             Threshold (default: $THRESHOLD)
  --tokens-to-generate NUM    Max tokens (default: $TOKENS_TO_GENERATE)
  --steps NUM                 Diffusion steps (default: $STEPS)
  --block-length NUM          Block length (default: $BLOCK_LENGTH)
  
Server Options:
  --batch-size SIZE           Server batch size (default: $BATCH_SIZE)
  --server-gpus NUM           Number of GPUs (default: $SERVER_GPUS)
  --port-base PORT            Base port number (increments for each eval) (default: $PORT_BASE)
  --verbose                   Enable verbose logging
  
Checkpoint Selection:
  --step-pattern PATTERN      Pattern to match checkpoints (default: "$STEP_PATTERN")
  --skip-existing             Skip checkpoints with existing results
  
Execution Mode:
  --slurm                     Run on SLURM (default: local/interactive)
  --server-partition PART     SLURM partition for server (default: $SERVER_PARTITION)
  --eval-partition PART       SLURM partition for eval (default: $EVAL_PARTITION)
  --container IMAGE           Container image path

Examples:
  # Evaluate all checkpoints on GSM8K
  $0 /path/to/results gsm8k:1
  
  # Evaluate on MATH with custom output directory
  $0 /path/to/results math:2 --output-base-dir /path/to/eval_results
  
  # SLURM mode with custom parameters
  export ACCOUNT=your_account
  $0 /path/to/results gsm8k:1 --slurm --steps 256 --server-gpus 4
  
  # Skip already evaluated checkpoints
  $0 /path/to/results gsm8k:1 --skip-existing
  
  # Evaluate specific pattern
  $0 /path/to/results gsm8k:1 --step-pattern "step_1[0-9][0-9][0-9]"

EOF
}

# Parse positional arguments
if [[ $# -lt 2 ]]; then
    show_usage
    exit 1
fi

RESULTS_DIR="$1"
BENCHMARK="$2"
shift 2

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) show_usage; exit 0 ;;
        --base-model) BASE_MODEL="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --output-base-dir) OUTPUT_BASE_DIR="$2"; shift 2 ;;
        --generation-algorithm) GENERATION_ALGORITHM="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --tokens-to-generate) TOKENS_TO_GENERATE="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --block-length) BLOCK_LENGTH="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --server-gpus) SERVER_GPUS="$2"; shift 2 ;;
        --port-base) PORT_BASE="$2"; shift 2 ;;
        --verbose) VERBOSE=true; shift 1 ;;
        --step-pattern) STEP_PATTERN="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=true; shift 1 ;;
        --slurm) USE_SLURM=true; shift 1 ;;
        --server-partition) SERVER_PARTITION="$2"; shift 2 ;;
        --eval-partition) EVAL_PARTITION="$2"; shift 2 ;;
        --container) CONTAINER_IMAGE="$2"; shift 2 ;;
        *) print_error "Unknown option: $1"; show_usage; exit 1 ;;
    esac
done

# Validate results directory
if [[ ! -d "$RESULTS_DIR" ]]; then
    print_error "Results directory does not exist: $RESULTS_DIR"
    exit 1
fi

# Set default output directory
if [[ -z "$OUTPUT_BASE_DIR" ]]; then
    OUTPUT_BASE_DIR="$RESULTS_DIR/eval_results_$(echo $BENCHMARK | tr ':' '_')"
fi

# Validate SLURM requirements
if [[ "$USE_SLURM" == true ]] && [[ -z "$ACCOUNT" ]]; then
    print_error "ACCOUNT environment variable must be set for SLURM mode"
    exit 1
fi

# Get script paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_EVAL_SCRIPT="$SCRIPT_DIR/evaluate_single_checkpoint.sh"

# Verify single eval script exists
if [[ ! -f "$SINGLE_EVAL_SCRIPT" ]]; then
    print_error "Single evaluation script not found: $SINGLE_EVAL_SCRIPT"
    exit 1
fi

# Find all step directories
print_status "Scanning for checkpoints in: $RESULTS_DIR"
print_status "Pattern: $STEP_PATTERN"

# Use array to store checkpoint paths
CHECKPOINTS=()
while IFS= read -r -d '' checkpoint; do
    CHECKPOINTS+=("$checkpoint")
done < <(find "$RESULTS_DIR" -maxdepth 1 -type d -name "$STEP_PATTERN" -print0 | sort -z)

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    print_error "No checkpoints found matching pattern: $STEP_PATTERN"
    print_error "In directory: $RESULTS_DIR"
    exit 1
fi

# Filter checkpoints if skip-existing is enabled
if [[ "$SKIP_EXISTING" == true ]]; then
    FILTERED_CHECKPOINTS=()
    for checkpoint in "${CHECKPOINTS[@]}"; do
        step_name=$(basename "$checkpoint")
        output_dir="$OUTPUT_BASE_DIR/$step_name"
        
        # Check if results already exist
        if [[ -d "$output_dir" ]] && [[ -n "$(ls -A "$output_dir" 2>/dev/null)" ]]; then
            print_warning "Skipping $step_name (results already exist in $output_dir)"
        else
            FILTERED_CHECKPOINTS+=("$checkpoint")
        fi
    done
    CHECKPOINTS=("${FILTERED_CHECKPOINTS[@]}")
    
    if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
        print_status "All checkpoints have been evaluated already (use without --skip-existing to re-run)"
        exit 0
    fi
fi

echo "=============================================================="
print_status "Evaluate All Checkpoints"
echo "=============================================================="
echo "Results Directory: $RESULTS_DIR"
echo "Output Base: $OUTPUT_BASE_DIR"
echo "Benchmark: $BENCHMARK"
echo "Found Checkpoints: ${#CHECKPOINTS[@]}"
echo "Mode: $([ "$USE_SLURM" == true ] && echo "SLURM" || echo "Local/Interactive")"
echo "=============================================================="
echo ""

# List all checkpoints to be evaluated
print_status "Checkpoints to evaluate:"
for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "  - $(basename "$checkpoint")"
done
echo ""

# Create summary log
SUMMARY_LOG="$OUTPUT_BASE_DIR/evaluation_summary_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUTPUT_BASE_DIR"

{
    echo "=============================================================="
    echo "Evaluation Summary"
    echo "=============================================================="
    echo "Results Directory: $RESULTS_DIR"
    echo "Benchmark: $BENCHMARK"
    echo "Total Checkpoints: ${#CHECKPOINTS[@]}"
    echo "Started: $(date)"
    echo "=============================================================="
    echo ""
} > "$SUMMARY_LOG"

print_status "Summary log: $SUMMARY_LOG"
echo ""

# Evaluate each checkpoint
SUCCESSFUL=0
FAILED=0
PORT=$PORT_BASE

for i in "${!CHECKPOINTS[@]}"; do
    checkpoint="${CHECKPOINTS[$i]}"
    step_name=$(basename "$checkpoint")
    checkpoint_num=$((i + 1))
    
    # Check for policy subdirectory (common pattern)
    DCP_PATH="$checkpoint"
    if [[ -d "$checkpoint/policy" ]]; then
        DCP_PATH="$checkpoint/policy"
        print_status "Using policy subdirectory: $DCP_PATH"
    fi
    
    output_dir="$OUTPUT_BASE_DIR/$step_name"
    
    echo ""
    echo "=============================================================="
    print_step "Checkpoint $checkpoint_num/${#CHECKPOINTS[@]}: $step_name"
    echo "=============================================================="
    print_status "DCP Path: $DCP_PATH"
    print_status "Output: $output_dir"
    print_status "Port: $PORT"
    echo ""
    
    # Build evaluation command
    EVAL_ARGS="--dcp-path '$DCP_PATH' --base-model '$BASE_MODEL'"
    EVAL_ARGS="$EVAL_ARGS --output-dir '$output_dir' --benchmark '$BENCHMARK'"
    EVAL_ARGS="$EVAL_ARGS --model-name '$MODEL_NAME' --generation-algorithm '$GENERATION_ALGORITHM'"
    EVAL_ARGS="$EVAL_ARGS --threshold $THRESHOLD --tokens-to-generate $TOKENS_TO_GENERATE"
    EVAL_ARGS="$EVAL_ARGS --steps $STEPS --block-length $BLOCK_LENGTH"
    EVAL_ARGS="$EVAL_ARGS --batch-size $BATCH_SIZE --server-gpus $SERVER_GPUS --port $PORT"
    
    if [[ "$USE_SLURM" == true ]]; then
        EVAL_ARGS="$EVAL_ARGS --slurm --server-partition '$SERVER_PARTITION' --eval-partition '$EVAL_PARTITION'"
        EVAL_ARGS="$EVAL_ARGS --container '$CONTAINER_IMAGE'"
    fi
    
    if [[ "$VERBOSE" == true ]]; then
        EVAL_ARGS="$EVAL_ARGS --verbose"
    fi
    
    # Run evaluation
    START_TIME=$(date +%s)
    
    {
        echo "=============================================================="
        echo "Checkpoint: $step_name"
        echo "Started: $(date)"
        echo "=============================================================="
    } >> "$SUMMARY_LOG"
    
    if eval "$SINGLE_EVAL_SCRIPT $EVAL_ARGS"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        SUCCESSFUL=$((SUCCESSFUL + 1))
        
        print_status "✓ Successfully evaluated $step_name (${DURATION}s)"
        
        {
            echo "Status: SUCCESS"
            echo "Duration: ${DURATION}s"
            echo "Output: $output_dir"
            echo ""
        } >> "$SUMMARY_LOG"
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        FAILED=$((FAILED + 1))
        
        print_error "✗ Failed to evaluate $step_name (${DURATION}s)"
        
        {
            echo "Status: FAILED"
            echo "Duration: ${DURATION}s"
            echo ""
        } >> "$SUMMARY_LOG"
    fi
    
    # Increment port for next checkpoint (avoid conflicts)
    PORT=$((PORT + 1))
    
    # Add a small delay between checkpoints
    if [[ $checkpoint_num -lt ${#CHECKPOINTS[@]} ]]; then
        print_status "Waiting 10 seconds before next checkpoint..."
        sleep 10
    fi
done

# Final summary
echo ""
echo "=============================================================="
print_status "Evaluation Complete!"
echo "=============================================================="
print_status "Total Checkpoints: ${#CHECKPOINTS[@]}"
print_status "Successful: $SUCCESSFUL"
if [[ $FAILED -gt 0 ]]; then
    print_error "Failed: $FAILED"
fi
echo "=============================================================="
echo ""
print_status "Results saved to: $OUTPUT_BASE_DIR"
print_status "Summary log: $SUMMARY_LOG"
echo ""

{
    echo "=============================================================="
    echo "Final Summary"
    echo "=============================================================="
    echo "Completed: $(date)"
    echo "Total Checkpoints: ${#CHECKPOINTS[@]}"
    echo "Successful: $SUCCESSFUL"
    echo "Failed: $FAILED"
    echo "=============================================================="
} >> "$SUMMARY_LOG"

# Exit with error if any failed
if [[ $FAILED -gt 0 ]]; then
    exit 1
fi

exit 0

