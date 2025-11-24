#!/bin/bash
#
# Checkpoint Sweep Evaluation Script
# 
# Runs LLaDA evaluation pipeline across multiple DCP checkpoints.
# For each step_* directory, launches a dedicated inference+eval job with:
# - Isolated eval job directory with timestamped results
# - Server info file stored within the eval directory
# - Eval results published to the eval directory
# - "latest" symlink updated to point to most recent completed eval
#
# Usage:
#   bash run_checkpoint_sweep_eval.sh <checkpoints_parent_dir> [options]
#
# Example:
#   bash run_checkpoint_sweep_eval.sh /path/to/checkpoints --steps "512,1024" --benchmark gsm8k:4
#
# ---------------------------------------------------------------------------

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

show_usage() {
    cat <<USAGE
Usage: $0 <checkpoints_parent_dir> [OPTIONS]

Run evaluation pipeline across multiple DCP checkpoints.

Arguments:
  checkpoints_parent_dir    Parent directory containing step_* checkpoint directories
                           Each step_* directory should contain a policy/ subdirectory

Options:
  --steps STEPS            Comma-separated list of step numbers to evaluate (default: all)
                           Example: --steps "1100,1200,1300"
  --pattern PATTERN        Custom pattern for checkpoint directories (default: "step_*")
  --benchmark BENCH        Override benchmark (default: from preset)
  --generation-algorithm   Override generation algorithm (default: from preset)
  --dry-run               Show what would be evaluated without running
  --help                  Show this help message

Examples:
  # Evaluate all checkpoints
  $0 /lustre/fsw/portfolios/llmservice/users/myuser/training/run1

  # Evaluate specific steps
  $0 /lustre/fsw/portfolios/llmservice/users/myuser/training/run1 --steps "1100,1200"

  # Dry run to see what would be evaluated
  $0 /lustre/fsw/portfolios/llmservice/users/myuser/training/run1 --dry-run

USAGE
}

# Parse arguments
if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_usage
    exit 0
fi

CHECKPOINTS_DIR="$1"
shift

# Default configuration
CHECKPOINT_PATTERN="step_*"
SPECIFIC_STEPS=""
BENCHMARK_OVERRIDE=""
GEN_ALGO_OVERRIDE=""
DRY_RUN=false

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)
            SPECIFIC_STEPS="$2"
            shift 2
            ;;
        --pattern)
            CHECKPOINT_PATTERN="$2"
            shift 2
            ;;
        --benchmark)
            BENCHMARK_OVERRIDE="$2"
            shift 2
            ;;
        --generation-algorithm)
            GEN_ALGO_OVERRIDE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift 1
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate checkpoints directory
if [[ ! -d "$CHECKPOINTS_DIR" ]]; then
    print_error "Checkpoints directory not found: $CHECKPOINTS_DIR"
    exit 1
fi

CHECKPOINTS_DIR="$(realpath "$CHECKPOINTS_DIR")"
print_status "Checkpoints directory: $CHECKPOINTS_DIR"

# Locate the pipeline preset script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_PRESET="$SCRIPT_DIR/run_llada_eval_pipeline_preset.sh"

if [[ ! -x "$PIPELINE_PRESET" ]]; then
    print_error "Pipeline preset script not found or not executable: $PIPELINE_PRESET"
    exit 1
fi

print_status "Pipeline preset: $PIPELINE_PRESET"

# Find checkpoint directories
print_status "Searching for checkpoint directories matching pattern: $CHECKPOINT_PATTERN"
CHECKPOINT_DIRS=()
while IFS= read -r -d '' dir; do
    CHECKPOINT_DIRS+=("$dir")
done < <(find "$CHECKPOINTS_DIR" -maxdepth 1 -type d -name "$CHECKPOINT_PATTERN" -print0 | sort -zV)

if [[ ${#CHECKPOINT_DIRS[@]} -eq 0 ]]; then
    print_error "No checkpoint directories found matching pattern: $CHECKPOINT_PATTERN"
    exit 1
fi

print_status "Found ${#CHECKPOINT_DIRS[@]} checkpoint directories"

# Filter by specific steps if requested
if [[ -n "$SPECIFIC_STEPS" ]]; then
    print_status "Filtering for specific steps: $SPECIFIC_STEPS"
    IFS=',' read -ra STEP_NUMS <<< "$SPECIFIC_STEPS"
    FILTERED_DIRS=()
    for dir in "${CHECKPOINT_DIRS[@]}"; do
        basename_dir="$(basename "$dir")"
        step_num="${basename_dir#step_}"
        for requested_step in "${STEP_NUMS[@]}"; do
            requested_step=$(echo "$requested_step" | xargs) # trim whitespace
            if [[ "$step_num" == "$requested_step" ]]; then
                FILTERED_DIRS+=("$dir")
                break
            fi
        done
    done
    CHECKPOINT_DIRS=("${FILTERED_DIRS[@]}")
    
    if [[ ${#CHECKPOINT_DIRS[@]} -eq 0 ]]; then
        print_error "No checkpoint directories found matching requested steps: $SPECIFIC_STEPS"
        exit 1
    fi
    print_status "Filtered to ${#CHECKPOINT_DIRS[@]} checkpoint directories"
fi

# Display checkpoint directories to be evaluated
print_header "Checkpoints to Evaluate"
for dir in "${CHECKPOINT_DIRS[@]}"; do
    checkpoint_name="$(basename "$dir")"
    policy_dir="$dir/policy"
    if [[ -d "$policy_dir" ]]; then
        echo "  ✓ $checkpoint_name (policy dir exists)"
    else
        echo "  ✗ $checkpoint_name (WARNING: no policy dir)"
    fi
done
echo ""

if [[ "$DRY_RUN" == true ]]; then
    print_warning "DRY RUN MODE - No jobs will be launched"
    exit 0
fi

# Confirm before proceeding
#read -p "Proceed with evaluation? [y/N] " -n 1 -r
#echo
#if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#    print_warning "Cancelled by user"
#    exit 0
#fi

# Process each checkpoint
TOTAL=${#CHECKPOINT_DIRS[@]}
CURRENT=0
FAILED_CHECKPOINTS=()

for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
    CURRENT=$((CURRENT + 1))
    checkpoint_name="$(basename "$checkpoint_dir")"
    
    print_header "[$CURRENT/$TOTAL] Evaluating: $checkpoint_name"
    
    # Validate policy directory exists
    policy_dir="$checkpoint_dir/policy"
    if [[ ! -d "$policy_dir" ]]; then
        print_error "Policy directory not found: $policy_dir"
        print_warning "Skipping $checkpoint_name"
        FAILED_CHECKPOINTS+=("$checkpoint_name (no policy dir)")
        continue
    fi
    
    # Create timestamped eval job directory
    timestamp="$(date +%Y%m%d_%H%M%S)"
    # Extract hyperparameters from preset (generation algorithm, benchmark, etc.)
    # For naming, we'll use a simplified approach
    eval_job_name="eval_${timestamp}"
    #eval_job_name="eval_ckpt_sweep"
    eval_job_dir="$checkpoint_dir/$eval_job_name"
    
    print_status "Creating eval job directory: $eval_job_dir"
    mkdir -p "$eval_job_dir"
    
    # Define paths for this eval job
    server_info_file="$eval_job_dir/server_info.env"
    eval_output_dir="$eval_job_dir/results"
    eval_log_file="$eval_job_dir/pipeline.log"
    
    print_status "Server info file: $server_info_file"
    print_status "Eval output dir: $eval_output_dir"
    print_status "Pipeline log: $eval_log_file"
    
    # Export environment variables for the pipeline
    export SERVER_DCP_PATH="$policy_dir"
    export SERVER_INFO_FILE="$server_info_file"
    export EVAL_OUTPUT_DIR="$eval_output_dir"
    export SEQ_EVAL_OUTPUT_DIR="$eval_output_dir"
    export SEQ_EVAL_EXPNAME="${checkpoint_name}"
    
    # Override benchmark/algorithm if requested
    if [[ -n "$BENCHMARK_OVERRIDE" ]]; then
        export SEQ_EVAL_BENCHMARK="$BENCHMARK_OVERRIDE"
        print_status "Override benchmark: $BENCHMARK_OVERRIDE"
    fi
    if [[ -n "$GEN_ALGO_OVERRIDE" ]]; then
        export SEQ_EVAL_GENERATION_ALGORITHM="$GEN_ALGO_OVERRIDE"
        print_status "Override generation algorithm: $GEN_ALGO_OVERRIDE"
    fi
    
    # Write metadata file
    cat > "$eval_job_dir/metadata.txt" <<METADATA
Evaluation Job Metadata
========================
Checkpoint: $checkpoint_name
Policy Path: $policy_dir
Timestamp: $timestamp
Job Directory: $eval_job_dir
Server Info File: $server_info_file
Output Directory: $eval_output_dir

Server Configuration:
---------------------
  Partition: ${SERVER_PARTITION:-unset}
  GPUs: ${SERVER_GPUS:-unset}
  Batch Size: ${SERVER_BATCH_SIZE:-unset}
  Base Model: ${SERVER_BASE_MODEL:-unset}
  DCP Path: ${SERVER_DCP_PATH:-unset}
  Engine: ${SERVER_ENGINE:-unset}
  Extra Args: ${SERVER_EXTRA_ARGS:-unset}

Evaluation Configuration:
-------------------------
  Benchmark: ${SEQ_EVAL_BENCHMARK:-unset}
  Experiment Name: ${SEQ_EVAL_EXPNAME:-unset}
  Generation Algorithm: ${SEQ_EVAL_GENERATION_ALGORITHM:-unset}
  Steps: ${SEQ_EVAL_STEPS:-unset}
  Block Length: ${SEQ_EVAL_BLOCK_LENGTH:-unset}
  Threshold: ${SEQ_EVAL_THRESHOLD:-unset}
  Tokens to Generate: ${SEQ_EVAL_TOKENS_TO_GENERATE:-unset}
  Partition: ${SEQ_EVAL_PARTITION:-unset}
  Use Same Node: ${SEQ_EVAL_USE_SAME_NODE:-unset}
  Extra Args: ${SEQ_EVAL_EXTRA_ARGS:-unset}

All Environment Variables:
--------------------------
$(env | grep -E '^(SERVER_|SEQ_EVAL_|EVAL_|ACCOUNT)' | sort)
METADATA
    
    print_status "Running pipeline..."
    echo ""
    
    # Run the pipeline and capture output
    if "$PIPELINE_PRESET" 2>&1 | tee "$eval_log_file"; then
        print_status "Pipeline completed successfully for $checkpoint_name"
        
        # Update "latest" symlink
        latest_link="$checkpoint_dir/latest_eval"
        if [[ -L "$latest_link" ]] || [[ -e "$latest_link" ]]; then
            rm -f "$latest_link"
        fi
        #ln -s "$eval_job_name" "$latest_link"
        #print_status "Updated 'latest_eval' symlink -> $eval_job_name"
        
        # Mark as completed
        touch "$eval_job_dir/COMPLETED"
        echo "$(date -Iseconds)" > "$eval_job_dir/COMPLETED"
        
    else
        exit_code=$?
        print_error "Pipeline failed for $checkpoint_name (exit code: $exit_code)"
        FAILED_CHECKPOINTS+=("$checkpoint_name (exit code: $exit_code)")
        
        # Mark as failed
        touch "$eval_job_dir/FAILED"
        echo "Exit code: $exit_code" > "$eval_job_dir/FAILED"
        echo "$(date -Iseconds)" >> "$eval_job_dir/FAILED"
    fi
    
    echo ""
    echo ""
done

# Summary
print_header "Evaluation Sweep Summary"
echo "Total checkpoints: $TOTAL"
echo "Successful: $((TOTAL - ${#FAILED_CHECKPOINTS[@]}))"
echo "Failed: ${#FAILED_CHECKPOINTS[@]}"

if [[ ${#FAILED_CHECKPOINTS[@]} -gt 0 ]]; then
    echo ""
    print_warning "Failed checkpoints:"
    for failed in "${FAILED_CHECKPOINTS[@]}"; do
        echo "  - $failed"
    done
fi

echo ""
print_status "Checkpoint sweep evaluation complete!"
echo ""
echo "Results are organized within each checkpoint directory:"
echo "  - Each eval run: step_*/eval_YYYYMMDD_HHMMSS/"
echo "  - Latest eval: step_*/latest_eval -> eval_YYYYMMDD_HHMMSS/"
echo ""

