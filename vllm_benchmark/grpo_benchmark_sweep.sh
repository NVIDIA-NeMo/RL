#!/bin/bash
# ============================================================
# vLLM Throughput Benchmark - NeMo-RL GRPO Configuration
# ============================================================
# This script runs vllm benchmarks matching NeMo-RL GRPO
# rollout generation configurations.
#
# ISL:OSL ratio = 1:4 (reflecting OpenMathInstruct-2 characteristics)
# - Math problems have short input prompts
# - Solutions require long chain-of-thought reasoning
#
# Usage:
#   ./grpo_benchmark_sweep.sh                    # Run all
#   ./grpo_benchmark_sweep.sh --dry-run          # Preview
#   ./grpo_benchmark_sweep.sh --model qwen32b    # Specific model
#   ./grpo_benchmark_sweep.sh --list             # List configs
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# System Settings
# ============================================================
DRY_RUN=0
FILTER_MODEL=""
GPUS_PER_NODE=${GPUS_PER_NODE:-4}  # GB200: 4, H100: 8

# ============================================================
# GRPO Configuration
# ============================================================
# Format: NAME|HF_PATH|TP|PP|EP|NODES|ISL|OSL|NUM_PROMPTS|AVG_SEQLEN|DESCRIPTION
#
# ISL:OSL = 1:4 ratio (OpenMathInstruct-2 dataset characteristics)
# - Input: Math problem (short)
# - Output: Chain-of-thought reasoning + answer (long)
#
# ISL + OSL ≈ Avg SeqLen from H100 experiments

declare -a GRPO_CONFIGS=(
    # ┌─────────────────────────────────────────────────────────────────────────────┐
    # │ qwen32b | 16 GPUs | 4 Nodes | R-GBS=2048 | Gen(TP=1, PP=1, DP=16)          │
    # │ H100 avg seqlen: 3226 → ISL=640 + OSL=2560 = 3200 (1:4 ratio)              │
    # └─────────────────────────────────────────────────────────────────────────────┘
    "qwen32b|Qwen/Qwen3-32B|1|1|1|4|640|2560|2048|3226|Qwen3-32B GRPO (TP=1 PP=1 DP=16)"
    
    # ┌─────────────────────────────────────────────────────────────────────────────┐
    # │ qwen30b MoE | 16 GPUs | 4 Nodes | R-GBS=2048 | Gen(TP=1, PP=1, DP=16)      │
    # │ H100 avg seqlen: 3156 → ISL=640 + OSL=2560 = 3200 (1:4 ratio)              │
    # └─────────────────────────────────────────────────────────────────────────────┘
    "qwen30b|Qwen/Qwen3-30B-A3B|1|1|1|4|640|2560|2048|3156|Qwen3-30B MoE GRPO (TP=1 PP=1 DP=16)"
    
    # ┌─────────────────────────────────────────────────────────────────────────────┐
    # │ llama8b | 8 GPUs | 2 Nodes | R-GBS=2048 | Gen(TP=1, PP=1, DP=8)            │
    # │ H100 avg seqlen: 1056 → ISL=256 + OSL=1024 = 1280 (1:4 ratio)              │
    # └─────────────────────────────────────────────────────────────────────────────┘
    "llama8b|meta-llama/Llama-3.1-8B-Instruct|1|1|1|2|256|1024|2048|1056|LLaMA3-8B GRPO (TP=1 PP=1 DP=8)"
    
    # ┌─────────────────────────────────────────────────────────────────────────────┐
    # │ llama70b | 16 GPUs | 4 Nodes | R-GBS=2048 | Gen(TP=2, PP=1, DP=8)          │
    # │ H100 avg seqlen: 740 → ISL=128 + OSL=512 = 640 (1:4 ratio)                 │
    # └─────────────────────────────────────────────────────────────────────────────┘
    "llama70b|meta-llama/Llama-3.1-70B-Instruct|2|1|1|4|128|512|2048|740|LLaMA3-70B GRPO (TP=2 PP=1 DP=8)"
    
    # ┌─────────────────────────────────────────────────────────────────────────────┐
    # │ llama70b-lowgbs | 16 GPUs | 4 Nodes | R-GBS=512 | Gen(TP=2, PP=1, DP=8)    │
    # │ H100 avg seqlen: 740 → ISL=128 + OSL=512 = 640 (1:4 ratio)                 │
    # └─────────────────────────────────────────────────────────────────────────────┘
    "llama70b-lowgbs|meta-llama/Llama-3.1-70B-Instruct|2|1|1|4|128|512|512|740|LLaMA3-70B LowGBS GRPO (TP=2 PP=1 DP=8)"
    
    # ┌─────────────────────────────────────────────────────────────────────────────┐
    # │ llama70b-highseq | 16 GPUs | 4 Nodes | R-GBS=2048 | Gen(TP=2, PP=1, DP=8)  │
    # │ MaxSeqLen: 16384, H100 avg seqlen: 785 → ISL=256 + OSL=1024 = 1280         │
    # │ (higher ISL/OSL for long context scenario)                                  │
    # └─────────────────────────────────────────────────────────────────────────────┘
    "llama70b-highseq|meta-llama/Llama-3.1-70B-Instruct|2|1|1|4|256|1024|2048|785|LLaMA3-70B HighSeq GRPO (TP=2 PP=1 DP=8)"
)

# ============================================================
# Additional sweep configurations (for sensitivity analysis)
# ============================================================
declare -a SWEEP_CONFIGS=(
    # Different ISL/OSL combinations for sensitivity analysis (1:4 ratio)
    "qwen32b-short|Qwen/Qwen3-32B|1|1|1|4|256|1024|2048|1280|Qwen3-32B Short (ISL=256 OSL=1024)"
    "qwen32b-long|Qwen/Qwen3-32B|1|1|1|4|768|3072|2048|3840|Qwen3-32B Long (ISL=768 OSL=3072)"
    "llama8b-long|meta-llama/Llama-3.1-8B-Instruct|1|1|1|2|512|2048|2048|2560|LLaMA3-8B Long (ISL=512 OSL=2048)"
)

# ============================================================
# Parse Arguments
# ============================================================
INCLUDE_SWEEP=0

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
        --include-sweep)
            INCLUDE_SWEEP=1
            shift
            ;;
        --list|-l)
            echo "╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗"
            echo "║                   NeMo-RL GRPO Benchmark Configurations (ISL:OSL = 1:4)                                ║"
            echo "╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣"
            echo ""
            printf "  %-18s %-5s %-5s %-4s %-4s %-4s %-4s %-12s %-8s %-8s\n" "NAME" "Nodes" "GPUs" "TP" "PP" "EP" "DP" "ISL→OSL" "R-GBS" "AvgSeq"
            echo "  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────"
            
            for config in "${GRPO_CONFIGS[@]}"; do
                IFS='|' read -r name path tp pp ep nodes isl osl nprompts avgseq desc <<< "$config"
                total_gpus=$((nodes * GPUS_PER_NODE))
                gpus_per_inst=$((tp * pp))
                dp=$((total_gpus / gpus_per_inst))
                total_seq=$((isl + osl))
                printf "  %-18s %-5s %-5s %-4s %-4s %-4s %-4s %4s→%-4s  %-8s %-8s (sum=%d)\n" \
                    "$name" "$nodes" "$total_gpus" "$tp" "$pp" "$ep" "$dp" "$isl" "$osl" "$nprompts" "$avgseq" "$total_seq"
            done
            
            echo ""
            echo "  Additional sweep configs (use --include-sweep):"
            echo "  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────"
            for config in "${SWEEP_CONFIGS[@]}"; do
                IFS='|' read -r name path tp pp ep nodes isl osl nprompts avgseq desc <<< "$config"
                total_gpus=$((nodes * GPUS_PER_NODE))
                gpus_per_inst=$((tp * pp))
                dp=$((total_gpus / gpus_per_inst))
                total_seq=$((isl + osl))
                printf "  %-18s %-5s %-5s %-4s %-4s %-4s %-4s %4s→%-4s  %-8s %-8s (sum=%d)\n" \
                    "$name" "$nodes" "$total_gpus" "$tp" "$pp" "$ep" "$dp" "$isl" "$osl" "$nprompts" "$avgseq" "$total_seq"
            done
            echo ""
            echo "  Note: ISL:OSL = 1:4 ratio reflects OpenMathInstruct-2 dataset characteristics"
            echo "        (short math problems, long chain-of-thought solutions)"
            echo ""
            echo "╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝"
            exit 0
            ;;
        --help|-h)
            echo "NeMo-RL GRPO Benchmark Sweep"
            echo ""
            echo "This script runs vLLM benchmarks matching your NeMo-RL GRPO"
            echo "rollout generation configurations."
            echo ""
            echo "ISL:OSL ratio = 1:4 (OpenMathInstruct-2 dataset characteristics)"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run, -n       Show what would be run without submitting"
            echo "  --model, -m NAME    Run only specific model configuration"
            echo "  --include-sweep     Include additional ISL/OSL sweep configs"
            echo "  --list, -l          List all available configurations"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  GPUS_PER_NODE       GPUs per node (default: 4 for GB200, set to 8 for H100)"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run all GRPO configs"
            echo "  $0 --dry-run                 # Preview without submitting"
            echo "  $0 --model qwen32b           # Run only qwen32b"
            echo "  $0 --model llama             # Run all LLaMA variants"
            echo "  $0 --include-sweep           # Include sensitivity analysis configs"
            echo "  GPUS_PER_NODE=8 $0           # For H100 cluster (8 GPUs/node)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================
# Build config list
# ============================================================
declare -a ALL_CONFIGS=("${GRPO_CONFIGS[@]}")
if [ $INCLUDE_SWEEP -eq 1 ]; then
    ALL_CONFIGS+=("${SWEEP_CONFIGS[@]}")
fi

# ============================================================
# Print Header
# ============================================================
echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                   NeMo-RL GRPO vLLM Benchmark Sweep (ISL:OSL = 1:4)                                    ║"
echo "╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣"
echo "║  GPUs per node:  $GPUS_PER_NODE (GB200=4, H100=8)                                                           ║"
echo "║  ISL:OSL ratio:  1:4 (OpenMathInstruct-2 dataset)                                                      ║"
if [ -n "$FILTER_MODEL" ]; then
echo "║  Filter:         $FILTER_MODEL                                                                             ║"
fi
if [ $DRY_RUN -eq 1 ]; then
echo "║  Mode:           DRY RUN                                                                               ║"
fi
echo "╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================
# Submit Jobs
# ============================================================
submitted=0
skipped=0

for config in "${ALL_CONFIGS[@]}"; do
    IFS='|' read -r name path tp pp ep nodes isl osl nprompts avgseq desc <<< "$config"
    
    # Filter by model name if specified
    if [ -n "$FILTER_MODEL" ] && [[ ! "$name" == *"$FILTER_MODEL"* ]]; then
        continue
    fi
    
    # Calculate DP
    total_gpus=$((nodes * GPUS_PER_NODE))
    gpus_per_inst=$((tp * pp))
    dp=$((total_gpus / gpus_per_inst))
    total_seq=$((isl + osl))
    
    echo "┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐"
    echo "│ $name"
    echo "├─────────────────────────────────────────────────────────────────────────────────────────────────────┤"
    echo "│  Model:       $path"
    echo "│  Resources:   ${nodes} nodes × ${GPUS_PER_NODE} GPUs = ${total_gpus} GPUs total"
    echo "│  Parallelism: TP=$tp, PP=$pp, EP=$ep, DP=$dp"
    echo "│  Sequences:   ISL=$isl → OSL=$osl (1:4 ratio, sum=$total_seq, target avg=$avgseq)"
    echo "│  Batch size:  $nprompts prompts (Rollout GBS)"
    echo "└─────────────────────────────────────────────────────────────────────────────────────────────────────┘"
    
    if [ $DRY_RUN -eq 1 ]; then
        echo "  ⏸️  [DRY RUN] Would submit job"
        skipped=$((skipped + 1))
    else
        echo "  🚀 Submitting..."
        
        # Submit the job with per-config ISL/OSL and batch size
        MODEL_PATH="$path" \
        TP_SIZE=$tp \
        PP_SIZE=$pp \
        EP_SIZE=$ep \
        NUM_NODES=$nodes \
        INPUT_LENS="$isl" \
        OUTPUT_LENS="$osl" \
        THROUGHPUT_NUM_PROMPTS=$nprompts \
        ./run_vllm_benchmark.sh run-throughput 2>&1 | grep -E "(Submitted|Logs)" | sed 's/^/  /' || true
        
        submitted=$((submitted + 1))
        
        # Small delay
        sleep 2
    fi
    echo ""
done

# ============================================================
# Summary
# ============================================================
echo "╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗"
if [ $DRY_RUN -eq 1 ]; then
echo "║  DRY RUN complete. Would have submitted $skipped job(s).                                                ║"
else
echo "║  Submitted $submitted job(s).                                                                             ║"
echo "╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣"
echo "║  Monitor: squeue -u \$USER                                                                              ║"
echo "║  Results: python collect_results.py --throughput                                                       ║"
fi
echo "╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝"
