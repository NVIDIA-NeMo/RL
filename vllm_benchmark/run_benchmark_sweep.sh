#!/bin/bash
# Run vLLM Benchmark Sweep
# 
# This script runs multiple benchmark configurations
# Results are saved to $SLURM_JOB_ID-logs/results.json for each job

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Define your experiment configurations here
# ============================================================

# Models to test
MODELS=(
    "Qwen/Qwen2.5-32B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
)

# Parallelism configurations: "NUM_NODES:TP_SIZE:PP_SIZE:EP_SIZE"
# EP_SIZE=1 for non-MoE models, EP_SIZE>1 for MoE models
CONFIGS=(
    "1:4:1:1"   # 1 node, TP=4, PP=1, EP=1, DP=1
    "2:4:1:1"   # 2 nodes, TP=4, PP=1, EP=1, DP=2
    "4:4:1:1"   # 4 nodes, TP=4, PP=1, EP=1, DP=4
    "4:1:1:1"
    "4:2:2:1"
    # MoE examples:
    # "4:4:1:4"   # 4 nodes, TP=4, PP=1, EP=4 (for MoE models)
    # "2:8:1:1"   # 2 nodes, TP=8, PP=1, DP=1
)

# Generation settings
NUM_PROMPTS=${NUM_PROMPTS:-64}
NUM_GENERATIONS=${NUM_GENERATIONS:-32}
MAX_TOKENS=${MAX_TOKENS:-2048}

# ============================================================
# Run experiments
# ============================================================

echo "============================================================"
echo "vLLM Benchmark Sweep"
echo "============================================================"
echo "Models: ${MODELS[*]}"
echo "Configs: ${CONFIGS[*]}"
echo "Prompts: $NUM_PROMPTS × $NUM_GENERATIONS"
echo "============================================================"

JOB_IDS=()

for MODEL in "${MODELS[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do
        # Parse config: NODES:TP:PP:EP
        IFS=':' read -r NODES TP PP EP <<< "$CONFIG"
        EP=${EP:-1}  # Default EP=1 if not specified
        
        echo ""
        if [ "$EP" -gt 1 ]; then
            echo "Submitting: MODEL=$MODEL, NODES=$NODES, TP=$TP, PP=$PP, EP=$EP (MoE)"
        else
            echo "Submitting: MODEL=$MODEL, NODES=$NODES, TP=$TP, PP=$PP"
        fi
        
        # Submit job and capture job ID
        JOB_OUTPUT=$(NUM_NODES=$NODES \
            TP_SIZE=$TP \
            PP_SIZE=$PP \
            EP_SIZE=$EP \
            MODEL_PATH="$MODEL" \
            NUM_PROMPTS=$NUM_PROMPTS \
            NUM_GENERATIONS=$NUM_GENERATIONS \
            MAX_TOKENS=$MAX_TOKENS \
            "$SCRIPT_DIR/run_vllm_benchmark.sh" run-random 2>&1)
        
        # Extract job ID from output
        JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'Submitted batch job \K\d+' || echo "unknown")
        JOB_IDS+=("$JOB_ID")
        
        echo "  → Job ID: $JOB_ID"
        
        # Small delay between submissions
        sleep 2
    done
done

echo ""
echo "============================================================"
echo "All jobs submitted!"
echo "============================================================"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor with: squeue -u $USER"
echo ""
echo "Results will be in: vllm_standalone_perf_exp/"
for JOB_ID in "${JOB_IDS[@]}"; do
    echo "  - vllm_standalone_perf_exp/${JOB_ID}-logs/results.json"
done
echo ""
echo "Collect results with: python collect_results.py"

