#!/bin/bash
# ============================================================
# vLLM Throughput Benchmark - All Models (GB200)
# ============================================================
# Configuration based on model_configs.yaml (GB200 settings)
#
# Models:
#   - Llama-3.1-8B-Instruct: 8 GPUs (2 nodes), Gen TP=1, PP=1
#   - Qwen3-32B: 16 GPUs (4 nodes), Gen TP=1, PP=1
#   - Qwen3-30B-A3B (MoE): 16 GPUs (4 nodes), Gen TP=1, PP=1
#
# Common settings:
#   - num_prompts: 64
#   - num_generations: 32 (n parameter)
#   - max_seqlen: 4096
#   - Input lengths: 64, 100, 150 tokens
#   - Output lengths: 2048, 4096 tokens
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# Common settings from model_configs.yaml
# ============================================================
NUM_PROMPTS=${NUM_PROMPTS:-64}
NUM_GENERATIONS=${NUM_GENERATIONS:-32}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
INPUT_LENS=${INPUT_LENS:-"64 100 150"}
OUTPUT_LENS=${OUTPUT_LENS:-"2048 4096"}

echo "============================================================"
echo "vLLM Throughput Benchmark - GB200 Models"
echo "============================================================"
echo "Reference: model_configs.yaml (GB200 settings)"
echo ""
echo "Common settings:"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  NUM_GENERATIONS (n): $NUM_GENERATIONS"
echo "  MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo "  INPUT_LENS: $INPUT_LENS"
echo "  OUTPUT_LENS: $OUTPUT_LENS"
echo ""
echo "Start time: $(date)"
echo ""

# ============================================================
# Llama-3.1-8B-Instruct (from model_configs.yaml llama8b.gb200)
# ============================================================
# num_gpus: 8 (2 nodes × 4 GPUs)
# generation: tp=1, pp=1
# ============================================================
echo "[1/3] Submitting Llama-3.1-8B-Instruct..."
echo "      Config: 8 GPUs (2 nodes), Gen TP=1, PP=1, DP=8"
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct \
NUM_NODES=2 \
TP_SIZE=1 \
PP_SIZE=1 \
THROUGHPUT_NUM_PROMPTS=$NUM_PROMPTS \
THROUGHPUT_NUM_GENERATIONS=$NUM_GENERATIONS \
MAX_MODEL_LEN=$MAX_MODEL_LEN \
INPUT_LENS="$INPUT_LENS" \
OUTPUT_LENS="$OUTPUT_LENS" \
./run_vllm_benchmark.sh run-throughput

echo ""

# ============================================================
# Qwen3-32B (from model_configs.yaml qwen32b.gb200)
# ============================================================
# num_gpus: 16 (4 nodes × 4 GPUs)
# generation: tp=1, pp=1
# ============================================================
echo "[2/3] Submitting Qwen3-32B..."
echo "      Config: 16 GPUs (4 nodes), Gen TP=1, PP=1, DP=16"
MODEL_PATH=Qwen/Qwen3-32B \
NUM_NODES=4 \
TP_SIZE=1 \
PP_SIZE=1 \
THROUGHPUT_NUM_PROMPTS=$NUM_PROMPTS \
THROUGHPUT_NUM_GENERATIONS=$NUM_GENERATIONS \
MAX_MODEL_LEN=$MAX_MODEL_LEN \
INPUT_LENS="$INPUT_LENS" \
OUTPUT_LENS="$OUTPUT_LENS" \
./run_vllm_benchmark.sh run-throughput

echo ""

# ============================================================
# Qwen3-30B-A3B MoE (from model_configs.yaml qwen30b.gb200)
# ============================================================
# num_gpus: 16 (4 nodes × 4 GPUs)
# generation: tp=1, pp=1
# Note: This is a MoE model, EP is for training only
# ============================================================
echo "[3/3] Submitting Qwen3-30B-A3B (MoE)..."
echo "      Config: 16 GPUs (4 nodes), Gen TP=1, PP=1, DP=16"
MODEL_PATH=Qwen/Qwen3-30B-A3B \
NUM_NODES=4 \
TP_SIZE=1 \
PP_SIZE=1 \
THROUGHPUT_NUM_PROMPTS=$NUM_PROMPTS \
THROUGHPUT_NUM_GENERATIONS=$NUM_GENERATIONS \
MAX_MODEL_LEN=$MAX_MODEL_LEN \
INPUT_LENS="$INPUT_LENS" \
OUTPUT_LENS="$OUTPUT_LENS" \
./run_vllm_benchmark.sh run-throughput

echo ""
echo "============================================================"
echo "All benchmarks submitted!"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Check job status with: squeue -u \$USER"
echo ""
echo "Results will be in:"
echo "  vllm_standalone_perf_exp/throughput/<job_id>-logs/"
