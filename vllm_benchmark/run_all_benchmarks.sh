#!/bin/bash
# ============================================================
# vLLM Throughput Benchmark - All Models (GB200)
# ============================================================
# Configuration based on model_configs.yaml (GB200 settings)
#
# NOTE: vllm bench throughput does NOT support Data Parallelism (DP)!
#       Only TP×PP GPUs are used per benchmark.
#       NUM_NODES is auto-adjusted to avoid GPU waste.
#
# Cluster support:
#   - Auto-detects cluster (lyris vs oci-hsg) from hostname
#   - Override with: CLUSTER=lyris ./run_all_benchmarks.sh
#   - Lyris:   partition=gb200, account=coreai_dlalgo_llm, no gres
#   - OCI-HSG: partition=batch, account=coreai_dlalgo_nemorl, gres=gpu:N
#
# Models (using TP×PP GPUs each):
#   - Llama-3.1-8B-Instruct: TP=1, PP=1 → 1 GPU
#   - Qwen3-32B: TP=1, PP=1 → 1 GPU
#   - Qwen3-30B-A3B (MoE): TP=1, PP=1 → 1 GPU
#   - Llama-3.1-70B-Instruct: TP=2, PP=1 → 2 GPUs
#
# Common settings:
#   - num_prompts: 64
#   - num_generations: 32 (n parameter)
#   - max_seqlen: 4096
#
# Input/Output Length Configurations:
#   All configurations maintain Total = Input + Output = 4096
#   to match Megatron-Bridge training seq_length=4096
#
#   Default pairs (Input:Output):
#     - 512:3584   (short input, long output - typical LLM generation)
#     - 1024:3072  (balanced towards output)
#     - 2048:2048  (equal split - fairest comparison)
#     - 3072:1024  (long input, short output - summarization)
#
# wandb settings:
#   WANDB_PROJECT=my-project ./run_all_benchmarks.sh
#   WANDB_UPLOAD=0 ./run_all_benchmarks.sh  # disable upload
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

# ============================================================
# Input/Output Length Pairs (Total = 4096 to match Megatron-Bridge)
# ============================================================
# Format: "input1:output1 input2:output2 ..."
# Each pair sums to 4096 for fair comparison with:
#   - Megatron-Bridge: seq_length=4096 (training)
#   - NeMo-RL GRPO: input_length + output_length = 4096
#
# OpenMathInstruct-2 dataset characteristics (from HuggingFace):
#   - problem: 13~13.6K chars, typically 50-150 tokens
#   - generated_solution: 190~50.7K chars, typically 200-500+ tokens (CoT)
#
# Reference: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2
#
# Default configuration (optimized for OpenMathInstruct-2):
#   128:3968   - RECOMMENDED (realistic math problem length)
#
# Alternative configurations:
#   64:4032    - Very short problems (simple arithmetic)
#   128:3968   - Typical math problems (RECOMMENDED)
#   256:3840   - Longer problems (word problems, complex equations)
#   512:3584   - Complex multi-step problems
#
# Override with: IO_PAIRS="128:3968 256:3840" ./run_all_benchmarks.sh
# ============================================================
IO_PAIRS=${IO_PAIRS:-"128:3968"}

# Legacy variables (kept for backward compatibility, but IO_PAIRS takes precedence)
INPUT_LENS=${INPUT_LENS:-""}
OUTPUT_LENS=${OUTPUT_LENS:-""}

# Parse IO_PAIRS into INPUT_LENS and OUTPUT_LENS if not explicitly set
if [[ -z "$INPUT_LENS" ]] && [[ -n "$IO_PAIRS" ]]; then
    INPUT_LENS=""
    OUTPUT_LENS=""
    for pair in $IO_PAIRS; do
        input_len="${pair%%:*}"
        output_len="${pair##*:}"
        INPUT_LENS="$INPUT_LENS $input_len"
        OUTPUT_LENS="$OUTPUT_LENS $output_len"
    done
    INPUT_LENS="${INPUT_LENS# }"  # trim leading space
    OUTPUT_LENS="${OUTPUT_LENS# }"
fi

# ============================================================
# Weights & Biases settings
# ============================================================
export WANDB_PROJECT=${WANDB_PROJECT:-vllm-gb200-throughput-benchmark}
export WANDB_ENTITY=${WANDB_ENTITY:-}
export WANDB_UPLOAD=${WANDB_UPLOAD:-1}

echo "============================================================"
echo "vLLM Throughput Benchmark - GB200 Models"
echo "============================================================"
echo "Reference: model_configs.yaml (GB200 settings)"
echo ""
echo "Common settings:"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  NUM_GENERATIONS (n): $NUM_GENERATIONS"
echo "  MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo ""
echo "Input/Output Configurations (Total=4096 for Megatron-Bridge comparison):"
echo "  IO_PAIRS: $IO_PAIRS"
echo "  INPUT_LENS:  $INPUT_LENS"
echo "  OUTPUT_LENS: $OUTPUT_LENS"
echo ""
echo "Weights & Biases:"
if [ "$WANDB_UPLOAD" == "1" ]; then
    echo "  WANDB_PROJECT: $WANDB_PROJECT"
    [ -n "$WANDB_ENTITY" ] && echo "  WANDB_ENTITY: $WANDB_ENTITY"
    echo "  (auto-upload enabled)"
else
    echo "  (disabled)"
fi
echo ""
echo "Start time: $(date)"
echo ""

# ============================================================
# Llama-3.1-8B-Instruct (from model_configs.yaml llama8b.gb200)
# ============================================================
# generation: tp=1, pp=1 → uses 1 GPU
# (NUM_NODES auto-adjusted by run_vllm_benchmark.sh)
# ============================================================
echo "[1/4] Submitting Llama-3.1-8B-Instruct..."
echo "      Config: TP=1, PP=1 (1 GPU)"
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct \
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
# generation: tp=1, pp=1 → uses 1 GPU
# (NUM_NODES auto-adjusted by run_vllm_benchmark.sh)
# ============================================================
echo "[2/4] Submitting Qwen3-32B..."
echo "      Config: TP=1, PP=1 (1 GPU)"
MODEL_PATH=Qwen/Qwen3-32B \
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
# generation: tp=1, pp=1 → uses 1 GPU
# Note: This is a MoE model, EP is for training only
# (NUM_NODES auto-adjusted by run_vllm_benchmark.sh)
# ============================================================
echo "[3/4] Submitting Qwen3-30B-A3B (MoE)..."
echo "      Config: TP=1, PP=1 (1 GPU)"
MODEL_PATH=Qwen/Qwen3-30B-A3B \
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
# Llama-3.1-70B-Instruct (from model_configs.yaml llama70b.gb200)
# ============================================================
# generation: tp=2, pp=1 → uses 2 GPUs
# (NUM_NODES auto-adjusted by run_vllm_benchmark.sh)
# ============================================================
echo "[4/4] Submitting Llama-3.1-70B-Instruct..."
echo "      Config: TP=2, PP=1 (2 GPUs)"
MODEL_PATH=meta-llama/Llama-3.1-70B-Instruct \
TP_SIZE=2 \
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
echo "  vllm_standalone_perf_exp/throughput/<model>/<parallelism>/<job_id>-logs/"
if [ "$WANDB_UPLOAD" == "1" ]; then
    echo ""
    echo "wandb dashboard: https://wandb.ai/${WANDB_ENTITY:-$USER}/$WANDB_PROJECT"
fi
