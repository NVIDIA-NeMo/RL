#!/bin/bash
# Alternative preset: Try running server and eval on same node with proper partitions
#
# Configuration:
# - Server runs on 'interactive' partition (REQUIRED for GPU access)
# - Eval runs on 'cpu' partition (for resource allocation)
# - Uses --use-same-node to attempt co-location
#
# This will ONLY work if:
# - The node allocated by 'interactive' partition is also accessible to 'cpu' partition
# - OR the node has enough resources for both jobs
#
# If this fails with node allocation errors, use run_llada_eval_pipeline_preset.sh instead.

set -euo pipefail

# Use defaults if not already set (allows override from parent scripts)
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/lustre/fsw/portfolios/llmservice/users/$USER/llada-eval}"

# Run SERVER on interactive partition (REQUIRED for GPU access)
export SERVER_PARTITION="${SERVER_PARTITION:-batch_singlenode}"
export SERVER_INFO_FILE="${SERVER_INFO_FILE:-${EVAL_OUTPUT_DIR}/server_info.env}"
export SERVER_GPUS="${SERVER_GPUS:-8}"
export SERVER_BATCH_SIZE="${SERVER_BATCH_SIZE:-1}"
#export SERVER_MODEL_PATH="${SERVER_MODEL_PATH:-nvidia/Nemotron-Diffusion-Research-4B-v0}"
export SERVER_BASE_MODEL="${SERVER_BASE_MODEL:-nvidia/Nemotron-Diffusion-Research-4B-v0}"
export SERVER_DCP_PATH="${SERVER_DCP_PATH:-}"
export SERVER_ENGINE="${SERVER_ENGINE:-nemotron}"
export SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:---verbose}"

# Run EVAL on cpu partition (attempt same node via --use-same-node)
export SEQ_EVAL_PARTITION="${SEQ_EVAL_PARTITION:-cpu}"
export SEQ_EVAL_BENCHMARK="${SEQ_EVAL_BENCHMARK:-gsm8k:1}"
export SEQ_EVAL_OUTPUT_DIR="${SEQ_EVAL_OUTPUT_DIR:-${EVAL_OUTPUT_DIR}}"
export SEQ_EVAL_GENERATION_ALGORITHM="${SEQ_EVAL_GENERATION_ALGORITHM:-nemotron}"
export SEQ_EVAL_THRESHOLD="${SEQ_EVAL_THRESHOLD:-0.9}"
export SEQ_EVAL_TOKENS_TO_GENERATE="${SEQ_EVAL_TOKENS_TO_GENERATE:-512}"
export SEQ_EVAL_STEPS="${SEQ_EVAL_STEPS:-512}"
export SEQ_EVAL_BLOCK_LENGTH="${SEQ_EVAL_BLOCK_LENGTH:-32}"
export SEQ_EVAL_TEMPERATURE="${SEQ_EVAL_TEMPERATURE:-0}"
export SEQ_EVAL_EXTRA_ARGS="${SEQ_EVAL_EXTRA_ARGS:---model nemotron-4b}"
export SEQ_EVAL_EXPNAME="${SEQ_EVAL_EXPNAME:-}"

# Enable same-node execution for guaranteed connectivity
export SEQ_EVAL_USE_SAME_NODE="true"

export PARALLEL_EVAL_JOBS_OVERRIDE=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_llada_eval_pipeline.sh"

