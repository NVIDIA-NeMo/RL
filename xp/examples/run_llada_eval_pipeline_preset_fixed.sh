#!/bin/bash
# Minimal preset launcher for the Nemotron diffusion evaluation pipeline.
# This version uses the --use-same-node fix for proper network connectivity.

set -euo pipefail

export SERVER_GPUS=8
export SERVER_BATCH_SIZE=1
export SERVER_BASE_MODEL="nvidia/Nemotron-Diffusion-Research-4B-v0"
export SERVER_DCP_PATH="/lustre/fsw/portfolios/llmservice/users/degert/results/diffusion_sft-reasoning_off_identity_fix_math_new_r1_strict-filter_holdout_rl_15percent-OCI-4b-nvidia-diffusion-qwen3-epochs-3-gbs-256-lr-2.5e6-lambda/step_1100/policy"
export SERVER_ENGINE="nemotron"

export SEQ_EVAL_BENCHMARK="gsm8k:1"
export SEQ_EVAL_GENERATION_ALGORITHM="nemotron"
export SEQ_EVAL_THRESHOLD="0.9"
export SEQ_EVAL_TOKENS_TO_GENERATE="512"
export SEQ_EVAL_STEPS="512"
export SEQ_EVAL_BLOCK_LENGTH="32"
export SEQ_EVAL_EXTRA_ARGS="--model nemotron-4b"

# RECOMMENDED FIX: Run eval on same node as server for network connectivity
export SEQ_EVAL_USE_SAME_NODE="true"
# Also change partition to same as server so the node is available
export SEQ_EVAL_PARTITION="interactive"

export PARALLEL_EVAL_JOBS_OVERRIDE=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_llada_eval_pipeline.sh"

