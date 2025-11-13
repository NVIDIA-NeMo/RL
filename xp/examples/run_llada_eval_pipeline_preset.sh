#!/bin/bash
# RECOMMENDED preset launcher for the Nemotron diffusion evaluation pipeline.
#
# Configuration:
# - Server runs on 'interactive' partition (REQUIRED for GPU access)
# - Eval runs on 'cpu' partition (for resource allocation)
# - Health check is SKIPPED (nodes on different partitions can't communicate)
#
# This is the correct configuration when eval must use 'cpu' partition.

set -euo pipefail

# Server configuration (runs on 'interactive' partition for GPU access)
export SERVER_PARTITION="interactive"
export SERVER_GPUS=8
export SERVER_BATCH_SIZE=1
export SERVER_BASE_MODEL="nvidia/Nemotron-Diffusion-Research-4B-v0"
export SERVER_DCP_PATH="/lustre/fsw/portfolios/llmservice/users/degert/results/diffusion_sft-reasoning_off_identity_fix_math_new_r1_strict-filter_holdout_rl_15percent-OCI-4b-nvidia-diffusion-qwen3-epochs-3-gbs-256-lr-2.5e6-lambda/step_1100/policy"
export SERVER_ENGINE="nemotron"

# Eval configuration (runs on 'cpu' partition)
export SEQ_EVAL_PARTITION="cpu"
export SEQ_EVAL_BENCHMARK="gsm8k:1"
export SEQ_EVAL_GENERATION_ALGORITHM="nemotron"
export SEQ_EVAL_THRESHOLD="0.9"
export SEQ_EVAL_TOKENS_TO_GENERATE="512"
export SEQ_EVAL_STEPS="512"
export SEQ_EVAL_BLOCK_LENGTH="32"
export SEQ_EVAL_EXTRA_ARGS="--model nemotron-4b"

# NETWORK FIX: Skip health check since eval must run on 'cpu' partition
# which likely can't reach GPU nodes on 'interactive' partition
export SEQ_EVAL_NO_WAIT_SERVER="true"

# Note: Cannot use SEQ_EVAL_USE_SAME_NODE="true" because the cpu partition
# doesn't include GPU nodes where the server runs (interactive partition)

export PARALLEL_EVAL_JOBS_OVERRIDE=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_llada_eval_pipeline.sh"
