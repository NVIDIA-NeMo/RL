#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
source /opt/nemo-rl/docker/nano35/environment.sh
export NANO35_RUN_DIR="$(dirname "$NANO35_DATA_PREFLIGHT_OUTPUT")/unused-run"
export NANO35_RUN_NAME=data-preflight
export UV_CACHE_DIR="/tmp/nano35-data-${SLURM_JOB_ID}/uv"
export HF_HOME="/tmp/nano35-data-${SLURM_JOB_ID}/hf"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=/opt/nemo-rl
cd /opt/nemo-rl
uv run --no-sync /nrl-data-check/tools/nano35/check_training_data.py \
    --config examples/configs/recipes/llm/grpo-nano3.5-swe-32n4g-tp4cp16-async-trtllm.v1.yaml \
    --output "$NANO35_DATA_PREFLIGHT_OUTPUT"
