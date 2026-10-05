#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
source /opt/nemo-rl/docker/nano35/environment.sh
cd /opt/nemo-rl
# ray.sub removes Slurm launcher variables before running the driver. Keep our
# bookkeeping ID separate so it is not mistaken for an MPI launcher context.
TASK_JOB_ID=${NANO35_JOB_ID:-${SLURM_JOB_ID:-}}
if [[ -z "$TASK_JOB_ID" ]]; then
    read -r TASK_JOB_ID < "$NANO35_RUN_DIR/job-id.txt"
fi
[[ "$TASK_JOB_ID" =~ ^[0-9]+$ ]] || { echo 'Invalid training job ID' >&2; exit 2; }
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
export NRL_GEN_BENCHMARK_SKIP_TRAINING=0 NRL_USE_FASTOKENS=0
export NRL_FORCE_REBUILD_VENVS=false TRTLLM_REQUIRE_CACHED_WHEEL=1
export TRTLLM_USE_MAMBA_FI_SSD=0 TLLM_LOG_LEVEL=INFO OMP_NUM_THREADS=16
export WANDB_MODE=disabled
export UV_CACHE_DIR="/tmp/nano35-${TASK_JOB_ID}/uv"
export TORCHINDUCTOR_CACHE_DIR="/tmp/nano35-${TASK_JOB_ID}/inductor"
export TRITON_CACHE_DIR="/tmp/nano35-${TASK_JOB_ID}/triton"
export HF_HOME="${NANO35_RUN_DIR}/hf-home"
export NRL_MEGATRON_CHECKPOINT_DIR="${NANO35_RUN_DIR}/model-conversion"
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000 TORCH_NCCL_DUMP_ON_TIMEOUT=true
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${NANO35_RUN_DIR}/nccl_traces/nccl_trace_rank_"
mkdir -p "$UV_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
    "$NANO35_RUN_DIR/nccl_traces" "$NANO35_RUN_DIR/attempts/$TASK_JOB_ID"
uv run --no-sync tools/nano35/prepare_run.py "$NANO35_START_MODE"
if [[ "$NANO35_START_MODE" == resume ]]; then
    python-MegatronPolicyWorker tools/nano35/verify_checkpoint.py \
        "$NANO35_RUN_DIR/checkpoints" \
        --output "$NANO35_RUN_DIR/attempts/$TASK_JOB_ID/resume-input.json"
fi
uv run --no-sync examples/nemo_gym/run_grpo_nemo_gym.py --config "$NANO35_RUN_DIR/config.resolved.yaml"
python-MegatronPolicyWorker tools/nano35/verify_checkpoint.py \
    "$NANO35_RUN_DIR/checkpoints" \
    --output "$NANO35_RUN_DIR/attempts/$TASK_JOB_ID/checkpoint-report.json"
printf 'completed\n' > "$NANO35_RUN_DIR/attempts/$TASK_JOB_ID/driver-status.txt"
