#!/bin/bash
# Perf tracker: Qwen series / on-policy / GB200 -- Qwen3-235B-A22B, 128 GPU
#   (row 51). Same parallelism as the 64-GPU config (gen TP8/DP8,
#   train TP2/CP2/EP16/PP4) scaled to 32 nodes. Recipe grpo-qwen3-235b-32n4g.
# NOTE: Qwen3-235B-A22B is NOT cached locally (~470GB) and this needs 32 nodes.
#   Pre-stage the model and ensure the allocation before running.
# JSONL -> dp_inflight_profiles/qwen3-235b_128g-gb200_onpolicy/inflight_timeline.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."

RECIPE=grpo-qwen3-235b-32n4g \
NUM_ACTOR_NODES=32 GPUS_PER_NODE=4 \
MODEL=${MODEL:-} \
RUN_TAG=qwen3-235b_128g-gb200_onpolicy \
PROFILE_INFLIGHT=1 \
WALLTIME=${WALLTIME:-00:50:00} \
bash ./scripts/run_qwen_perf.sh
