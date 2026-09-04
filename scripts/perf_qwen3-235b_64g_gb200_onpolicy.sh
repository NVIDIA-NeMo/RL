#!/bin/bash
# Perf tracker: Qwen series / on-policy / GB200 -- Qwen3-235B-A22B, 64 GPU
#   (rows 45,50). GRPO, OpenMathInstruct-2, seqlen 8192, BF16, colocated,
#   rollout GBS 512, train GBS 512. gen TP8/DP8, train TP2/CP2/EP16/PP4.
#   Recipe grpo-qwen3-235b-16n4g (16 nodes x 4 GPU).
# NOTE: Qwen3-235B-A22B is NOT cached locally (~470GB) and this needs 16 nodes.
#   Pre-stage the model and ensure the allocation before running.
# JSONL -> dp_inflight_profiles/qwen3-235b_64g-gb200_onpolicy/inflight_timeline.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."

RECIPE=grpo-qwen3-235b-16n4g \
NUM_ACTOR_NODES=16 GPUS_PER_NODE=4 \
MODEL=${MODEL:-} \
RUN_TAG=qwen3-235b_64g-gb200_onpolicy \
PROFILE_INFLIGHT=1 \
WALLTIME=${WALLTIME:-00:50:00} \
bash ./scripts/run_qwen_perf.sh
