#!/bin/bash
# Perf tracker: Qwen series / on-policy / GB200 -- Qwen3-30B-A3B (MoE), 16 GPU
#   (rows 25,30,35). GRPO, OpenMathInstruct-2, seqlen 4096, BF16, colocated,
#   rollout GBS 2048, train GBS 512. gen TP1/DP16, train EP16/TP1 (MoE expert
#   parallel). Recipe grpo-qwen3-30ba3b-4n4g.
# NOTE: Qwen3-30B-A3B is NOT cached locally (~60GB) -- first run downloads it to
#   HF_HOME. Pre-stage it or expect added startup time.
# JSONL -> dp_inflight_profiles/qwen3-30ba3b_16g-gb200_onpolicy/inflight_timeline.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."

RECIPE=grpo-qwen3-30ba3b-4n4g \
NUM_ACTOR_NODES=4 GPUS_PER_NODE=4 \
MODEL=${MODEL:-} \
RUN_TAG=qwen3-30ba3b_16g-gb200_onpolicy \
PROFILE_INFLIGHT=1 \
WALLTIME=${WALLTIME:-00:40:00} \
bash ./scripts/run_qwen_perf.sh
