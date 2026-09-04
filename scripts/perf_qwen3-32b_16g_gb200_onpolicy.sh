#!/bin/bash
# Perf tracker: Qwen series / on-policy / GB200 -- Qwen3-32B, 16 GPU (rows 11,15).
#   GRPO, OpenMathInstruct-2, seqlen 4096, BF16, colocated, rollout GBS 2048,
#   train GBS 512. gen TP2, train TP2/PP4 (recipe grpo-qwen3-32b-4n4g, v0.7 row).
# JSONL -> dp_inflight_profiles/qwen3-32b_16g-gb200_onpolicy/inflight_timeline.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."

RECIPE=grpo-qwen3-32b-4n4g \
NUM_ACTOR_NODES=4 GPUS_PER_NODE=4 \
MODEL=${MODEL:-${WORK_DIR:?set MODEL, or WORK_DIR to a shared path holding the HF cache}/hf_home/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137} \
RUN_TAG=qwen3-32b_16g-gb200_onpolicy \
PROFILE_INFLIGHT=1 \
WALLTIME=${WALLTIME:-00:30:00} \
bash ./scripts/run_qwen_perf.sh
