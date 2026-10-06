#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Paired 20-step comparison of context compaction against no compaction on the
# same prompts, in the same order, from the same step-120 weights.
#   CC_ARMS="compaction identity" bash scripts/agentic_vbench/compare_cc_20steps.sh
# Arms:
#   compaction  token_capture.context_compaction=true, OpenCode compaction.auto=true
#   identity    token_capture.context_compaction=true, OpenCode compaction.auto=false
#               (framework-owned route, no context rewrite: one row per rollout)
#   off         token_capture.context_compaction=false, ordinary capture
# OpenCode's tool-output pruning is off in every arm (also a context rewrite).
# Every arm shares TRAIN_PATH, data.shuffle=false and the policy seed, so the
# only difference is the compaction policy. Reward/response-length/row-count
# curves in W&B (one run per arm) are the comparison; both arms also keep their
# capture ledgers under results/<arm>/token_capture for the saved-evidence audit.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
code_dir=$(cd "$script_dir/../.." && pwd)
export AVB_DATA_ROOT=${AVB_DATA_ROOT:-$code_dir/data/agentic_vbench}
export MAX_STEPS=${MAX_STEPS:-20}
export NUM_PROMPTS=${NUM_PROMPTS:-8}
export NUM_GENERATIONS=${NUM_GENERATIONS:-8}
export SAVE_PERIOD=${SAVE_PERIOD:-10}
export CHECKPOINT_KEEP_TOP_K=${CHECKPOINT_KEEP_TOP_K:-2}
export NUM_NODES=${NUM_NODES:-16}
export NUM_GEN_NODES=${NUM_GEN_NODES:-8}
export SEGMENT_SIZE=${SEGMENT_SIZE:-8}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-12:00:00}
# Long episodes are the point of compaction: 32 turns and 8k per turn against
# the policy's real 49k context, so OpenCode has to compact to keep going.
export AGENTIC_VBENCH_TASKS=${AGENTIC_VBENCH_TASKS:-all}
export TRAIN_REPEATS=${TRAIN_REPEATS:-$(( (MAX_STEPS * NUM_PROMPTS + 99) / 100 + 1 ))}
export AGENTIC_VBENCH_MAX_TURNS=${AGENTIC_VBENCH_MAX_TURNS:-32}
export AGENTIC_VBENCH_CONTEXT_TOKENS=${AGENTIC_VBENCH_CONTEXT_TOKENS:-49152}
export AGENTIC_VBENCH_OUTPUT_TOKENS=${AGENTIC_VBENCH_OUTPUT_TOKENS:-8192}
export AGENTIC_VBENCH_CONCURRENCY=${AGENTIC_VBENCH_CONCURRENCY:-16}
export AGENTIC_VBENCH_COMPACTION_PRUNE=false
export POLICY_SEED=${POLICY_SEED:-42}
stamp=${COMPARE_STAMP:-$(date -u +%Y%m%d-%H%M%S)}
arms=${CC_ARMS:-"compaction identity"}
for arm in $arms; do
  case $arm in
    compaction) cc=true; auto=true ;;
    identity) cc=true; auto=false ;;
    off) cc=false; auto=false ;;
    *) echo "Unknown arm: $arm (compaction|identity|off)" >&2; exit 2 ;;
  esac
  echo "=== arm=$arm context_compaction=$cc opencode_compaction_auto=$auto ==="
  CONTEXT_COMPACTION=$cc AGENTIC_VBENCH_COMPACTION_AUTO=$auto \
    BASE_NAME=${BASE_NAME_PREFIX:-agentic_vbench_cc20}_${stamp}_${arm} \
    WANDB_RUN_NAME=${BASE_NAME_PREFIX:-agentic_vbench_cc20}_${stamp}_${arm} \
    bash "$script_dir/run_sc.sh" data.shuffle=false grpo.seed="$POLICY_SEED" "$@"
done
