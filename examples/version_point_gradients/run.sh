#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ARM=${1:?Usage: run.sh ARM [config overrides...]}
shift
NODES=8
case "$ARM" in
    bf16_lag1) RECIPE=bf16-baseline-lag1; MODE=baseline; LAG=1; GATE_STEPS=5 ;;
    bf16_vpg_lag1) RECIPE=bf16-vpg-lag1; NODES=5; MODE=vpg; LAG=1; GATE_STEPS=5 ;;
    bf16_vpg_lag4) RECIPE=bf16-vpg-lag4; NODES=5; MODE=vpg; LAG=4; GATE_STEPS=6 ;;
    bf16_lag4) RECIPE=bf16-baseline-lag4; MODE=baseline; LAG=4; GATE_STEPS=5 ;;
    nvfp4_vpg_lag1) RECIPE=nvfp4-vpg-lag1; MODE=vpg; LAG=1; GATE_STEPS=5 ;;
    nvfp4_vpg_lag4) RECIPE=nvfp4-vpg-lag4; MODE=vpg; LAG=4; GATE_STEPS=6 ;;
    *) echo "Unknown arm: $ARM" >&2; exit 2 ;;
esac
PHASE=${PHASE:-all}
case "$PHASE" in all|smoke|train) ;; *) echo "Unknown PHASE: $PHASE" >&2; exit 2 ;; esac
PYTHON=${PYTHON:-python3}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO/results/precision_comparison}
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT=$(cd "$OUTPUT_ROOT" && pwd)
OUT=$OUTPUT_ROOT/$ARM
CONFIG=$REPO/examples/configs/recipes/llm/vpg/grpo-qwen3-30ba3b-base-${NODES}n4g-$RECIPE.yaml
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
run_command() {
    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}
run_training() {
    local dir=$1 seed=$2 steps=$3 checkpoint=$4
    shift 4
    run_command uv run --no-project --python "$PYTHON" python examples/run_grpo.py --config "$CONFIG"         "logger.log_dir=$dir/logs" "grpo.seed=$seed" "grpo.max_num_steps=$steps"         "checkpointing.enabled=$checkpoint" "checkpointing.checkpoint_dir=$dir/ckpt"         checkpointing.save_period=25 checkpointing.keep_top_k=1 checkpointing.metric_name=null         "$@"
}
if [ "$PHASE" != train ] && [ ! -s "$OUT/STARTUP_PASSED" ]; then
    run_training "$OUT/smoke_seed901" 901 "$GATE_STEPS" false "$@"
    VERIFY_ARGS=("$MODE" "$GATE_STEPS")
    if [ "$MODE" = vpg ]; then VERIFY_ARGS+=("$LAG"); fi
    run_command uv run --no-project --python "$PYTHON" python "$HERE/verify_startup.py" "$OUT/smoke_seed901" "${VERIFY_ARGS[@]}"
    if [ "${DRY_RUN:-0}" != 1 ]; then date -Is > "$OUT/STARTUP_PASSED"; fi
fi
if [ "$PHASE" != smoke ]; then
    if [ "${DRY_RUN:-0}" != 1 ] && [ ! -s "$OUT/STARTUP_PASSED" ]; then
        echo "Run PHASE=smoke successfully before PHASE=train" >&2
        exit 2
    fi
    run_training "$OUT/seed${SEED:-1}" "${SEED:-1}" 1500 true "$@"
fi
