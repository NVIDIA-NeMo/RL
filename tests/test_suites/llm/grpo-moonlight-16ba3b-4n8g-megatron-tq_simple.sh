#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# ===== BEGIN CONFIG =====
# Mirrors grpo-moonlight-16ba3b-4n8g-megatron.sh (delegated base).
NUM_NODES=4
STEPS_PER_RUN=30
MAX_STEPS=30
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=60
# ===== END CONFIG =====

source "$SCRIPT_DIR/common-tq.env"
# Run base script under this wrapper's identity (own log/ckpt dirs, wandb name).
# The matching TQ YAML inherits from <base>.yaml and turns on data_plane.
export EXP_NAME="$TQ_EXP_NAME"
bash "$SCRIPT_DIR/$BASE_RECIPE.sh" "$@"

# The wire guard only counts; assert it found nothing.
# The delegated base runs in a subshell, so common.env's TEST_DRYRUN exit
# does not reach here. Skip explicitly, or the dryrun check in
# tests/unit/test_recipes_and_test_suites.py fails on a missing metrics.json.
# `if` form, not `&& exit`: a false `[[ ]]` returns 1 and set -e would abort.
if [[ -n "${TEST_DRYRUN:-}" ]]; then exit 0; fi
cd "$SCRIPT_DIR/../../.."
uv run tests/check_metrics.py "$SCRIPT_DIR/$TQ_EXP_NAME/metrics.json" \
    'max(data.get("data_plane/cluster/step/hash/mismatches", data.get("data_plane/driver/step/hash/mismatches", {}))) == 0' \
    'max(data.get("data_plane/cluster/step/hash/guard_failures", data.get("data_plane/driver/step/hash/guard_failures", {}))) == 0'
