#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# ===== BEGIN CONFIG =====
# Mirrors vlm_grpo-qwen2.5-vl-3b-instruct-clevr-1n8g-automodel.v1.sh, the
# delegated base, which common-tq.env derives by stripping -tq_nixl.
NUM_NODES=1
# 20 of the base's 200 steps: the wire check below needs no convergence, and
# the base's step-200 reward gate is left to the base recipe's own run.
STEPS_PER_RUN=20
MAX_STEPS=20
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=30
# ===== END CONFIG =====

source "$SCRIPT_DIR/common-tq.env"
# Run base script under this wrapper's identity (own log/ckpt dirs, wandb name).
# The matching TQ YAML inherits from <base>.yaml and turns on data_plane.
export EXP_NAME="$TQ_EXP_NAME"
bash "$SCRIPT_DIR/$BASE_RECIPE.sh" \
    grpo.max_num_steps=$MAX_STEPS \
    data_plane.observability.verify_tensor_hash=True \
    "$@"

# Same wire guard as the LLM TQ wrappers: it only counts, so assert it looked
# (rows_checked) and agreed (mismatches). Skip under dryrun, as they do.
if [[ -n "${TEST_DRYRUN:-}" ]]; then exit 0; fi
cd "$SCRIPT_DIR/../../.."
uv run tests/check_metrics.py "$SCRIPT_DIR/$TQ_EXP_NAME/metrics.json" \
    'max({**data.get("data_plane/cluster/step/hash/mismatches", {}), **data.get("data_plane/driver/step/hash/mismatches", {})}) == 0' \
    'max({**data.get("data_plane/cluster/step/hash/rows_checked", {}), **data.get("data_plane/driver/step/hash/rows_checked", {})}) > 0'
