#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# ===== BEGIN CONFIG =====
# Mirrors grpo-qwen2.5-math-1.5b-instruct-1n8g-megatron-single-controller-sync.sh
# (delegated base).
NUM_NODES=1
STEPS_PER_RUN=450
MAX_STEPS=450
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=240
# ===== END CONFIG =====

source "$SCRIPT_DIR/common-tq.env"
# Run base script under this wrapper's identity (own log/ckpt dirs, wandb name).
# The matching TQ YAML inherits from <base>.yaml and turns on data_plane.
export EXP_NAME="$TQ_EXP_NAME"

# The YAML turns on data_plane.mooncake_cpu.ssd_offload; the path is overridden
# below with a fresh per-run dir so a stale dir from another run cannot satisfy
# the check. Watch for the first offloaded file during the run rather than
# after it, in case Mooncake clears the store on close.
SSD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/nrl_mooncake_ssd_XXXXXX")
SSD_SEEN="$SSD_DIR.seen"
( until [[ -n "$(find "$SSD_DIR" -type f -size +0 -print -quit 2>/dev/null)" ]]; do
    sleep 5
done; touch "$SSD_SEEN" ) &
SSD_WATCH_PID=$!
trap 'kill "$SSD_WATCH_PID" 2>/dev/null || true; rm -rf "$SSD_DIR" "$SSD_SEEN"' EXIT

bash "$SCRIPT_DIR/$BASE_RECIPE.sh" \
    data_plane.mooncake_cpu.ssd_offload.path="$SSD_DIR" \
    "$@"

# The delegated base runs in a subshell, so common.env's TEST_DRYRUN exit
# does not reach here. Skip explicitly, or the dryrun check in
# tests/unit/test_recipes_and_test_suites.py fails on a missing metrics.json.
if [[ -n "${TEST_DRYRUN:-}" ]]; then exit 0; fi
if [[ ! -e "$SSD_SEEN" ]]; then
    echo "[ERROR] ssd_offload was enabled but no file ever appeared under $SSD_DIR" >&2
    exit 1
fi
# The base recipe checks the wire hash only once MAX_STEPS is reached; check it
# on every run, since offloaded values are read back from SSD.
cd "$SCRIPT_DIR/../../.."
uv run tests/check_metrics.py "$SCRIPT_DIR/$TQ_EXP_NAME/metrics.json" \
    'max({**data.get("data_plane/cluster/step/hash/mismatches", {}), **data.get("data_plane/driver/step/hash/mismatches", {})}) == 0' \
    'max({**data.get("data_plane/cluster/step/hash/rows_checked", {}), **data.get("data_plane/driver/step/hash/rows_checked", {})}) > 0'
