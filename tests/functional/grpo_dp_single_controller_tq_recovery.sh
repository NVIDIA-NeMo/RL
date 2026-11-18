#!/bin/bash
# Two-process functional test for native TQ + metadata-only replay recovery.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
BASE_TEST=$SCRIPT_DIR/grpo_dp_single_controller.sh
# simple / mooncake_cpu / nixl, or a variant of one:
#   simple_placed   simple with SimpleStorageUnits pinned (storage_unit_placement=all)
#   mooncake_units  mooncake_cpu with CPU storage units as the only memory owners
#   nixl_ssd        nixl with its units' SSD tier on
# mooncake_cpu and nixl_ssd save and restore with values on SSD (a run holds
# up to ~400 KB, so each sets a limit below that) and fail unless SSD was used.
VARIANT=${1:-simple}
VARIANT_OVERRIDES=()
# check_metrics.py expressions run on both phases' metrics.
VARIANT_CHECKS=()
case "$VARIANT" in
    simple) BACKEND=simple; TEST_NAME=grpo_dp_single_controller_tq_recovery ;;
    mooncake_cpu)
        BACKEND=mooncake_cpu
        TEST_NAME=grpo_dp_mooncake_tq_recovery
        SSD_DIR=$SCRIPT_DIR/$TEST_NAME/ssd
        # Offload past 1e-6 of the clients' 4 GiB segments: a few KB each.
        VARIANT_OVERRIDES=(
            data_plane.mooncake_cpu.ssd_offload.enabled=true
            data_plane.mooncake_cpu.ssd_offload.path="$SSD_DIR"
            data_plane.mooncake_cpu.ssd_offload.offload_threshold=0.000001
        )
        ;;
    simple_placed)
        BACKEND=simple
        TEST_NAME=grpo_dp_simple_placed_tq_recovery
        VARIANT_OVERRIDES=(data_plane.simple.storage_unit_placement=all)
        ;;
    mooncake_units)
        BACKEND=mooncake_cpu
        TEST_NAME=grpo_dp_mooncake_units_tq_recovery
        VARIANT_OVERRIDES=(data_plane.mooncake_cpu.storage_unit_segment_size=4294967296)
        ;;
    nixl)
        BACKEND=nixl
        TEST_NAME=grpo_dp_nixl_tq_recovery
        # Per-step unit metrics the driver logs (tq_nixl.unit_metrics).
        VARIANT_CHECKS=('max(data["data_plane/nixl/retired_mb_sum"]) == 0')
        ;;
    nixl_ssd)
        BACKEND=nixl
        TEST_NAME=grpo_dp_nixl_ssd_tq_recovery
        SSD_DIR=$SCRIPT_DIR/$TEST_NAME/ssd
        # 256 KiB slab per unit: a 128 KiB put area and a 128 KiB SSD IO area.
        VARIANT_OVERRIDES=(
            data_plane.nixl.unit_slab_bytes=262144
            data_plane.nixl.ssd.enabled=true
            data_plane.nixl.ssd.path="$SSD_DIR"
            data_plane.nixl.ssd.staging_bytes=131072
        )
        # nixl unlinks its SSD files once open, so count spills instead: the
        # units spill in both phases (training, and the restore into them).
        VARIANT_CHECKS=(
            'max(data["data_plane/nixl/retired_mb_sum"]) == 0'
            'max(data["data_plane/nixl/ssd_spills_sum"]) > 0'
        )
        ;;
    *) echo "Unsupported recovery variant: $VARIANT" >&2; exit 1 ;;
esac
TEST_DIR=$SCRIPT_DIR/$TEST_NAME
CHECKPOINT_DIR=$TEST_DIR/checkpoints
BASE_RUN_LOG=$TEST_DIR/run/run.log
PHASE1_LOG=$TEST_DIR/phase1.log
PHASE2_LOG=$TEST_DIR/phase2.log

rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

COMMON_OVERRIDES=(
    data_plane.backend="$BACKEND"
    checkpointing.enabled=true
    checkpointing.checkpoint_dir="$CHECKPOINT_DIR"
    checkpointing.save_period=1
    checkpointing.save_data_plane=true
    async_rl.sampler.name=windowed
    '~async_rl.sampler.max_lookahead_versions'
    '+async_rl.sampler.max_staleness_versions=1'
    async_rl.max_inflight_prompts=8
    async_rl.max_buffered_rollouts=8
    # Bridge rewrites the scheduler position on resume only when this is on.
    '+policy.megatron_cfg.scheduler.override_opt_param_scheduler=true'
    # Every get re-checks each row's put-time hash (restored rows too); a
    # mismatch is logged, and checked for after each phase.
    data_plane.observability.verify_tensor_hash=true
    "${VARIANT_OVERRIDES[@]}"
)
if [[ "$BACKEND" == "mooncake_cpu" ]]; then
    COMMON_OVERRIDES+=(
        data_plane.mooncake_cpu.global_segment_size=4294967296
        data_plane.mooncake_cpu.local_buffer_size=2147483648
    )
fi

if [[ "$VARIANT" == "mooncake_cpu" ]]; then
    # Offload files are deleted once their keys are cleared, so record the
    # peak count while both phases run (not Mooncake's .lock).
    SSD_PEAK=$TEST_DIR/ssd_peak
    echo 0 > "$SSD_PEAK"
    (
        # Not errexit/pipefail: find fails until Mooncake creates the dir.
        set +eo pipefail
        peak=0
        while sleep 1; do
            n=$(find "$SSD_DIR" -type f ! -name '.lock*' 2>/dev/null | wc -l)
            if (( n > peak )); then peak=$n; echo "$peak" > "$SSD_PEAK"; fi
        done
    ) &
    SSD_WATCHER=$!
    trap 'kill "$SSD_WATCHER" 2>/dev/null || true' EXIT
fi

echo "=== Phase 1: save an authoritative native TQ checkpoint ==="
# Keep the two-step training horizon identical across both processes so the
# Megatron optimizer scheduler can be restored. The timeout makes phase 1 save
# after its first completed step and exit early, simulating an interrupted job.
# Each phase also runs the base test's convergence checks on its own steps.
EXP_NAME="$TEST_NAME/run" bash "$BASE_TEST" \
    "${COMMON_OVERRIDES[@]}" \
    grpo.max_num_steps=2 \
    checkpointing.checkpoint_must_save_by=0:0:0:1
cp "$BASE_RUN_LOG" "$PHASE1_LOG"
# Phase 2 re-runs the base test, which deletes its log dir; keep phase 1's.
cp -r "$TEST_DIR/run/logs" "$TEST_DIR/phase1_logs"

test -d "$CHECKPOINT_DIR/step_1/data_plane"
test -f "$CHECKPOINT_DIR/step_1/replay_buffer_metadata.pt"
test ! -f "$CHECKPOINT_DIR/step_1/replay_buffer.pt"
if [[ "$BACKEND" == "mooncake_cpu" ]]; then
    test -s "$CHECKPOINT_DIR/step_1/data_plane/mooncake_storage/manifest.json"
fi
REPLAY_GROUP_COUNT=$(uv run --directory "$PROJECT_ROOT" --no-sync python -c \
    'import json, sys; metadata = json.load(open(sys.argv[1]))["user_metadata"]; assert metadata["mode"] == "authoritative"; assert metadata["replay_group_count"] > 0, metadata; print(metadata["replay_group_count"])' \
    "$CHECKPOINT_DIR/step_1/data_plane/metadata.json")

echo "=== Phase 2: start a fresh process, restore TQ, and train one more step ==="
EXP_NAME="$TEST_NAME/run" bash "$BASE_TEST" \
    "${COMMON_OVERRIDES[@]}" grpo.max_num_steps=2
cp "$BASE_RUN_LOG" "$PHASE2_LOG"

# Both phases are still in LR warmup, so the resumed LR must be higher than
# phase 1's. A scheduler reset on resume makes them equal.
uv run --directory "$PROJECT_ROOT" tests/json_dump_tb_logs.py "$TEST_DIR/phase1_logs" \
    --output_path "$TEST_DIR/phase1_metrics.json"
uv run --directory "$PROJECT_ROOT" tests/json_dump_tb_logs.py "$TEST_DIR/run/logs" \
    --output_path "$TEST_DIR/phase2_metrics.json"
uv run --directory "$PROJECT_ROOT" python - "$TEST_DIR/phase1_metrics.json" "$TEST_DIR/phase2_metrics.json" <<'EOF'
import json
import sys

before, after = (json.load(open(path))["train/lr"] for path in sys.argv[1:3])
assert min(after.values()) > max(before.values()), (
    f"train/lr did not advance across the resume: {before} -> {after}"
)
print(f"✓ train/lr advanced across the resume: {before} -> {after}")
EOF

grep -q "Native TQ checkpoint restored and validated: groups=${REPLAY_GROUP_COUNT}" "$PHASE2_LOG"
grep -q "Native TQ replay inventory validated" "$PHASE2_LOG"
grep -qF "Restored ${REPLAY_GROUP_COUNT} replay group(s) from checkpoint" "$PHASE2_LOG"
test -d "$CHECKPOINT_DIR/step_2/data_plane"
test -f "$CHECKPOINT_DIR/step_2/replay_buffer_metadata.pt"
if [[ "$BACKEND" == "mooncake_cpu" ]]; then
    test -s "$CHECKPOINT_DIR/step_2/data_plane/mooncake_storage/manifest.json"
fi

if [[ "$VARIANT" == "mooncake_cpu" ]]; then
    PEAK=$(cat "$SSD_PEAK")
    if (( PEAK == 0 )); then
        echo "FAIL: Mooncake SSD offload never wrote a file under $SSD_DIR" >&2
        exit 1
    fi
    echo "✓ Mooncake SSD offload held up to $PEAK file(s) across save and restore"
fi

# Hash guard: rows were checked, and no process logged a mismatch or a guard
# failure (workers' lines reach the run log through Ray).
for PHASE in phase1 phase2; do
    if grep -E "data-plane hash (mismatch|guard failed)" "$TEST_DIR/$PHASE.log"; then
        echo "FAIL: data-plane hash check failed in $PHASE" >&2
        exit 1
    fi
    uv run --directory "$PROJECT_ROOT" tests/check_metrics.py \
        "$TEST_DIR/${PHASE}_metrics.json" \
        'max({**data.get("data_plane/cluster/step/hash/rows_checked", {}), **data.get("data_plane/driver/step/hash/rows_checked", {})}) > 0' \
        "${VARIANT_CHECKS[@]}"
done

if [[ "$BACKEND" == "simple" ]]; then
    echo "=== Verify the standalone Simple TQ checkpoint CLI round trip ==="
    uv run --directory "$PROJECT_ROOT" --no-sync python \
        tools/verify_tq_data_plane_checkpoint.py \
        --checkpoint-dir "$TEST_DIR/verifier_bundle"
fi

echo "Native TQ recovery functional test passed ($VARIANT)."
