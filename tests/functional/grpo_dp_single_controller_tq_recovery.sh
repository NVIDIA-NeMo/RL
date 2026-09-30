#!/bin/bash
# Two-process functional test for native TQ + metadata-only replay recovery.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
BASE_TEST=$SCRIPT_DIR/grpo_dp_single_controller.sh
BACKEND=${1:-simple}
case "$BACKEND" in
    simple) TEST_NAME=grpo_dp_single_controller_tq_recovery ;;
    mooncake_cpu) TEST_NAME=grpo_dp_mooncake_tq_recovery ;;
    *) echo "Unsupported recovery backend: $BACKEND" >&2; exit 1 ;;
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
)
if [[ "$BACKEND" == "mooncake_cpu" ]]; then
    COMMON_OVERRIDES+=(
        data_plane.mooncake_cpu.global_segment_size=4294967296
        data_plane.mooncake_cpu.local_buffer_size=1073741824
    )
fi

echo "=== Phase 1: save an authoritative native TQ checkpoint ==="
# Keep the two-step training horizon identical across both processes so the
# Megatron optimizer scheduler can be restored. The timeout makes phase 1 save
# after its first completed step and exit early, simulating an interrupted job.
EXP_NAME="$TEST_NAME/run" RUN_CONVERGENCE_CHECKS=0 bash "$BASE_TEST" \
    "${COMMON_OVERRIDES[@]}" \
    grpo.max_num_steps=2 \
    checkpointing.checkpoint_must_save_by=0:0:0:1
cp "$BASE_RUN_LOG" "$PHASE1_LOG"

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
EXP_NAME="$TEST_NAME/run" RUN_CONVERGENCE_CHECKS=0 bash "$BASE_TEST" \
    "${COMMON_OVERRIDES[@]}" grpo.max_num_steps=2
cp "$BASE_RUN_LOG" "$PHASE2_LOG"

grep -q "Native TQ checkpoint restored and validated: groups=${REPLAY_GROUP_COUNT}" "$PHASE2_LOG"
grep -q "Native TQ replay inventory validated" "$PHASE2_LOG"
grep -qF "Restored ${REPLAY_GROUP_COUNT} replay group(s) from checkpoint" "$PHASE2_LOG"
test -d "$CHECKPOINT_DIR/step_2/data_plane"
test -f "$CHECKPOINT_DIR/step_2/replay_buffer_metadata.pt"
if [[ "$BACKEND" == "mooncake_cpu" ]]; then
    test -s "$CHECKPOINT_DIR/step_2/data_plane/mooncake_storage/manifest.json"
fi

if [[ "$BACKEND" == "simple" ]]; then
    echo "=== Verify the standalone Simple TQ checkpoint CLI round trip ==="
    uv run --directory "$PROJECT_ROOT" --no-sync python \
        tools/verify_tq_data_plane_checkpoint.py \
        --checkpoint-dir "$TEST_DIR/verifier_bundle"
fi

echo "Native TQ recovery functional test passed ($BACKEND)."
