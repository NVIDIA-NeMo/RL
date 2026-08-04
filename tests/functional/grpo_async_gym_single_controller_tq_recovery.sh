#!/bin/bash
# Two-process NeMo-Gym test for pre-step TQ + partial-rollout ledger recovery.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BASE_TEST=$SCRIPT_DIR/grpo_async_gym_single_controller.sh
TEST_DIR=$SCRIPT_DIR/grpo_async_gym_single_controller_tq_recovery
CHECKPOINT_DIR=$TEST_DIR/checkpoints
DATA_DIR=$TEST_DIR/data
PHASE1_DIR=$TEST_DIR/phase1
PHASE2_DIR=$TEST_DIR/phase2
PHASE1_LOG=$PHASE1_DIR/run.log
PHASE2_LOG=$PHASE2_DIR/run.log

rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"
trap 'rm -rf "$CHECKPOINT_DIR"' EXIT

COMMON_OVERRIDES=(
    ++token_capture.enabled=true
    checkpointing.enabled=true
    checkpointing.checkpoint_dir="$CHECKPOINT_DIR"
    checkpointing.save_period=100
    ++data_plane.checkpointing_enabled=true
    async_rl.sampler.name=windowed
    '~async_rl.sampler.max_lookahead_versions'
    '+async_rl.sampler.max_staleness_versions=1'
    async_rl.max_inflight_prompts=8
    async_rl.max_buffered_rollouts=8
    ++rollout_checkpointing.interval_s=0.25
    ++rollout_checkpointing.keep_latest_k=2
    grpo.max_num_steps=10
)

echo "=== Phase 1: stop before train step one after a durable rollout cut ==="
SC_GYM_EXP_DIR="$PHASE1_DIR" \
SC_GYM_RUN_LOG="$PHASE1_LOG" \
SC_GYM_CHECKPOINT_DIR="$CHECKPOINT_DIR" \
SC_GYM_DATA_DIR="$DATA_DIR" \
SC_GYM_KEEP_CHECKPOINTS=1 \
SC_GYM_RUN_CONVERGENCE_CHECKS=0 \
bash "$BASE_TEST" \
    "${COMMON_OVERRIDES[@]}" \
    checkpointing.checkpoint_must_save_by=0:0:0:1

test ! -d "$CHECKPOINT_DIR/step_1"
test -f "$CHECKPOINT_DIR/bootstrap/manifest.json"
test -f "$CHECKPOINT_DIR/bootstrap/rollout_snapshots/LATEST"
SNAPSHOT_NAME=$(tr -d '\n' < "$CHECKPOINT_DIR/bootstrap/rollout_snapshots/LATEST")
SNAPSHOT_DIR=$CHECKPOINT_DIR/bootstrap/rollout_snapshots/$SNAPSHOT_NAME
test -f "$SNAPSHOT_DIR/COMMITTED"
test -d "$SNAPSHOT_DIR/data_plane"
test -f "$SNAPSHOT_DIR/replay_buffer_metadata.pt"
test -f "$SNAPSHOT_DIR/rollout_recovery.pt"
test -f "$SNAPSHOT_DIR/train_dataloader.pt"
test ! -d "$SNAPSHOT_DIR/policy"
uv run --no-sync python -c \
    'import sys, torch; state = torch.load(sys.argv[1], weights_only=False); assert state["groups"], state; print("saved rollout groups=", len(state["groups"]))' \
    "$SNAPSHOT_DIR/rollout_recovery.pt"
grep -q "stopping after a durable rollout snapshot" "$PHASE1_LOG"

echo "=== Phase 2: restore in a fresh process and verify convergence ==="
SC_GYM_EXP_DIR="$PHASE2_DIR" \
SC_GYM_RUN_LOG="$PHASE2_LOG" \
SC_GYM_CHECKPOINT_DIR="$CHECKPOINT_DIR" \
SC_GYM_DATA_DIR="$DATA_DIR" \
SC_GYM_KEEP_CHECKPOINTS=1 \
SC_GYM_RUN_CONVERGENCE_CHECKS=1 \
bash "$BASE_TEST" "${COMMON_OVERRIDES[@]}"

grep -q "Selected rollout recovery snapshot" "$PHASE2_LOG"
grep -q "Native TQ checkpoint restored and validated" "$PHASE2_LOG"
grep -q "Restored rollout recovery ledger" "$PHASE2_LOG"
grep -q "rollout recovery replay completed" "$PHASE2_LOG"
grep -q "train step 10/10" "$PHASE2_LOG"
test -d "$CHECKPOINT_DIR/step_10/data_plane"

echo "Pre-step partial-rollout recovery and convergence test passed."
