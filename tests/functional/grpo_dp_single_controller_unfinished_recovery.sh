#!/bin/bash
# Two-process functional test for one admitted, unfinished rollout group.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
BASE_TEST=$SCRIPT_DIR/grpo_dp_single_controller.sh
TEST_DIR=$SCRIPT_DIR/grpo_dp_single_controller_unfinished_recovery
CHECKPOINT_DIR=$TEST_DIR/checkpoints
BASE_RUN_LOG=$SCRIPT_DIR/grpo_dp_single_controller/run.log
PHASE1_LOG=$TEST_DIR/phase1.log
PHASE2_LOG=$TEST_DIR/phase2.log
PHASE1_EVENTS=$TEST_DIR/phase1-events.jsonl
PHASE2_EVENTS=$TEST_DIR/phase2-events.jsonl
RECOVERY_HOOK=$SCRIPT_DIR/_single_controller_rollout_recovery_hook.py

rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

COMMON_OVERRIDES=(
    checkpointing.enabled=true
    checkpointing.checkpoint_dir="$CHECKPOINT_DIR"
    checkpointing.save_period=1
    checkpointing.save_data_plane=true
    async_rl.sampler.name=in_order
    async_rl.sampler.max_lookahead_versions=1
    async_rl.max_inflight_prompts=4
    async_rl.max_buffered_rollouts=4
    # A recovery-ordering regression otherwise leaves zero rollouts in flight and
    # only warns forever. Bound both slow generation and whole-run stalls in CI.
    ++async_rl.rollout_failure.native.generation_timeout_s=60
    ++async_rl.stall_watchdog.interval_s=10
    ++async_rl.stall_watchdog.stall_timeout_s=180
    ++async_rl.stall_watchdog.stall_action=abort
)

echo "=== Phase 1: checkpoint one admitted rollout before its TQ commit ==="
# The wrapper permanently parks one target-step-1 group. save_period=1 captures
# that ownership after train step 1; checkpoint_must_save_by only terminates the
# first process afterward. No sleep determines whether the group is unfinished.
SC_TEST_ENTRYPOINT="$RECOVERY_HOOK" \
SC_RECOVERY_TEST_EVENTS="$PHASE1_EVENTS" \
SC_RECOVERY_TEST_BLOCK_TARGET_STEP=1 \
RUN_CONVERGENCE_CHECKS=0 bash "$BASE_TEST" \
    "${COMMON_OVERRIDES[@]}" \
    grpo.max_num_steps=2 \
    checkpointing.checkpoint_must_save_by=0:0:0:1
cp "$BASE_RUN_LOG" "$PHASE1_LOG"

STEP1=$CHECKPOINT_DIR/step_1
test -d "$STEP1/data_plane"
test -f "$STEP1/replay_buffer_metadata.pt"
test -f "$STEP1/rollout_recovery.pt"
test ! -f "$STEP1/replay_buffer.pt"
uv run --directory "$PROJECT_ROOT" --no-sync python -c \
    'import json, sys; metadata = json.load(open(sys.argv[1]))["user_metadata"]; assert metadata["mode"] == "authoritative", metadata; assert metadata["rollout_recovery_group_count"] > 0, metadata' \
    "$STEP1/data_plane/metadata.json"
BLOCKED_GROUP_ID=$(uv run --directory "$PROJECT_ROOT" --no-sync python -c \
    'import json, sys; events = [json.loads(line) for line in open(sys.argv[1])]; blocked = [event for event in events if event["event"] == "blocked_before_tq_commit"]; assert len(blocked) == 1, blocked; print(blocked[0]["group_id"])' \
    "$PHASE1_EVENTS")
uv run --directory "$PROJECT_ROOT" --no-sync python -c \
    'import sys, torch; state = torch.load(sys.argv[1], weights_only=True); group_id = sys.argv[2]; groups = [group for group in state["groups"] if group["group_id"] == group_id]; assert len(groups) == 1, state; assert groups[0]["phase"] == "admitted", groups[0]' \
    "$STEP1/rollout_recovery.pt" "$BLOCKED_GROUP_ID"

echo "=== Phase 2: restore and canonically commit the same logical group once ==="
SC_TEST_ENTRYPOINT="$RECOVERY_HOOK" \
SC_RECOVERY_TEST_EVENTS="$PHASE2_EVENTS" \
RUN_CONVERGENCE_CHECKS=0 bash "$BASE_TEST" \
    "${COMMON_OVERRIDES[@]}" grpo.max_num_steps=2
cp "$BASE_RUN_LOG" "$PHASE2_LOG"

grep -q "Native TQ checkpoint restored and validated" "$PHASE2_LOG"
grep -q "Loaded .* unfinished rollout group(s)" "$PHASE2_LOG"
test -d "$CHECKPOINT_DIR/step_2/data_plane"
test -f "$CHECKPOINT_DIR/step_2/replay_buffer_metadata.pt"
test -f "$CHECKPOINT_DIR/step_2/rollout_recovery.pt"
uv run --directory "$PROJECT_ROOT" --no-sync python -c \
    'import json, sys; events = [json.loads(line) for line in open(sys.argv[1])]; group_id = sys.argv[2]; dispatches = [event for event in events if event["event"] == "dispatch" and event["group_id"] == group_id]; commits = [event for event in events if event["event"] == "canonical_tq_commit" and event["group_id"] == group_id]; assert len(dispatches) == 1, dispatches; assert len(commits) == 1, commits' \
    "$PHASE2_EVENTS" "$BLOCKED_GROUP_ID"
uv run --directory "$PROJECT_ROOT" --no-sync python -c \
    'import sys, torch; state = torch.load(sys.argv[1], weights_only=True); group_id = sys.argv[2]; assert group_id not in {group["group_id"] for group in state["groups"]}, state' \
    "$CHECKPOINT_DIR/step_2/rollout_recovery.pt" "$BLOCKED_GROUP_ID"

echo "Unfinished rollout recovery functional test passed."
