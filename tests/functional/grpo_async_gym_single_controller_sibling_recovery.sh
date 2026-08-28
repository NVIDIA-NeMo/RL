#!/bin/bash
# Two-process functional test for sibling-level token-capture recovery.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
BASE_TEST=$SCRIPT_DIR/grpo_async_gym_single_controller.sh
TEST_DIR=$SCRIPT_DIR/grpo_async_gym_single_controller_sibling_recovery
CHECKPOINT_DIR=$TEST_DIR/checkpoints
BASE_RUN_LOG=$SCRIPT_DIR/grpo_async_gym_single_controller/run.log
PHASE1_LOG=$TEST_DIR/phase1.log
PHASE2_LOG=$TEST_DIR/phase2.log
PHASE1_EVENTS=$TEST_DIR/phase1-events.jsonl
PHASE2_EVENTS=$TEST_DIR/phase2-events.jsonl
RECOVERY_HOOK=$SCRIPT_DIR/_single_controller_sibling_recovery_hook.py

rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

COMMON_OVERRIDES=(
    checkpointing.enabled=true
    checkpointing.checkpoint_dir="$CHECKPOINT_DIR"
    checkpointing.save_period=1
    checkpointing.save_data_plane=true
    ++token_capture.enabled=true
    ++rollout_recovery.default_granularity=sibling
    async_rl.sampler.name=in_order
    async_rl.sampler.max_lookahead_versions=1
    async_rl.max_inflight_prompts=8
    async_rl.max_buffered_rollouts=8
    ++async_rl.rollout_failure.native.generation_timeout_s=120
    ++async_rl.stall_watchdog.interval_s=10
    ++async_rl.stall_watchdog.stall_timeout_s=300
    ++async_rl.stall_watchdog.stall_action=abort
    grpo.max_num_steps=2
)

echo "=== Phase 1: checkpoint one sealed and one unfinished sibling ==="
# Target step 1 is the lookahead batch while trainer step 1 consumes target
# step 0. The hook holds target-step-0 completions until the selected group has
# sealed one sibling, then parks its next completion before the ledger update.
SC_TEST_ENTRYPOINT="$RECOVERY_HOOK" \
SC_SIBLING_RECOVERY_TEST_EVENTS="$PHASE1_EVENTS" \
SC_SIBLING_RECOVERY_BLOCK_TARGET_STEP=1 \
RUN_CONVERGENCE_CHECKS=0 bash "$BASE_TEST" \
    "${COMMON_OVERRIDES[@]}" \
    checkpointing.checkpoint_must_save_by=0:0:0:1
cp "$BASE_RUN_LOG" "$PHASE1_LOG"

STEP1=$CHECKPOINT_DIR/step_1
test -d "$STEP1/data_plane"
test -f "$STEP1/replay_buffer_metadata.pt"
test -f "$STEP1/rollout_recovery.pt"
PARTIAL_GROUP_ID=$(uv run --directory "$PROJECT_ROOT" --no-sync python -c \
    'import json, sys; events = [json.loads(line) for line in open(sys.argv[1])]; sealed = [event for event in events if event["event"] == "sibling_sealed"]; blocked = [event for event in events if event["event"] == "blocked_before_ledger_seal"]; assert len(sealed) == 1, sealed; assert len(blocked) == 1, blocked; assert sealed[0]["group_id"] == blocked[0]["group_id"], (sealed, blocked); assert sealed[0]["generation_index"] != blocked[0]["generation_index"], (sealed, blocked); print(sealed[0]["group_id"])' \
    "$PHASE1_EVENTS")
uv run --directory "$PROJECT_ROOT" --no-sync python -c \
    'import sys, torch; state = torch.load(sys.argv[1], weights_only=True); group_id = sys.argv[2]; groups = [group for group in state["groups"] if group["group_id"] == group_id]; assert len(groups) == 1, state; group = groups[0]; statuses = [sibling["attempts"][-1]["status"] for sibling in group["siblings"]]; assert group["recovery_granularity"] == "sibling", group; assert sorted(statuses) == ["dispatched", "sealed"], statuses' \
    "$STEP1/rollout_recovery.pt" "$PARTIAL_GROUP_ID"

echo "=== Phase 2: reuse the sealed sibling and regenerate only its peer ==="
SC_TEST_ENTRYPOINT="$RECOVERY_HOOK" \
SC_SIBLING_RECOVERY_TEST_EVENTS="$PHASE2_EVENTS" \
RUN_CONVERGENCE_CHECKS=0 bash "$BASE_TEST" \
    "${COMMON_OVERRIDES[@]}"
cp "$BASE_RUN_LOG" "$PHASE2_LOG"

grep -q "Native TQ checkpoint restored and validated" "$PHASE2_LOG"
grep -q "Loaded .* unfinished rollout group(s)" "$PHASE2_LOG"
test -d "$CHECKPOINT_DIR/step_2/data_plane"
test -f "$CHECKPOINT_DIR/step_2/rollout_recovery.pt"
uv run --directory "$PROJECT_ROOT" --no-sync python -c \
    'import json, sys, torch, uuid; phase1 = torch.load(sys.argv[1], weights_only=True); group_id = sys.argv[2]; events = [json.loads(line) for line in open(sys.argv[3])]; group = next(group for group in phase1["groups"] if group["group_id"] == group_id); old_attempts = ["{}_g{}_a{}".format(group_id, i, uuid.UUID(bytes=sibling["attempts"][-1]["attempt_uuid"]).hex) for i, sibling in enumerate(group["siblings"])]; sealed_index = next(i for i, sibling in enumerate(group["siblings"]) if sibling["attempts"][-1]["status"] == "sealed"); unfinished_index = 1 - sealed_index; dispatches = [event for event in events if event["event"] == "dispatch" and event["group_id"] == group_id]; assert len(dispatches) == 1, dispatches; dispatch = dispatches[0]; assert dispatch["generation_indices"] == [unfinished_index], dispatch; assert dispatch["rollout_ids"][sealed_index] == old_attempts[sealed_index], dispatch; assert dispatch["rollout_ids"][unfinished_index] != old_attempts[unfinished_index], dispatch' \
    "$STEP1/rollout_recovery.pt" "$PARTIAL_GROUP_ID" "$PHASE2_EVENTS"
uv run --directory "$PROJECT_ROOT" --no-sync python -c \
    'import sys, torch; state = torch.load(sys.argv[1], weights_only=True); group_id = sys.argv[2]; assert group_id not in {group["group_id"] for group in state["groups"]}, state' \
    "$CHECKPOINT_DIR/step_2/rollout_recovery.pt" "$PARTIAL_GROUP_ID"

echo "Sibling-level token-capture recovery functional test passed."
