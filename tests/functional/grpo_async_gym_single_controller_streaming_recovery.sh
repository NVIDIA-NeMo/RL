#!/bin/bash
# Two-process NeMo-Gym test for a periodic snapshot taken during a streamed
# optimizer step. The snapshot intentionally omits uncommitted gradients and
# makes every already-claimed group replayable from its durable trainer anchor.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BASE_TEST=$SCRIPT_DIR/grpo_async_gym_single_controller.sh
TEST_DIR=$SCRIPT_DIR/grpo_async_gym_single_controller_streaming_recovery
CHECKPOINT_DIR=$TEST_DIR/checkpoints
DATA_DIR=$TEST_DIR/data
PHASE1_DIR=$TEST_DIR/phase1
PHASE2_DIR=$TEST_DIR/phase2
PHASE1_LOG=$PHASE1_DIR/run.log
PHASE2_LOG=$PHASE2_DIR/run.log
SELECTION_FILE=$TEST_DIR/selected_snapshot
PHASE1_PID=""

NUM_PROMPTS=${SC_TQ_STREAMING_RECOVERY_NUM_PROMPTS:-8}
NUM_GENERATIONS=${SC_TQ_STREAMING_RECOVERY_NUM_GENERATIONS:-2}
MIN_STREAMING_GROUPS=${SC_TQ_STREAMING_RECOVERY_MIN_GROUPS:-2}
CLAIMED_GROUPS=${SC_TQ_STREAMING_RECOVERY_CLAIMED_GROUPS:-2}
MAX_INFLIGHT_PROMPTS=${SC_TQ_STREAMING_RECOVERY_MAX_INFLIGHT_PROMPTS:-2}
MAX_BUFFERED_ROLLOUTS=${SC_TQ_STREAMING_RECOVERY_MAX_BUFFERED_ROLLOUTS:-16}
MAX_NUM_STEPS=${SC_TQ_STREAMING_RECOVERY_MAX_NUM_STEPS:-3}
SNAPSHOT_INTERVAL_S=${SC_TQ_STREAMING_RECOVERY_INTERVAL_S:-0.05}
SNAPSHOT_TIMEOUT_S=${SC_TQ_STREAMING_RECOVERY_SNAPSHOT_TIMEOUT_S:-2400}
TRAIN_GLOBAL_BATCH_SIZE=$((NUM_PROMPTS * NUM_GENERATIONS))
REQUIRED_RECOVERY_CAPACITY=$((NUM_PROMPTS + MIN_STREAMING_GROUPS - 1))

if (( NUM_PROMPTS < 2 || NUM_GENERATIONS < 1 )); then
    echo "NUM_PROMPTS must be at least 2 and NUM_GENERATIONS must be positive"
    exit 2
fi
if (( MIN_STREAMING_GROUPS < 1 || MIN_STREAMING_GROUPS >= NUM_PROMPTS )); then
    echo "MIN_STREAMING_GROUPS must be between 1 and NUM_PROMPTS - 1"
    exit 2
fi
if (( CLAIMED_GROUPS < MIN_STREAMING_GROUPS || CLAIMED_GROUPS >= NUM_PROMPTS )); then
    echo "CLAIMED_GROUPS must be between MIN_STREAMING_GROUPS and NUM_PROMPTS - 1"
    exit 2
fi
if (( CLAIMED_GROUPS % MIN_STREAMING_GROUPS != 0 )); then
    echo "CLAIMED_GROUPS must be a multiple of MIN_STREAMING_GROUPS"
    exit 2
fi
if (( MAX_NUM_STEPS < 2 )); then
    echo "MAX_NUM_STEPS must be at least 2 so step_1 can anchor the snapshot"
    exit 2
fi
if (( MAX_BUFFERED_ROLLOUTS < REQUIRED_RECOVERY_CAPACITY )); then
    echo "MAX_BUFFERED_ROLLOUTS must be at least $REQUIRED_RECOVERY_CAPACITY"
    exit 2
fi

rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

stop_phase1() {
    if [[ -z "$PHASE1_PID" ]]; then
        return
    fi
    # Use an abrupt whole-process-group failure. A graceful delay could let
    # step 2 commit after the exact streamed-step cut selected above.
    kill -KILL -- "-$PHASE1_PID" 2>/dev/null || true
    wait "$PHASE1_PID" 2>/dev/null || true
    PHASE1_PID=""
}

cleanup() {
    stop_phase1
    rm -rf "$CHECKPOINT_DIR"
}
trap cleanup EXIT

COMMON_OVERRIDES=(
    ++token_capture.enabled=true
    checkpointing.enabled=true
    checkpointing.checkpoint_dir="$CHECKPOINT_DIR"
    checkpointing.save_period=1
    ++data_plane.checkpointing_enabled=true
    async_rl.sampler.name=in_order
    async_rl.sampler.max_lookahead_versions=0
    async_rl.min_groups_for_streaming_train="$MIN_STREAMING_GROUPS"
    async_rl.max_inflight_prompts="$MAX_INFLIGHT_PROMPTS"
    async_rl.max_buffered_rollouts="$MAX_BUFFERED_ROLLOUTS"
    ++rollout_checkpointing.interval_s="$SNAPSHOT_INTERVAL_S"
    ++rollout_checkpointing.keep_latest_k=256
    ++rollout_checkpointing.restore_mode=latest
    grpo.num_prompts_per_step="$NUM_PROMPTS"
    grpo.num_generations_per_prompt="$NUM_GENERATIONS"
    policy.train_global_batch_size="$TRAIN_GLOBAL_BATCH_SIZE"
    grpo.max_num_steps="$MAX_NUM_STEPS"
)

echo "=== Phase 1: crash with $CLAIMED_GROUPS/$NUM_PROMPTS streamed groups claimed ==="
command -v setsid >/dev/null
setsid env \
    SC_GYM_EXP_DIR="$PHASE1_DIR" \
    SC_GYM_RUN_LOG="$PHASE1_LOG" \
    SC_GYM_CHECKPOINT_DIR="$CHECKPOINT_DIR" \
    SC_GYM_DATA_DIR="$DATA_DIR" \
    SC_GYM_KEEP_CHECKPOINTS=1 \
    SC_GYM_RUN_CONVERGENCE_CHECKS=0 \
    bash "$BASE_TEST" "${COMMON_OVERRIDES[@]}" &
PHASE1_PID=$!

uv run --no-sync python - \
    "$CHECKPOINT_DIR/step_1/rollout_snapshots" \
    "$SELECTION_FILE" \
    "$PHASE1_PID" \
    "$CLAIMED_GROUPS" \
    "$SNAPSHOT_TIMEOUT_S" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
selection = Path(sys.argv[2])
phase_pid = int(sys.argv[3])
expected_claimed = int(sys.argv[4])
deadline = time.monotonic() + float(sys.argv[5])

while time.monotonic() < deadline:
    for snapshot in sorted(root.glob("snapshot_*"), reverse=True):
        manifest_path = snapshot / "manifest.json"
        if not (snapshot / "COMMITTED").is_file() or not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["base_train_step"] == 1
            and manifest["trainer_version"] == 1
            and manifest["rolled_back_train_group_count"] == expected_claimed
        ):
            selection.write_text(snapshot.name + "\n")
            print(
                f"selected {snapshot.name}: "
                f"rolled_back_train_group_count={expected_claimed}",
                flush=True,
            )
            raise SystemExit(0)
    try:
        os.kill(phase_pid, 0)
    except ProcessLookupError as error:
        raise RuntimeError(
            "phase one exited before producing the requested streamed-step cut"
        ) from error
    time.sleep(0.1)

raise TimeoutError(
    f"no committed snapshot captured {expected_claimed} claimed groups"
)
PY

stop_phase1
SNAPSHOT_NAME=$(tr -d '\n' < "$SELECTION_FILE")
SNAPSHOT_ROOT=$CHECKPOINT_DIR/step_1/rollout_snapshots
SNAPSHOT_DIR=$SNAPSHOT_ROOT/$SNAPSHOT_NAME

# The resolver scans immutable directories, not only LATEST. Retain exactly
# the fault-injection cut so phase two cannot silently choose a newer snapshot.
for candidate in "$SNAPSHOT_ROOT"/snapshot_*; do
    if [[ -d "$candidate" && "$(basename "$candidate")" != "$SNAPSHOT_NAME" ]]; then
        rm -rf "$candidate"
    fi
done
printf '%s\n' "$SNAPSHOT_NAME" > "$SNAPSHOT_ROOT/LATEST"

test -f "$CHECKPOINT_DIR/step_1/training_info.json"
test -d "$CHECKPOINT_DIR/step_1/policy"
test -f "$SNAPSHOT_DIR/COMMITTED"
test -d "$SNAPSHOT_DIR/data_plane"
test -f "$SNAPSHOT_DIR/replay_buffer_metadata.pt"
test -f "$SNAPSHOT_DIR/rollout_recovery.pt"
test -f "$SNAPSHOT_DIR/train_dataloader.pt"
test ! -d "$SNAPSHOT_DIR/policy"

uv run --no-sync python - \
    "$SNAPSHOT_DIR/manifest.json" \
    "$SNAPSHOT_DIR/rollout_recovery.pt" \
    "$CLAIMED_GROUPS" <<'PY'
import json
import sys

import torch

manifest = json.load(open(sys.argv[1]))
lineage = torch.load(sys.argv[2], weights_only=False)
expected_claimed = int(sys.argv[3])

assert manifest["base_train_step"] == 1, manifest
assert manifest["trainer_version"] == 1, manifest
assert manifest["rolled_back_train_group_count"] == expected_claimed, manifest
assert lineage["open_train_step"] is None, lineage["open_train_step"]
assert sum(group["status"] == "finalized" for group in lineage["groups"]) >= expected_claimed
print(
    "validated restart-ready streamed-step snapshot: "
    f"claimed_groups={expected_claimed}"
)
PY

echo "=== Phase 2: replay the rolled-back groups and finish each step once ==="
SC_GYM_EXP_DIR="$PHASE2_DIR" \
SC_GYM_RUN_LOG="$PHASE2_LOG" \
SC_GYM_CHECKPOINT_DIR="$CHECKPOINT_DIR" \
SC_GYM_DATA_DIR="$DATA_DIR" \
SC_GYM_KEEP_CHECKPOINTS=1 \
SC_GYM_RUN_CONVERGENCE_CHECKS=0 \
bash "$BASE_TEST" "${COMMON_OVERRIDES[@]}"

grep -Fq "Selected rollout recovery snapshot: $SNAPSHOT_DIR" "$PHASE2_LOG"
grep -q "Native TQ checkpoint restored and validated" "$PHASE2_LOG"
grep -q "Native TQ replay inventory validated" "$PHASE2_LOG"
grep -Eq "Restored [1-9][0-9]* replay group" "$PHASE2_LOG"
grep -q "Restored sampler dispatch state" "$PHASE2_LOG"
grep -q "train step $MAX_NUM_STEPS/$MAX_NUM_STEPS" "$PHASE2_LOG"

uv run --no-sync python - \
    "$PHASE2_LOG" \
    "$CHECKPOINT_DIR/step_$MAX_NUM_STEPS/training_info.json" \
    "$MAX_NUM_STEPS" <<'PY'
import json
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text()
training_info_path = Path(sys.argv[2])
max_steps = int(sys.argv[3])

assert training_info_path.is_file(), training_info_path
training_info = json.loads(training_info_path.read_text())
assert training_info["current_step"] == max_steps, training_info
assert training_info["trainer_version"] == max_steps, training_info

# The durable anchor already contains step 1. Recovery must resume at step 2,
# replay its rolled-back groups, and never apply that optimizer step twice.
assert not re.search(r"train step 1/", log), "phase two repeated anchored step 1"
for step in range(2, max_steps + 1):
    matches = re.findall(rf"train step {step}/{max_steps}(?:\s|$)", log)
    assert len(matches) == 1, f"step {step} completed {len(matches)} times"
print(f"validated exactly-once resumed steps 2..{max_steps}")
PY

echo "Streamed-step periodic recovery functional test passed."
