#!/bin/bash
# Crash/restart coverage for a periodic cut taken during streamed GRPO train.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
BASE_TEST=$SCRIPT_DIR/grpo_async_gym_single_controller.sh
BASE_RUN_LOG=$SCRIPT_DIR/grpo_async_gym_single_controller/run.log
TEST_DIR=$SCRIPT_DIR/grpo_async_gym_single_controller_streaming_recovery
CHECKPOINT_DIR=$TEST_DIR/checkpoints
PHASE1_LOG=$TEST_DIR/phase1.log
PHASE2_LOG=$TEST_DIR/phase2.log
SELECTION_FILE=$TEST_DIR/selected_snapshot
PHASE1_PID=""

NUM_PROMPTS=${SC_STREAMING_RECOVERY_NUM_PROMPTS:-8}
NUM_GENERATIONS=${SC_STREAMING_RECOVERY_NUM_GENERATIONS:-2}
MIN_STREAMING_GROUPS=${SC_STREAMING_RECOVERY_MIN_GROUPS:-2}
CLAIMED_GROUPS=${SC_STREAMING_RECOVERY_CLAIMED_GROUPS:-2}
MAX_STEPS=${SC_STREAMING_RECOVERY_MAX_STEPS:-3}
SNAPSHOT_INTERVAL_S=${SC_STREAMING_RECOVERY_INTERVAL_S:-0.05}
SNAPSHOT_TIMEOUT_S=${SC_STREAMING_RECOVERY_TIMEOUT_S:-2400}
TRAIN_GLOBAL_BATCH_SIZE=$((NUM_PROMPTS * NUM_GENERATIONS))

if (( CLAIMED_GROUPS < MIN_STREAMING_GROUPS || CLAIMED_GROUPS >= NUM_PROMPTS )); then
    echo "CLAIMED_GROUPS must be in [MIN_STREAMING_GROUPS, NUM_PROMPTS)"
    exit 2
fi

rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

stop_phase1() {
    if [[ -z "$PHASE1_PID" ]]; then
        return
    fi
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
    checkpointing.enabled=true
    checkpointing.checkpoint_dir="$CHECKPOINT_DIR"
    checkpointing.save_period=1
    checkpointing.metric_name=null
    +checkpointing.save_data_plane=true
    ++token_capture.enabled=true
    ++rollout_recovery.default_granularity=sibling
    ++rollout_checkpointing.interval_s="$SNAPSHOT_INTERVAL_S"
    ++rollout_checkpointing.keep_latest_k=256
    ++rollout_checkpointing.restore_mode=latest
    async_rl.sampler.name=in_order
    async_rl.sampler.max_lookahead_versions=0
    async_rl.min_groups_for_streaming_train="$MIN_STREAMING_GROUPS"
    async_rl.max_inflight_prompts="$MIN_STREAMING_GROUPS"
    async_rl.max_buffered_rollouts=$((NUM_PROMPTS + MIN_STREAMING_GROUPS))
    grpo.num_prompts_per_step="$NUM_PROMPTS"
    grpo.num_generations_per_prompt="$NUM_GENERATIONS"
    grpo.max_num_steps="$MAX_STEPS"
    policy.train_global_batch_size="$TRAIN_GLOBAL_BATCH_SIZE"
)

echo "=== Phase 1: crash with $CLAIMED_GROUPS/$NUM_PROMPTS groups claimed ==="
command -v setsid >/dev/null
setsid env RUN_CONVERGENCE_CHECKS=0 bash "$BASE_TEST" \
    "${COMMON_OVERRIDES[@]}" &
PHASE1_PID=$!

uv run --directory "$PROJECT_ROOT" --no-sync python - \
    "$CHECKPOINT_DIR/step_1/rollout_snapshots" \
    "$SELECTION_FILE" \
    "$PHASE1_PID" \
    "$BASE_RUN_LOG" \
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
phase_log = Path(sys.argv[4])
expected_claimed = int(sys.argv[5])
deadline = time.monotonic() + float(sys.argv[6])

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
            raise SystemExit(0)
    try:
        os.kill(phase_pid, 0)
    except ProcessLookupError as error:
        tail = ""
        if phase_log.is_file():
            tail = "\n".join(phase_log.read_text(errors="replace").splitlines()[-40:])
        raise RuntimeError(
            "phase one exited before producing the requested streamed cut:\n" + tail
        ) from error
    time.sleep(0.1)
raise TimeoutError(f"no snapshot captured {expected_claimed} claimed groups")
PY

stop_phase1
cp "$BASE_RUN_LOG" "$PHASE1_LOG"
SNAPSHOT_NAME=$(tr -d '\n' < "$SELECTION_FILE")
SNAPSHOT_ROOT=$CHECKPOINT_DIR/step_1/rollout_snapshots
SNAPSHOT_DIR=$SNAPSHOT_ROOT/$SNAPSHOT_NAME

# Force restore to use the exact fault-injection cut.
for candidate in "$SNAPSHOT_ROOT"/snapshot_*; do
    if [[ -d "$candidate" && "$(basename "$candidate")" != "$SNAPSHOT_NAME" ]]; then
        rm -rf "$candidate"
    fi
done
printf '%s\n' "$SNAPSHOT_NAME" > "$SNAPSHOT_ROOT/LATEST"

uv run --directory "$PROJECT_ROOT" --no-sync python - \
    "$SNAPSHOT_DIR/manifest.json" \
    "$SNAPSHOT_DIR/replay_buffer_metadata.pt" \
    "$SNAPSHOT_DIR/rollout_recovery.pt" \
    "$CLAIMED_GROUPS" <<'PY'
import json
import sys

import torch

manifest = json.load(open(sys.argv[1]))
replay = torch.load(sys.argv[2], weights_only=False)
lineage = torch.load(sys.argv[3], weights_only=False)
expected_claimed = int(sys.argv[4])

assert manifest["rolled_back_train_group_count"] == expected_claimed, manifest
assert len(replay["groups"]) >= expected_claimed, replay
assert lineage["open_train_step"] is None, lineage
PY

echo "=== Phase 2: restore claimed rows and finish without duplicate steps ==="
RUN_CONVERGENCE_CHECKS=0 bash "$BASE_TEST" "${COMMON_OVERRIDES[@]}"
cp "$BASE_RUN_LOG" "$PHASE2_LOG"

grep -Fq "Selected rollout recovery snapshot: $SNAPSHOT_DIR" "$PHASE2_LOG"
grep -q "Native TQ checkpoint restored and validated" "$PHASE2_LOG"
grep -q "train step $MAX_STEPS/$MAX_STEPS" "$PHASE2_LOG"

uv run --directory "$PROJECT_ROOT" --no-sync python - \
    "$PHASE2_LOG" "$CHECKPOINT_DIR/step_$MAX_STEPS/training_info.json" \
    "$MAX_STEPS" <<'PY'
import json
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text()
training_info = json.loads(Path(sys.argv[2]).read_text())
max_steps = int(sys.argv[3])

assert training_info["current_step"] == max_steps, training_info
assert training_info["trainer_version"] == max_steps, training_info
assert not re.search(r"train step 1/", log), "restored run repeated anchor step 1"
for step in range(2, max_steps + 1):
    matches = re.findall(rf"train step {step}/{max_steps}(?:\s|$)", log)
    assert len(matches) == 1, (step, len(matches))
PY

echo "Streamed-step periodic recovery functional test passed."
