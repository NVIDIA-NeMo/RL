#!/bin/bash
# Two-process NeMo-Gym test for pre-step TQ + partial-sibling recovery.

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
SELECTION_FILE=$TEST_DIR/selected_snapshot
EXPECTED_STATE=$TEST_DIR/expected_rollout_recovery.pt
PHASE1_PID=""

MODEL_NAME=${SC_TQ_RECOVERY_MODEL_NAME:-Qwen/Qwen3-0.6B}
NUM_PROMPTS=${SC_TQ_RECOVERY_NUM_PROMPTS:-4}
NUM_GENERATIONS=${SC_TQ_RECOVERY_NUM_GENERATIONS:-4}
MIN_SEALED=${SC_TQ_RECOVERY_MIN_SEALED:-1}
MAX_INFLIGHT_PROMPTS=${SC_TQ_RECOVERY_MAX_INFLIGHT_PROMPTS:-8}
MAX_BUFFERED_ROLLOUTS=${SC_TQ_RECOVERY_MAX_BUFFERED_ROLLOUTS:-8}
MAX_NUM_STEPS=${SC_TQ_RECOVERY_MAX_NUM_STEPS:-6}
SNAPSHOT_TIMEOUT_S=${SC_TQ_RECOVERY_SNAPSHOT_TIMEOUT_S:-1800}
TRAIN_GLOBAL_BATCH_SIZE=$((NUM_PROMPTS * NUM_GENERATIONS))

if (( NUM_PROMPTS < 1 || NUM_GENERATIONS < 2 )); then
    echo "NUM_PROMPTS must be positive and NUM_GENERATIONS must be at least 2"
    exit 2
fi
if (( MIN_SEALED < 1 || MIN_SEALED >= NUM_GENERATIONS )); then
    echo "MIN_SEALED must be between 1 and NUM_GENERATIONS - 1"
    exit 2
fi
if (( MAX_BUFFERED_ROLLOUTS < (2 * NUM_PROMPTS - 1) )); then
    echo "MAX_BUFFERED_ROLLOUTS must be at least 2 * NUM_PROMPTS - 1"
    exit 2
fi

rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

stop_phase1() {
    if [[ -z "$PHASE1_PID" ]]; then
        return
    fi
    # Use an abrupt whole-process-group failure. A graceful delay could let
    # training publish a newer trainer checkpoint after the selected cut.
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
    # The inherited Gym recipe tracks val:accuracy, but SC has no validation
    # loop. This test exercises checkpoint mechanics, not top-k retention.
    checkpointing.metric_name=null
    ++data_plane.checkpointing_enabled=true
    async_rl.sampler.name=windowed
    '~async_rl.sampler.max_lookahead_versions'
    '+async_rl.sampler.max_staleness_versions=1'
    async_rl.min_groups_for_streaming_train="$NUM_PROMPTS"
    async_rl.max_inflight_prompts="$MAX_INFLIGHT_PROMPTS"
    async_rl.max_buffered_rollouts="$MAX_BUFFERED_ROLLOUTS"
    ++rollout_checkpointing.interval_s=0.25
    ++rollout_checkpointing.keep_latest_k=128
    policy.model_name="$MODEL_NAME"
    grpo.num_prompts_per_step="$NUM_PROMPTS"
    grpo.num_generations_per_prompt="$NUM_GENERATIONS"
    policy.train_global_batch_size="$TRAIN_GLOBAL_BATCH_SIZE"
    grpo.max_num_steps="$MAX_NUM_STEPS"
)

echo "=== Phase 1: crash after a committed partial-sibling snapshot ==="
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
    "$CHECKPOINT_DIR/bootstrap/rollout_snapshots" \
    "$SELECTION_FILE" \
    "$PHASE1_PID" \
    "$PHASE1_LOG" \
    "$NUM_GENERATIONS" \
    "$MIN_SEALED" \
    "$SNAPSHOT_TIMEOUT_S" <<'PY'
import os
import sys
import time
from pathlib import Path

import torch

root = Path(sys.argv[1])
selection = Path(sys.argv[2])
phase_pid = int(sys.argv[3])
phase_log = Path(sys.argv[4])
expected_generations = int(sys.argv[5])
min_sealed = int(sys.argv[6])
deadline = time.monotonic() + float(sys.argv[7])


def phase_log_tail() -> str:
    if not phase_log.is_file():
        return f"phase-one log was not created at {phase_log}"
    lines = phase_log.read_text(errors="replace").splitlines()
    return "\n".join(lines[-40:])

while time.monotonic() < deadline:
    for snapshot in sorted(root.glob("snapshot_*"), reverse=True):
        ledger_path = snapshot / "rollout_recovery.pt"
        if not (snapshot / "COMMITTED").is_file() or not ledger_path.is_file():
            continue
        state = torch.load(ledger_path, weights_only=False)
        partial = []
        for group in state["groups"]:
            statuses = [
                sibling["attempts"][-1]["status"]
                for sibling in group["siblings"]
            ]
            if len(statuses) != expected_generations:
                raise RuntimeError(
                    f"group {group['group_id']} has {len(statuses)} siblings; "
                    f"expected {expected_generations}"
                )
            sealed = statuses.count("sealed")
            if min_sealed <= sealed < expected_generations:
                partial.append((group["group_id"], sealed, expected_generations))
        if partial:
            selection.write_text(snapshot.name + "\n")
            print(f"selected {snapshot.name}: partial_groups={partial}", flush=True)
            raise SystemExit(0)
    try:
        os.kill(phase_pid, 0)
    except ProcessLookupError as error:
        raise RuntimeError(
            "phase one exited before producing a partial-sibling snapshot; "
            f"last log lines from {phase_log}:\n{phase_log_tail()}"
        ) from error
    time.sleep(0.25)

raise TimeoutError("no committed partial-sibling snapshot was produced")
PY

stop_phase1
SNAPSHOT_NAME=$(tr -d '\n' < "$SELECTION_FILE")
SNAPSHOT_ROOT=$CHECKPOINT_DIR/bootstrap/rollout_snapshots
SNAPSHOT_DIR=$SNAPSHOT_ROOT/$SNAPSHOT_NAME

# resolve_latest_snapshot scans immutable directories rather than trusting only
# LATEST. Retain exactly the cut selected by the fault-injection predicate.
for candidate in "$SNAPSHOT_ROOT"/snapshot_*; do
    if [[ -d "$candidate" && "$(basename "$candidate")" != "$SNAPSHOT_NAME" ]]; then
        rm -rf "$candidate"
    fi
done
printf '%s\n' "$SNAPSHOT_NAME" > "$SNAPSHOT_ROOT/LATEST"

test -f "$CHECKPOINT_DIR/bootstrap/manifest.json"
test -f "$SNAPSHOT_DIR/COMMITTED"
test -d "$SNAPSHOT_DIR/data_plane"
test -f "$SNAPSHOT_DIR/replay_buffer_metadata.pt"
test -f "$SNAPSHOT_DIR/rollout_recovery.pt"
test -f "$SNAPSHOT_DIR/train_dataloader.pt"
test ! -d "$SNAPSHOT_DIR/policy"
cp "$SNAPSHOT_DIR/rollout_recovery.pt" "$EXPECTED_STATE"

echo "=== Phase 2: restore sealed siblings and redispatch only missing ones ==="
SC_GYM_EXP_DIR="$PHASE2_DIR" \
SC_GYM_RUN_LOG="$PHASE2_LOG" \
SC_GYM_CHECKPOINT_DIR="$CHECKPOINT_DIR" \
SC_GYM_DATA_DIR="$DATA_DIR" \
SC_GYM_KEEP_CHECKPOINTS=1 \
SC_GYM_RUN_CONVERGENCE_CHECKS=1 \
bash "$BASE_TEST" "${COMMON_OVERRIDES[@]}"

grep -Fq "Selected rollout recovery snapshot: $SNAPSHOT_DIR" "$PHASE2_LOG"
grep -q "Native TQ checkpoint restored and validated" "$PHASE2_LOG"
grep -q "Rollout recovery completed" "$PHASE2_LOG"
grep -q "train step $MAX_NUM_STEPS/$MAX_NUM_STEPS" "$PHASE2_LOG"

uv run --no-sync python - "$EXPECTED_STATE" "$PHASE2_LOG" <<'PY'
import re
import sys
from pathlib import Path

import torch

state = torch.load(sys.argv[1], weights_only=False)
log = Path(sys.argv[2]).read_text()
expected = {}
for group in state["groups"]:
    if group["status"] not in {"generating", "ready_to_finalize"}:
        continue
    total = len(group["siblings"])
    sealed = sum(
        sibling["attempts"][-1]["status"] == "sealed"
        for sibling in group["siblings"]
    )
    expected[group["group_id"]] = (sealed, total - sealed)

pattern = re.compile(
    r"rollout recovery finalized group: group=(\S+) "
    r"reused=(\d+) redispatched=(\d+)"
)
observed = {
    group_id: (int(reused), int(redispatched))
    for group_id, reused, redispatched in pattern.findall(log)
}
assert observed == expected, f"recovery mismatch: {observed=} {expected=}"
assert any(reused > 0 and redispatched > 0 for reused, redispatched in observed.values())
print(f"validated partial-sibling reuse: {observed}")
PY

echo "Partial-sibling TQ recovery functional test passed."
