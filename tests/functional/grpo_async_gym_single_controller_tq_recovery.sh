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
PARTIAL_SNAPSHOT_FILE=$TEST_DIR/partial_snapshot_name
EXPECTED_RECOVERY_STATE=$TEST_DIR/expected_rollout_recovery.pt
PHASE1_PID=""

# Keep the L1 defaults small while allowing a larger cluster stress run without
# maintaining a second copy of this recovery harness.
MODEL_NAME=${SC_TQ_RECOVERY_MODEL_NAME:-Qwen/Qwen3-0.6B}
NUM_PROMPTS=${SC_TQ_RECOVERY_NUM_PROMPTS:-4}
NUM_GENERATIONS=${SC_TQ_RECOVERY_NUM_GENERATIONS:-4}
MIN_SEALED=${SC_TQ_RECOVERY_MIN_SEALED:-1}
MAX_INFLIGHT_PROMPTS=${SC_TQ_RECOVERY_MAX_INFLIGHT_PROMPTS:-8}
MAX_BUFFERED_ROLLOUTS=${SC_TQ_RECOVERY_MAX_BUFFERED_ROLLOUTS:-8}
MAX_NUM_STEPS=${SC_TQ_RECOVERY_MAX_NUM_STEPS:-10}
TRAINER_SAVE_PERIOD=${SC_TQ_RECOVERY_SAVE_PERIOD:-5}
PARTIAL_SNAPSHOT_TIMEOUT_S=${SC_TQ_RECOVERY_SNAPSHOT_TIMEOUT_S:-1800}
SAMPLER_NAME=${SC_TQ_RECOVERY_SAMPLER_NAME:-windowed}
IN_ORDER_LOOKAHEAD=${SC_TQ_RECOVERY_IN_ORDER_LOOKAHEAD:-1}
TRAIN_GLOBAL_BATCH_SIZE=$((NUM_PROMPTS * NUM_GENERATIONS))

case "$SAMPLER_NAME" in
    windowed)
        SAMPLER_OVERRIDES=(
            async_rl.sampler.name=windowed
            '~async_rl.sampler.max_lookahead_versions'
            '+async_rl.sampler.max_staleness_versions=1'
        )
        ;;
    in_order)
        SAMPLER_OVERRIDES=(
            async_rl.sampler.name=in_order
            async_rl.sampler.max_lookahead_versions="$IN_ORDER_LOOKAHEAD"
        )
        ;;
    *)
        echo "SC_TQ_RECOVERY_SAMPLER_NAME must be windowed or in_order"
        exit 2
        ;;
esac

if (( NUM_PROMPTS < 1 || NUM_GENERATIONS < 2 )); then
    echo "NUM_PROMPTS must be positive and NUM_GENERATIONS must be at least 2"
    exit 2
fi
if (( MIN_SEALED < 1 || MIN_SEALED >= NUM_GENERATIONS )); then
    echo "MIN_SEALED must be between 1 and NUM_GENERATIONS - 1"
    exit 2
fi
if (( MAX_BUFFERED_ROLLOUTS < NUM_PROMPTS )); then
    echo "MAX_BUFFERED_ROLLOUTS must hold at least one prompt batch"
    exit 2
fi
if [[ "$SAMPLER_NAME" == "in_order" ]] &&
    (( MAX_BUFFERED_ROLLOUTS < NUM_PROMPTS * (IN_ORDER_LOOKAHEAD + 1) )); then
    echo "MAX_BUFFERED_ROLLOUTS is too small for the InOrder lookahead window"
    exit 2
fi
if (( TRAINER_SAVE_PERIOD < 1 || TRAINER_SAVE_PERIOD >= MAX_NUM_STEPS )); then
    echo "TRAINER_SAVE_PERIOD must be positive and smaller than MAX_NUM_STEPS"
    exit 2
fi

echo "Recovery test configuration: model=$MODEL_NAME prompts=$NUM_PROMPTS generations=$NUM_GENERATIONS min_sealed=$MIN_SEALED max_inflight=$MAX_INFLIGHT_PROMPTS max_buffered=$MAX_BUFFERED_ROLLOUTS steps=$MAX_NUM_STEPS trainer_save_period=$TRAINER_SAVE_PERIOD snapshot_timeout_s=$PARTIAL_SNAPSHOT_TIMEOUT_S sampler=$SAMPLER_NAME in_order_lookahead=$IN_ORDER_LOOKAHEAD"

rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

stop_phase1() {
    if [[ -z "$PHASE1_PID" ]]; then
        return
    fi

    kill -TERM -- "-$PHASE1_PID" 2>/dev/null || true
    for _ in {1..20}; do
        if ! kill -0 -- "-$PHASE1_PID" 2>/dev/null; then
            break
        fi
        sleep 0.5
    done
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
    checkpointing.save_period="$TRAINER_SAVE_PERIOD"
    ++data_plane.checkpointing_enabled=true
    "${SAMPLER_OVERRIDES[@]}"
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

echo "=== Phase 1: crash after a durable partial-sibling rollout cut ==="
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

# Poll only atomically committed snapshots. The selected snapshot must contain
# at least one group with both reusable SEALED siblings and missing siblings.
uv run --no-sync python - \
    "$CHECKPOINT_DIR/bootstrap/rollout_snapshots" \
    "$PARTIAL_SNAPSHOT_FILE" \
    "$PHASE1_PID" \
    "$NUM_GENERATIONS" \
    "$MIN_SEALED" \
    "$PARTIAL_SNAPSHOT_TIMEOUT_S" <<'PY'
import os
import sys
import time
from pathlib import Path

import torch

snapshot_root = Path(sys.argv[1])
selection_path = Path(sys.argv[2])
phase1_pid = int(sys.argv[3])
expected_generations = int(sys.argv[4])
min_sealed = int(sys.argv[5])
timeout_s = float(sys.argv[6])
deadline = time.monotonic() + timeout_s
inspected: set[str] = set()

while time.monotonic() < deadline:
    candidates = sorted(snapshot_root.glob("snapshot_*"), reverse=True)
    for snapshot in candidates:
        if snapshot.name in inspected:
            continue
        if not (snapshot / "COMMITTED").is_file():
            continue
        ledger_path = snapshot / "rollout_recovery.pt"
        if not ledger_path.is_file():
            continue

        state = torch.load(ledger_path, weights_only=False)
        inspected.add(snapshot.name)
        partial_groups: list[tuple[str, int, int]] = []
        for group in state["groups"]:
            statuses = [
                sibling["attempts"][-1]["status"]
                for sibling in group["siblings"]
            ]
            if len(statuses) != expected_generations:
                raise RuntimeError(
                    f"group {group['group_id']} has {len(statuses)} generations; "
                    f"expected {expected_generations}"
                )
            sealed = statuses.count("sealed")
            if min_sealed <= sealed < len(statuses):
                partial_groups.append((group["group_id"], sealed, len(statuses)))

        if partial_groups:
            selection_path.write_text(snapshot.name + "\n")
            print(
                "selected partial rollout snapshot:",
                snapshot.name,
                "groups=",
                partial_groups,
                flush=True,
            )
            raise SystemExit(0)
    try:
        os.kill(phase1_pid, 0)
    except ProcessLookupError as error:
        raise RuntimeError(
            "phase one exited before producing a partial rollout snapshot"
        ) from error
    time.sleep(0.25)

raise TimeoutError(
    "no committed snapshot satisfied the partial rollout threshold "
    f"within {timeout_s:g}s; "
    f"required_sealed={min_sealed}/{expected_generations}, "
    f"inspected={sorted(inspected)}"
)
PY

# Terminate the whole phase-one process group to model a scheduler/process
# failure, then make the selected immutable snapshot the newest candidate.
stop_phase1
SNAPSHOT_NAME=$(tr -d '\n' < "$PARTIAL_SNAPSHOT_FILE")
SNAPSHOT_ROOT=$CHECKPOINT_DIR/bootstrap/rollout_snapshots
# Validate the pointer published by commit_snapshot before deliberately
# repointing it to the selected partial cut used by this recovery test.
test -f "$SNAPSHOT_ROOT/LATEST"
PRODUCTION_LATEST=$(tr -d '\n' < "$SNAPSHOT_ROOT/LATEST")
test -f "$SNAPSHOT_ROOT/$PRODUCTION_LATEST/COMMITTED"
for candidate in "$SNAPSHOT_ROOT"/snapshot_*; do
    if [[ -d "$candidate" && "$(basename "$candidate")" != "$SNAPSHOT_NAME" ]]; then
        rm -rf "$candidate"
    fi
done
printf '%s\n' "$SNAPSHOT_NAME" > "$SNAPSHOT_ROOT/LATEST"

test ! -d "$CHECKPOINT_DIR/step_1"
test -f "$CHECKPOINT_DIR/bootstrap/manifest.json"
SNAPSHOT_DIR=$CHECKPOINT_DIR/bootstrap/rollout_snapshots/$SNAPSHOT_NAME
test -f "$SNAPSHOT_DIR/COMMITTED"
test -d "$SNAPSHOT_DIR/data_plane"
test -f "$SNAPSHOT_DIR/replay_buffer_metadata.pt"
test -f "$SNAPSHOT_DIR/rollout_recovery.pt"
test -f "$SNAPSHOT_DIR/train_dataloader.pt"
test ! -d "$SNAPSHOT_DIR/policy"
uv run --no-sync python - \
    "$SNAPSHOT_DIR/rollout_recovery.pt" \
    "$MAX_BUFFERED_ROLLOUTS" \
    "$NUM_GENERATIONS" \
    "$MIN_SEALED" <<'PY'
import sys

import torch

state = torch.load(sys.argv[1], weights_only=False)
max_buffered_rollouts = int(sys.argv[2])
expected_generations = int(sys.argv[3])
min_sealed = int(sys.argv[4])
groups = state["groups"]
partial_groups = []
for group in groups:
    assert len(group["siblings"]) == expected_generations, group
    sealed = sum(
        sibling["attempts"][-1]["status"] == "sealed"
        for sibling in group["siblings"]
    )
    if min_sealed <= sealed < len(group["siblings"]):
        partial_groups.append((group["group_id"], sealed, len(group["siblings"])))

assert partial_groups, state
assert len(groups) <= max_buffered_rollouts, state
print("saved partial rollout groups=", partial_groups)
PY
grep -Fq "rollout checkpoint save completed: $SNAPSHOT_DIR " "$PHASE1_LOG"
# Phase two creates a durable trainer checkpoint and correctly prunes the
# obsolete bootstrap snapshots on shutdown. Preserve only the small ledger
# metadata needed for the post-run exact reuse/redispatch assertion.
cp "$SNAPSHOT_DIR/rollout_recovery.pt" "$EXPECTED_RECOVERY_STATE"

echo "=== Phase 2: restore in a fresh process and verify convergence ==="
SC_GYM_EXP_DIR="$PHASE2_DIR" \
SC_GYM_RUN_LOG="$PHASE2_LOG" \
SC_GYM_CHECKPOINT_DIR="$CHECKPOINT_DIR" \
SC_GYM_DATA_DIR="$DATA_DIR" \
SC_GYM_KEEP_CHECKPOINTS=1 \
SC_GYM_RUN_CONVERGENCE_CHECKS=1 \
bash "$BASE_TEST" "${COMMON_OVERRIDES[@]}"

grep -Fq "Selected rollout recovery snapshot: $SNAPSHOT_DIR" "$PHASE2_LOG"
grep -q "Native TQ checkpoint restored and validated" "$PHASE2_LOG"
grep -q "Restored rollout recovery ledger" "$PHASE2_LOG"
if [[ "$SAMPLER_NAME" == "in_order" ]]; then
    grep -Eq "Restored sampler dispatch state: sampler=InOrderSampler, .*newest_target=" "$PHASE2_LOG"
fi
grep -Eq "rollout recovery finalized group: .*reused=[1-9][0-9]* redispatched=[1-9][0-9]*" "$PHASE2_LOG"
grep -q "rollout recovery replay completed" "$PHASE2_LOG"
grep -q "train step $MAX_NUM_STEPS/$MAX_NUM_STEPS" "$PHASE2_LOG"
test -d "$CHECKPOINT_DIR/step_$MAX_NUM_STEPS/data_plane"
# Exercise the post-step snapshot path against real TQ. The trainer checkpoint
# at TRAINER_SAVE_PERIOD anchors a newer rollout-only cut while training
# continues toward MAX_NUM_STEPS.
POST_STEP_SNAPSHOT_ROOT=$CHECKPOINT_DIR/step_$TRAINER_SAVE_PERIOD/rollout_snapshots
test -f "$POST_STEP_SNAPSHOT_ROOT/LATEST"
POST_STEP_SNAPSHOT_NAME=$(tr -d '\n' < "$POST_STEP_SNAPSHOT_ROOT/LATEST")
POST_STEP_SNAPSHOT_DIR=$POST_STEP_SNAPSHOT_ROOT/$POST_STEP_SNAPSHOT_NAME
test -f "$POST_STEP_SNAPSHOT_DIR/COMMITTED"
test -d "$POST_STEP_SNAPSHOT_DIR/data_plane"
test -f "$POST_STEP_SNAPSHOT_DIR/replay_buffer_metadata.pt"
test -f "$POST_STEP_SNAPSHOT_DIR/rollout_recovery.pt"
test -f "$POST_STEP_SNAPSHOT_DIR/train_dataloader.pt"
test -f "$POST_STEP_SNAPSHOT_DIR/manifest.json"
grep -Fq "rollout checkpoint save completed: $POST_STEP_SNAPSHOT_DIR " "$PHASE2_LOG"
uv run --no-sync python - \
    "$POST_STEP_SNAPSHOT_DIR/manifest.json" \
    "$TRAINER_SAVE_PERIOD" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
expected_step = int(sys.argv[2])
assert manifest["base_train_step"] == expected_step, manifest
assert manifest["trainer_version"] == expected_step, manifest
assert manifest["bootstrap_fingerprint"] is None, manifest
print("validated post-step rollout snapshot:", sys.argv[1])
PY
# Once a durable trainer checkpoint exists, bootstrap snapshots are obsolete
# and must be pruned rather than accumulating full TQ snapshots indefinitely.
test ! -d "$CHECKPOINT_DIR/bootstrap/rollout_snapshots"

# Every SEALED sibling in the selected snapshot must be reported as reused;
# every other sibling must be redispatched. This detects accidental full-group
# regeneration even when training still converges.
uv run --no-sync python - \
    "$EXPECTED_RECOVERY_STATE" \
    "$PHASE2_LOG" \
    "$MIN_SEALED" <<'PY'
import re
import sys
from pathlib import Path

import torch

state = torch.load(sys.argv[1], weights_only=False)
log = Path(sys.argv[2]).read_text()
min_sealed = int(sys.argv[3])

expected = {}
for group in state["groups"]:
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

assert set(observed) == set(expected), (
    f"recovery log groups differ from snapshot: "
    f"missing={sorted(set(expected) - set(observed))}, "
    f"unexpected={sorted(set(observed) - set(expected))}"
)
assert observed == expected, f"recovery counts differ: {observed=} {expected=}"
assert any(
    reused >= min_sealed and redispatched > 0
    for reused, redispatched in observed.values()
), observed

print(
    "validated recovery reuse:",
    f"groups={len(observed)}",
    f"reused={sum(value[0] for value in observed.values())}",
    f"redispatched={sum(value[1] for value in observed.values())}",
)
PY

echo "Pre-step partial-rollout recovery and convergence test passed."
