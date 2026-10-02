#!/bin/bash
# Two-process functional coverage for Gym turn-level recovery with a real,
# stateful Workplace Assistant resources server. By default this uses one Gym
# actor and runs on one Ray node. Set SC_TURN_RECOVERY_TEST_CONFIG to the
# two-replica config and SC_TURN_RECOVERY_EXPECTED_GYM_INSTANCES=2 to exercise
# cross-node replica ownership. Policy training and vLLM generation still use
# the small model.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
BASE_TEST=$SCRIPT_DIR/grpo_async_gym_single_controller.sh
BASE_RUN_LOG=$SCRIPT_DIR/grpo_async_gym_single_controller/run.log
TEST_CONFIG=${SC_TURN_RECOVERY_TEST_CONFIG:-$PROJECT_ROOT/examples/nemo_gym/grpo_qwen3_30ba3b_instruct.yaml}
RECOVERY_HOOK=$SCRIPT_DIR/_single_controller_turn_recovery_hook.py
TEST_DIR=$SCRIPT_DIR/grpo_async_gym_single_controller_turn_recovery
CHECKPOINT_DIR=$TEST_DIR/checkpoints
PHASE1_LOG=$TEST_DIR/phase1.log
PHASE2_LOG=$TEST_DIR/phase2.log
PHASE1_EVENTS=$TEST_DIR/phase1-events.jsonl
PHASE2_EVENTS=$TEST_DIR/phase2-events.jsonl
SELECTION_FILE=$TEST_DIR/selected-snapshot.json
# Bootstrap snapshots are pruned once step 1's trainer checkpoint lands, so the
# selector copies the chosen cut here before phase one can delete it.
SELECTED_BACKUP=$TEST_DIR/selected-snapshot
TEST_DATA=$TEST_DIR/test_data.jsonl
GYM_ROOT=$PROJECT_ROOT/3rdparty/Gym-workspace/Gym
SENTINEL_EVENT="NeMo RL checkpoint recovery sentinel"
PHASE1_PID=""

EXPECTED_GYM_INSTANCES=${SC_TURN_RECOVERY_EXPECTED_GYM_INSTANCES:-1}
NUM_PROMPTS=${SC_TURN_RECOVERY_NUM_PROMPTS:-4}
NUM_GENERATIONS=${SC_TURN_RECOVERY_NUM_GENERATIONS:-2}
MAX_STEPS=${SC_TURN_RECOVERY_MAX_STEPS:-3}
SNAPSHOT_INTERVAL_S=${SC_TURN_RECOVERY_INTERVAL_S:-0.2}
SNAPSHOT_TIMEOUT_S=${SC_TURN_RECOVERY_TIMEOUT_S:-2400}
PHASE2_TIMEOUT_S=${SC_TURN_RECOVERY_PHASE2_TIMEOUT_S:-2400}
TRAIN_GLOBAL_BATCH_SIZE=$((NUM_PROMPTS * NUM_GENERATIONS))

rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

# Force one real Workplace mutation per rollout. The checkpoint test agent holds
# the first boundary after it in phase one, so a published checkpoint always
# holds a turn whose calendar write has already been applied.
jq -c -s --argjson count "$NUM_PROMPTS" --arg event "$SENTINEL_EVENT" '
    limit($count; .[])
    | del(.agent_ref)
    | .task_source = "workplace_assistant_checkpoint_test_agent"
    | .responses_create_params.input = [{
        "role": "user",
        "content": ("Call calendar_create_event exactly once with event_name " + $event
            + ", participant_email checkpoint-recovery@example.com, event_start "
            + "2025-01-15 10:00:00, and duration 30.")
      }]
    | .responses_create_params.tools = [
        .responses_create_params.tools[] | select(.name == "calendar_create_event")
      ]
    | .responses_create_params.tool_choice = {"type": "function", "name": "calendar_create_event"}
    | .responses_create_params.parallel_tool_calls = false
    | .ground_truth = [{
        "name": "calendar_create_event",
        "arguments": ({
          "event_name": $event,
          "participant_email": "checkpoint-recovery@example.com",
          "event_start": "2025-01-15 10:00:00",
          "duration": "30"
        } | tojson)
      }]
    | .category = "workplace_assistant_calendar"
    | .environment_name = "workplace_assistant"
' "$GYM_ROOT/resources_servers/workplace_assistant/data/example.jsonl" > "$TEST_DATA"

stop_phase1() {
    if [[ -z "$PHASE1_PID" ]]; then
        return
    fi
    kill -KILL -- "-$PHASE1_PID" 2>/dev/null || true
    wait "$PHASE1_PID" 2>/dev/null || true
    PHASE1_PID=""
    # Ray actors run in their own process groups and outlive the driver until
    # their raylet notices; the controller may still be writing a snapshot.
    local deadline=$((SECONDS + 120))
    while pgrep -f "^ray::SingleControllerActor" >/dev/null; do
        if ((SECONDS >= deadline)); then
            echo "[ERROR] phase-one SingleControllerActor outlived its driver"
            return 1
        fi
        sleep 1
    done
}

cleanup() {
    local status=$?
    stop_phase1
    if [[ "$status" -eq 0 && "${SC_TURN_RECOVERY_KEEP_CHECKPOINTS:-0}" != "1" ]]; then
        rm -rf "$CHECKPOINT_DIR" "$SELECTED_BACKUP"
    else
        echo "Preserving recovery artifacts for inspection: $TEST_DIR"
    fi
    return "$status"
}
trap cleanup EXIT

COMMON_OVERRIDES=(
    checkpointing.enabled=true
    checkpointing.checkpoint_dir="$CHECKPOINT_DIR"
    checkpointing.metric_name=null
    checkpointing.save_period=1
    ++checkpointing.save_data_plane=true
    ++token_capture.enabled=true
    ++rollout_recovery.target_level=turn
    ++rollout_checkpointing.snapshot_attempt_interval_s="$SNAPSHOT_INTERVAL_S"
    ++rollout_checkpointing.keep_latest_k=8
    ++rollout_checkpointing.restore_mode=latest
    async_rl.sampler.name=in_order
    async_rl.sampler.max_lookahead_versions=1
    async_rl.min_groups_for_streaming_train=2
    async_rl.max_inflight_prompts=4
    async_rl.max_buffered_rollouts=8
    ++async_rl.rollout_failure.nemo_gym.rollout_timeout_s=180
    ++async_rl.stall_watchdog.interval_s=10
    ++async_rl.stall_watchdog.stall_timeout_s=300
    ++async_rl.stall_watchdog.stall_action=abort
    grpo.num_prompts_per_step="$NUM_PROMPTS"
    grpo.num_generations_per_prompt="$NUM_GENERATIONS"
    grpo.max_num_steps="$MAX_STEPS"
    policy.train_global_batch_size="$TRAIN_GLOBAL_BATCH_SIZE"
    # The base config overrides a code_gen server it never launches; Gym
    # checkpoint discovery would probe it as a participant.
    '~env.nemo_gym.code_gen'
    "env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,responses_api_agents/checkpoint_test_agent/configs/workplace_assistant.yaml]"
    data.train.data_path="$TEST_DATA"
    data.validation.data_path="$TEST_DATA"
    policy.generation.temperature=0.1
)

echo "=== Phase 1: publish a Workplace turn checkpoint, then crash ==="
command -v setsid >/dev/null
setsid env \
    SC_TEST_ENTRYPOINT="$RECOVERY_HOOK" \
    SC_TEST_CONFIG="$TEST_CONFIG" \
    SC_TURN_RECOVERY_TEST_EVENTS="$PHASE1_EVENTS" \
    NEMO_GYM_TEST_HOLD_FIRST_MUTATED_BOUNDARY=1 \
    RUN_CONVERGENCE_CHECKS=0 \
    bash "$BASE_TEST" "${COMMON_OVERRIDES[@]}" "$@" &
PHASE1_PID=$!

# Select only a fully published bootstrap (pre-step-1) snapshot whose Gym
# manifest, RL ledger, Workplace resource export and TQ references all describe
# the same episode.
uv run --directory "$PROJECT_ROOT" --no-sync python - \
    "$CHECKPOINT_DIR/bootstrap/rollout_snapshots" \
    "$SELECTION_FILE" \
    "$PHASE1_PID" \
    "$BASE_RUN_LOG" \
    "$SNAPSHOT_TIMEOUT_S" \
    "$EXPECTED_GYM_INSTANCES" \
    "$SELECTED_BACKUP" \
    "$SENTINEL_EVENT" <<'PY'
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import torch

root = Path(sys.argv[1])
selection_path = Path(sys.argv[2])
phase_pid = int(sys.argv[3])
phase_log = Path(sys.argv[4])
deadline = time.monotonic() + float(sys.argv[5])
expected_gym_instances = int(sys.argv[6])
backup_path = Path(sys.argv[7])
sentinel_event = sys.argv[8]
if expected_gym_instances < 1:
    raise ValueError("expected Gym instance count must be positive")


def ledger_episode_index(path: Path) -> dict[tuple[str, int], dict]:
    state = torch.load(path, weights_only=True)
    result = {}
    for group in state["groups"]:
        for sibling in group["siblings"]:
            attempt = sibling["attempts"][-1]
            generation_index = sibling["generation_index"]
            rollout_id = (
                f"{group['group_id']}_g{generation_index}_a"
                f"{uuid.UUID(bytes=attempt['attempt_uuid']).hex}"
            )
            result[(rollout_id, attempt["gym_attempt"])] = {
                "group_id": group["group_id"],
                "generation_index": generation_index,
                "target_step": group["target_step"],
                "restore_level": group["restore_level"],
                "status": attempt["status"],
                "gym_instance_id": attempt["gym_instance_id"],
            }
    return result


def sentinel_count(state: dict) -> int:
    """Count calendar rows the held turn wrote before the crash."""
    columns = state["calendar"]["_calendar_events"]["columns"]
    names = next(column["values"] for column in columns if column["name"] == "event_name")
    return sum(str(name).lower() == sentinel_event.lower() for name in names)


def participant_records(snapshot: Path, kind: str) -> list[tuple[Path, dict]]:
    records = []
    pattern = f"gym-instances/*/replica-*/gym/{kind}/*/manifest.json"
    for manifest_path in snapshot.glob(pattern):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("kind") != kind or manifest.get("record_count", 0) <= 0:
            continue
        records_path = manifest_path.parent / manifest["records_file"]
        for line in records_path.read_text().splitlines():
            if line:
                records.append((manifest_path, json.loads(line)))
    return records


while time.monotonic() < deadline:
    for snapshot in sorted(root.glob("snapshot_*"), reverse=True):
        required = [
            snapshot / "manifest.json",
            snapshot / "gym_checkpoint.json",
            snapshot / "rollout_recovery.pt",
            snapshot / "replay_buffer_metadata.pt",
        ]
        if not all(path.is_file() for path in required) or not (snapshot / "data_plane").is_dir():
            continue
        manifest = json.loads(required[0].read_text())
        if manifest.get("base_train_step") != 0 or manifest.get("trainer_version") != 0:
            continue
        try:
            gym = json.loads(required[1].read_text())
            ledger = ledger_episode_index(required[2])
            resources = participant_records(snapshot, "resources")
            model = participant_records(snapshot, "model")
        except FileNotFoundError:
            # Pruned by phase one's first trainer checkpoint mid-inspection.
            continue
        if len(gym["instances"]) != expected_gym_instances:
            continue
        resource_index = {
            (record["episode_id"]["rollout_id"], record["episode_id"]["attempt"]): (
                path,
                record["state"],
            )
            for path, record in resources
            if record.get("state")
        }
        model_index = {
            (record["episode_id"]["rollout_id"], record["episode_id"]["attempt"]): path
            for path, record in model
            if record.get("rows")
        }
        for instance_id, episodes in gym["instances"].items():
            if not gym["staging_keys"].get(instance_id):
                continue
            for episode in episodes:
                key = (episode["rollout_id"], episode["attempt"])
                recovery = ledger.get(key)
                resources_entry = resource_index.get(key)
                model_manifest = model_index.get(key)
                if (
                    recovery is None
                    or resources_entry is None
                    or model_manifest is None
                ):
                    continue
                resources_manifest, resources_state = resources_entry
                # The held turn applied its calendar write exactly once.
                if sentinel_count(resources_state) != 1:
                    continue
                if (
                    recovery["restore_level"] != "turn"
                    or recovery["status"] != "dispatched"
                    or recovery["gym_instance_id"] != instance_id
                ):
                    continue
                try:
                    shutil.copytree(snapshot, backup_path)
                except FileNotFoundError:
                    shutil.rmtree(backup_path, ignore_errors=True)
                    continue
                selection_path.write_text(
                    json.dumps(
                        {
                            "snapshot": str(snapshot),
                            "snapshot_name": snapshot.name,
                            "gym_instance_id": instance_id,
                            "gym_rollout_id": episode["rollout_id"],
                            "gym_attempt": episode["attempt"],
                            "group_id": recovery["group_id"],
                            "generation_index": recovery["generation_index"],
                            "target_step": recovery["target_step"],
                            "resources_manifest": str(resources_manifest.relative_to(snapshot)),
                            "model_manifest": str(model_manifest.relative_to(snapshot)),
                        },
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n"
                )
                raise SystemExit(0)
    try:
        os.kill(phase_pid, 0)
    except ProcessLookupError as error:
        tail = ""
        if phase_log.is_file():
            tail = "\n".join(phase_log.read_text(errors="replace").splitlines()[-60:])
        raise RuntimeError(
            "phase one exited before publishing a recoverable Workplace turn:\n" + tail
        ) from error
    time.sleep(0.2)
raise TimeoutError("no bootstrap snapshot contained a recoverable Workplace turn")
PY

stop_phase1
cp "$BASE_RUN_LOG" "$PHASE1_LOG"
SNAPSHOT_DIR=$(uv run --directory "$PROJECT_ROOT" --no-sync python -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["snapshot"])' \
    "$SELECTION_FILE")
SNAPSHOT_ROOT=$(dirname "$SNAPSHOT_DIR")

# Make recovery select exactly the validated cut: phase one may have published
# newer snapshots or pruned this one, and any trainer checkpoint it wrote would
# take precedence over the bootstrap anchor.
rm -rf "$SNAPSHOT_ROOT"
mkdir -p "$SNAPSHOT_ROOT"
cp -a "$SELECTED_BACKUP" "$SNAPSHOT_DIR"
for trainer_checkpoint in "$CHECKPOINT_DIR"/step_*; do
    if [[ -d "$trainer_checkpoint" ]]; then
        rm -rf "$trainer_checkpoint"
    fi
done

echo "=== Phase 2: restore the turn on its original Gym replica ==="
timeout --signal=TERM --kill-after=30s "${PHASE2_TIMEOUT_S}s" \
    env \
        SC_TEST_ENTRYPOINT="$RECOVERY_HOOK" \
        SC_TEST_CONFIG="$TEST_CONFIG" \
        SC_TURN_RECOVERY_TEST_EVENTS="$PHASE2_EVENTS" \
        RUN_CONVERGENCE_CHECKS=0 \
    bash "$BASE_TEST" "${COMMON_OVERRIDES[@]}" "$@"
cp "$BASE_RUN_LOG" "$PHASE2_LOG"

grep -Fq "Selected rollout recovery snapshot: $SNAPSHOT_DIR" "$PHASE2_LOG"
grep -q "Native TQ checkpoint restored and validated" "$PHASE2_LOG"
grep -q "Loaded .* unfinished rollout group(s)" "$PHASE2_LOG"
grep -q "train step $MAX_STEPS/$MAX_STEPS" "$PHASE2_LOG"

PHASE2_METRICS=$TEST_DIR/phase2-metrics.json
uv run --directory "$PROJECT_ROOT" --no-sync tests/json_dump_tb_logs.py \
    "$SCRIPT_DIR/grpo_async_gym_single_controller/logs" \
    --output_path "$PHASE2_METRICS"
uv run --directory "$PROJECT_ROOT" --no-sync tests/check_metrics.py "$PHASE2_METRICS" \
    'max(data["train/finalize/invalid_row_rate"]) == 0' \
    'max(data["train/finalize/capture_poisoned_rollouts"]) == 0'

uv run --directory "$PROJECT_ROOT" --no-sync python - \
    "$SELECTION_FILE" "$PHASE1_EVENTS" "$PHASE2_EVENTS" "$PHASE2_LOG" \
    "$CHECKPOINT_DIR/step_$MAX_STEPS/training_info.json" "$MAX_STEPS" <<'PY'
import json
import re
import sys
from pathlib import Path

selection = json.loads(Path(sys.argv[1]).read_text())
phase1 = [json.loads(line) for line in Path(sys.argv[2]).read_text().splitlines()]
phase2 = [json.loads(line) for line in Path(sys.argv[3]).read_text().splitlines()]
log = Path(sys.argv[4]).read_text()
training_info = json.loads(Path(sys.argv[5]).read_text())
max_steps = int(sys.argv[6])


def matching_siblings(events: list[dict], event_name: str) -> list[dict]:
    found = []
    for event in events:
        if event["event"] != event_name or event["group_id"] != selection["group_id"]:
            continue
        siblings = event.get("siblings") or [event.get("sibling")]
        found.extend(
            sibling
            for sibling in siblings
            if sibling is not None
            and sibling["gym_rollout_id"] == selection["gym_rollout_id"]
        )
    return found


phase1_dispatch = matching_siblings(phase1, "dispatch")
assert phase1_dispatch, "selected Gym episode was not observed in phase-one dispatches"
assert all(
    sibling["gym_instance_id"] == selection["gym_instance_id"]
    and sibling["gym_attempt"] == selection["gym_attempt"]
    for sibling in phase1_dispatch
), phase1_dispatch

phase2_dispatch = matching_siblings(phase2, "dispatch")
# A submission Gym refused at checkpoint admission never ran; it is re-sent.
phase2_refused = matching_siblings(phase2, "refused")
assert len(phase2_dispatch) - len(phase2_refused) == 1, (phase2_dispatch, phase2_refused)
restored = phase2_dispatch[-1]
assert restored["gym_instance_id"] == selection["gym_instance_id"], restored
assert restored["gym_attempt"] == selection["gym_attempt"] + 1, restored
assert restored["generation_index"] == selection["generation_index"], restored

phase2_completion = matching_siblings(phase2, "completion")
assert phase2_completion, "restored Gym episode never completed"
assert any(
    sibling["gym_instance_id"] == selection["gym_instance_id"]
    and sibling["gym_attempt"] == selection["gym_attempt"] + 1
    and sibling["status"] == "sealed"
    for sibling in phase2_completion
), phase2_completion

assert training_info["current_step"] == max_steps, training_info
assert training_info["trainer_version"] == max_steps, training_info
# Restored from the bootstrap anchor, so every step trains exactly once.
for step in range(1, max_steps + 1):
    matches = re.findall(rf"train step {step}/{max_steps}(?:\s|$)", log)
    assert len(matches) == 1, (step, len(matches))
PY

echo "Workplace turn-level recovery functional test passed."
