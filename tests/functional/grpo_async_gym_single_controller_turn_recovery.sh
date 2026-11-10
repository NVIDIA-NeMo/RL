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
SENTINEL_PARTICIPANT="checkpoint-recovery@example.com"
FOLLOW_UP_TOOL=calendar_search_events
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

# Force one real Workplace mutation per rollout. In phase one the test hook's
# policy proxy holds the first model call that carries a tool result, so Gym
# stays at the post-tool boundary and a published checkpoint always holds a turn
# whose calendar write has already been applied. After the write, the proxy
# forces one calendar_search_events call, which reads the calendar the episode
# continued from, and then sends tool_choice none, so no turn writes a second
# event. The search names the participant only this test's events use.
jq -c -s --argjson count "$NUM_PROMPTS" --arg event "$SENTINEL_EVENT" \
    --arg participant "$SENTINEL_PARTICIPANT" '
    limit($count; .[])
    | del(.agent_ref)
    | .task_source = "workplace_assistant_simple_agent"
    | .responses_create_params.input = [{
        "role": "user",
        "content": ("Call calendar_create_event exactly once with event_name " + $event
            + ", participant_email " + $participant + ", event_start "
            + "2025-01-15 10:00:00, and duration 30. Then call calendar_search_events "
            + "exactly once with query " + $participant + ".")
      }]
    | .responses_create_params.tools = [
        .responses_create_params.tools[]
        | select(.name == "calendar_create_event" or .name == "calendar_search_events")
      ]
    | .responses_create_params.tool_choice = {"type": "function", "name": "calendar_create_event"}
    | .responses_create_params.parallel_tool_calls = false
    | .ground_truth = [{
        "name": "calendar_create_event",
        "arguments": ({
          "event_name": $event,
          "participant_email": $participant,
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

# A sharded config names its servers per shard, rejects a top-level list, and
# already drops the base config's code_gen server.
if [[ "$EXPECTED_GYM_INSTANCES" -eq 1 ]]; then
    GYM_CONFIG_OVERRIDES=(
        # The base config overrides a code_gen server it never launches; Gym
        # checkpoint discovery would probe it as a participant.
        '~env.nemo_gym.code_gen'
    )
else
    GYM_CONFIG_OVERRIDES=()
fi

COMMON_OVERRIDES=(
    checkpointing.enabled=true
    checkpointing.checkpoint_dir="$CHECKPOINT_DIR"
    checkpointing.metric_name=null
    checkpointing.save_period=1
    ++checkpointing.save_data_plane=true
    ++token_capture.enabled=true
    ++rollout_recovery.target_level=turn
    # stall_timeout_s below must outlast a Gym park:
    # gym_prepare_timeout_s + 2 * token_capture.control_timeout_s (60s).
    ++rollout_recovery.gym_prepare_timeout_s=120
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
    data.train.data_path="$TEST_DATA"
    data.validation.data_path="$TEST_DATA"
    policy.generation.temperature=0.1
    # The prompt names two tools, and an episode makes three calls.
    policy.max_total_sequence_length=1024
    "${GYM_CONFIG_OVERRIDES[@]}"
)

echo "=== Phase 1: publish a Workplace turn checkpoint, then crash ==="
command -v setsid >/dev/null
setsid env \
    SC_TEST_ENTRYPOINT="$RECOVERY_HOOK" \
    SC_TEST_CONFIG="$TEST_CONFIG" \
    SC_TURN_RECOVERY_TEST_EVENTS="$PHASE1_EVENTS" \
    SC_TURN_RECOVERY_FOLLOW_UP_TOOL="$FOLLOW_UP_TOOL" \
    SC_TURN_RECOVERY_HOLD_FIRST_TOOL_RESULT=1 \
    RUN_CONVERGENCE_CHECKS=0 \
    bash "$BASE_TEST" "${COMMON_OVERRIDES[@]}" "$@" &
PHASE1_PID=$!

# Select only a fully published bootstrap (pre-step-1) snapshot whose Gym
# manifest, RL ledger, Workplace resource export and TQ references all describe
# the episode the proxy held. Another episode also counts as at its boundary
# while its own post-tool call is briefly in flight, so the hold, not the
# snapshot contents, names the episode.
uv run --directory "$PROJECT_ROOT" --no-sync python - \
    "$CHECKPOINT_DIR/bootstrap/rollout_snapshots" \
    "$SELECTION_FILE" \
    "$PHASE1_PID" \
    "$BASE_RUN_LOG" \
    "$SNAPSHOT_TIMEOUT_S" \
    "$EXPECTED_GYM_INSTANCES" \
    "$SELECTED_BACKUP" \
    "$SENTINEL_EVENT" \
    "$PHASE1_EVENTS" <<'PY'
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

root = Path(sys.argv[1])
selection_path = Path(sys.argv[2])
phase_pid = int(sys.argv[3])
phase_log = Path(sys.argv[4])
deadline = time.monotonic() + float(sys.argv[5])
expected_gym_instances = int(sys.argv[6])
backup_path = Path(sys.argv[7])
sentinel_event = sys.argv[8]
events_path = Path(sys.argv[9])
if expected_gym_instances < 1:
    raise ValueError("expected Gym instance count must be positive")


def ledger_episode_index(path: Path) -> dict[tuple[str, int], dict]:
    # Imported only once a snapshot exists: phase 1's training may still be
    # syncing the venv this interpreter shares when the selector starts.
    import torch

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


def held_rollout_id() -> str | None:
    """The Gym rollout ID of the policy call the proxy holds, once it holds one."""
    if not events_path.is_file():
        return None
    held = [
        event["rollout_id"]
        for event in map(json.loads, events_path.read_text().splitlines())
        if event["event"] == "policy_call_held"
    ]
    if len(held) > 1:
        raise RuntimeError(f"the policy proxy held more than one call: {held}")
    return held[0] if held else None


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
    held = held_rollout_id()
    for snapshot in sorted(root.glob("snapshot_*"), reverse=True) if held else []:
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
            (record["episode_id"]["rollout_id"], record["episode_id"]["attempt"]): (
                path,
                [str(row["model_call_id"]) for row in record["rows"]],
            )
            for path, record in model
            if record.get("rows")
        }
        for instance_id, episodes in gym["instances"].items():
            if not gym["staging_keys"].get(instance_id):
                continue
            for episode in episodes:
                # The capture key of a later attempt carries an -a<N> suffix.
                if held != episode["rollout_id"] and not held.startswith(
                    episode["rollout_id"] + "-a"
                ):
                    continue
                key = (episode["rollout_id"], episode["attempt"])
                recovery = ledger.get(key)
                resources_entry = resource_index.get(key)
                model_entry = model_index.get(key)
                if (
                    recovery is None
                    or resources_entry is None
                    or model_entry is None
                ):
                    continue
                model_manifest, pre_cut_model_call_ids = model_entry
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
                            # Calls delivered before the cut; a continued episode keeps them.
                            "pre_cut_model_call_ids": pre_cut_model_call_ids,
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
        SC_TURN_RECOVERY_FOLLOW_UP_TOOL="$FOLLOW_UP_TOOL" \
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
    "$CHECKPOINT_DIR/step_$MAX_STEPS/training_info.json" "$MAX_STEPS" \
    "$SENTINEL_EVENT" "$SENTINEL_PARTICIPANT" "$FOLLOW_UP_TOOL" <<'PY'
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
sentinel_event, sentinel_participant, follow_up_tool = sys.argv[7:10]


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


# The proxy held exactly one post-tool policy call in phase one, and it is the
# selected episode's: that hold is what kept Gym at the post-tool boundary.
held = [event for event in phase1 if event["event"] == "policy_call_held"]
assert len(held) == 1, held
assert (held[0]["rollout_id"] or "").startswith(selection["gym_rollout_id"]), (
    held,
    selection,
)

phase1_dispatch = matching_siblings(phase1, "dispatch")
assert phase1_dispatch, "selected Gym episode was not observed in phase-one dispatches"
assert all(
    sibling["gym_instance_id"] == selection["gym_instance_id"]
    and sibling["gym_attempt"] == selection["gym_attempt"]
    for sibling in phase1_dispatch
), phase1_dispatch

# Gym installed the selected episode's saved state; without it, Gym would
# open a fresh session and the episode would start over from its prompt.
restores = [event for event in phase2 if event["event"] == "gym_restore"]
assert len(restores) == 1, restores
assert {
    "gym_instance_id": selection["gym_instance_id"],
    "gym_rollout_id": selection["gym_rollout_id"],
    "gym_attempt": selection["gym_attempt"],
} in restores[0]["episodes"], (selection, restores)

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

# Continued, not restarted: the sealed receipt keeps every model call the
# episode made before the cut. A restart from the prompt makes only new calls.
# The calendar search below cannot tell the two apart, since a restart writes
# the event again into a fresh calendar; this check does.
restored_completions = [
    event
    for event in phase2
    if event["event"] == "completion"
    and event["group_id"] == selection["group_id"]
    and event["sibling"]["gym_rollout_id"] == selection["gym_rollout_id"]
    and event["sibling"]["status"] == "sealed"
]
assert len(restored_completions) == 1, restored_completions

# Phase two ran behind the proxy without a hold: the restored episode was free to
# finish, and every post-tool call it made went through the tool_choice rules.
proxies = {
    phase: [event for event in events if event["event"] == "policy_proxy_started"]
    for phase, events in (("phase1", phase1), ("phase2", phase2))
}
assert [event["hold_first_tool_result"] for event in proxies["phase1"]] == [True], proxies
assert [event["hold_first_tool_result"] for event in proxies["phase2"]] == [False], proxies


def of_selected(event_name: str) -> list[dict]:
    return [
        event
        for event in phase2
        if event["event"] == event_name
        and (event["rollout_id"] or "").startswith(selection["gym_rollout_id"])
    ]


# The restored episode re-issued the held call, and the proxy forced the
# read-only search with it, so it could not write a second event.
reissued = of_selected("policy_call_tool_choice")
assert reissued, "the restored episode never re-issued its post-tool policy call"
assert reissued[0]["tool_choice"] == {
    "type": "function",
    "function": {"name": follow_up_tool},
}, reissued
# That search read the calendar the restored episode continued from: the event
# the held turn wrote before the crash, exactly once. A restore that lost the
# Workplace state would find none; a second write would find two.
searches = of_selected("follow_up_tool_result")
assert searches, "the restored episode never answered its calendar search"
search_arguments = json.loads(searches[-1]["arguments"] or "{}")
assert search_arguments.get("query") == sentinel_participant, searches[-1]
found = str(searches[-1]["result"]).lower().count(sentinel_event.lower())
assert found == 1, (found, searches[-1])
# Workplace replays the episode's calls on a fresh calendar and compares it with
# the ground truth's single event, so the episode made exactly one correct write.
assert restored_completions[0]["reward"] == 1.0, restored_completions[0]
pre_cut_calls = selection["pre_cut_model_call_ids"]
assert pre_cut_calls, selection
missing_calls = set(pre_cut_calls) - set(restored_completions[0]["receipt_model_call_ids"])
assert not missing_calls, (missing_calls, restored_completions[0])

assert training_info["current_step"] == max_steps, training_info
assert training_info["trainer_version"] == max_steps, training_info
# Restored from the bootstrap anchor, so every step trains exactly once.
for step in range(1, max_steps + 1):
    matches = re.findall(rf"train step {step}/{max_steps}(?:\s|$)", log)
    assert len(matches) == 1, (step, len(matches))
PY

echo "Workplace turn-level recovery functional test passed."
