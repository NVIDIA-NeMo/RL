#!/bin/bash
# Full-process recovery test for one Gym turn boundary plus the matching TQ cut.
#
# This test intentionally uses an opt-in Gym source checkout until the Gym
# checkpoint stack lands in NeMo-RL's submodule. Example:
#   NEMO_GYM_SOURCE_DIR=~/projects/Gym-turn-level-recovery \
#     uv run --no-sync bash tests/functional/grpo_async_gym_single_controller_turn_recovery.sh

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
BASE_TEST=$SCRIPT_DIR/grpo_async_gym_single_controller.sh
BASE_RUN_LOG=$SCRIPT_DIR/grpo_async_gym_single_controller/run.log
RECOVERY_HOOK=$SCRIPT_DIR/_single_controller_sibling_recovery_hook.py
SNAPSHOT_HELPER=$SCRIPT_DIR/_gym_turn_recovery_snapshot.py
TEST_DIR=$SCRIPT_DIR/grpo_async_gym_single_controller_turn_recovery
CHECKPOINT_DIR=$TEST_DIR/checkpoints
PHASE1_LOG=$TEST_DIR/phase1.log
PHASE2_LOG=$TEST_DIR/phase2.log
PHASE1_EVENTS=$TEST_DIR/phase1-events.jsonl
PHASE2_EVENTS=$TEST_DIR/phase2-events.jsonl
SELECTION_FILE=$TEST_DIR/selected_snapshot.json
COUNTER_DATA=$TEST_DIR/session_state_mgmt.jsonl
PHASE1_PID=""

GYM_ROOT=${NEMO_GYM_SOURCE_DIR:-$PROJECT_ROOT/3rdparty/Gym-workspace/Gym}
SNAPSHOT_INTERVAL_S=${SC_GYM_TURN_RECOVERY_INTERVAL_S:-5}
SNAPSHOT_TIMEOUT_S=${SC_GYM_TURN_RECOVERY_TIMEOUT_S:-2400}
PHASE2_TIMEOUT_S=${SC_GYM_TURN_RECOVERY_PHASE2_TIMEOUT_S:-2400}
MAX_STEPS=${SC_GYM_TURN_RECOVERY_MAX_STEPS:-2}
NUM_PROMPTS=${SC_GYM_TURN_RECOVERY_NUM_PROMPTS:-4}
NUM_GENERATIONS=${SC_GYM_TURN_RECOVERY_NUM_GENERATIONS:-2}
TRAIN_GLOBAL_BATCH_SIZE=$((NUM_PROMPTS * NUM_GENERATIONS))

if [[ ! -f "$GYM_ROOT/nemo_gym/_checkpoint/agent.py" ]]; then
    echo "[ERROR] $GYM_ROOT does not contain the Gym turn-checkpoint stack."
    echo "Set NEMO_GYM_SOURCE_DIR to a checkout containing Gym PRs #2939-#2946."
    exit 2
fi

export NEMO_GYM_SOURCE_DIR=$GYM_ROOT
export NEMO_GYM_CHECKPOINT_CONTROL_TOKEN=${NEMO_GYM_CHECKPOINT_CONTROL_TOKEN:-functional-test-checkpoint-token}

rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

# Use a deterministic, stateful environment. A restored rollout receives reward
# 1 only when its saved counter is continued exactly once; replaying a committed
# pre-checkpoint increment necessarily makes the positive-only count too large.
jq -c \
    '.task_source //= "example_session_state_mgmt_simple_agent"' \
    "$GYM_ROOT/resources_servers/example_session_state_mgmt/data/example.jsonl" \
    > "$COUNTER_DATA"
export NEMO_GYM_TRAIN_DATA_PATH=$COUNTER_DATA
export NEMO_GYM_VALIDATION_DATA_PATH=$COUNTER_DATA

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
    ++rollout_checkpointing.snapshot_attempt_interval_s="$SNAPSHOT_INTERVAL_S"
    ++rollout_checkpointing.keep_latest_k=8
    ++rollout_checkpointing.restore_mode=latest
    ++rollout_checkpointing.gym.capability_discovery_enabled=true
    ++rollout_checkpointing.gym.participant_checkpointing_enabled=true
    ++env.nemo_gym.nemo_gym_log_dir="$TEST_DIR/gym_logs"
    ++rollout_checkpointing.gym.prepare_timeout_s=180
    async_rl.sampler.name=in_order
    async_rl.sampler.max_lookahead_versions=0
    async_rl.min_groups_for_streaming_train="$NUM_PROMPTS"
    async_rl.max_inflight_prompts="$NUM_PROMPTS"
    async_rl.max_buffered_rollouts="$NUM_PROMPTS"
    ++async_rl.rollout_failure.nemo_gym.rollout_timeout_s=180
    ++async_rl.stall_watchdog.interval_s=10
    ++async_rl.stall_watchdog.stall_timeout_s=300
    ++async_rl.stall_watchdog.stall_action=abort
    grpo.num_prompts_per_step="$NUM_PROMPTS"
    grpo.num_generations_per_prompt="$NUM_GENERATIONS"
    grpo.max_num_steps="$MAX_STEPS"
    policy.train_global_batch_size="$TRAIN_GLOBAL_BATCH_SIZE"
    policy.generation.temperature=0.1
    policy.generation.max_new_tokens=128
    'env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,responses_api_agents/checkpoint_test_agent/configs/example_session_state_mgmt.yaml]'
    '~env.nemo_gym.code_gen'
)

echo "=== Phase 1: publish one coordinated Gym + TQ turn checkpoint ==="
command -v setsid >/dev/null
setsid env \
    SC_TEST_ENTRYPOINT="$RECOVERY_HOOK" \
    SC_SIBLING_RECOVERY_TEST_EVENTS="$PHASE1_EVENTS" \
    NEMO_GYM_TEST_HOLD_FIRST_BOUNDARY=1 \
    RUN_CONVERGENCE_CHECKS=0 \
    NEMO_GYM_SOURCE_DIR="$GYM_ROOT" \
    NEMO_GYM_CHECKPOINT_CONTROL_TOKEN="$NEMO_GYM_CHECKPOINT_CONTROL_TOKEN" \
    bash "$BASE_TEST" "${COMMON_OVERRIDES[@]}" &
PHASE1_PID=$!

uv run --directory "$PROJECT_ROOT" --no-sync python "$SNAPSHOT_HELPER" select \
    "$CHECKPOINT_DIR" \
    "$SELECTION_FILE" \
    "$PHASE1_PID" \
    "$BASE_RUN_LOG" \
    "$COUNTER_DATA" \
    "$SNAPSHOT_TIMEOUT_S"

stop_phase1
cp "$BASE_RUN_LOG" "$PHASE1_LOG"

SNAPSHOT_DIR=$(uv run --directory "$PROJECT_ROOT" --no-sync python -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["snapshot_path"])' \
    "$SELECTION_FILE")
SNAPSHOT_ROOT=$(dirname "$SNAPSHOT_DIR")

# Force phase two to recover the exact turn boundary selected above. Anything
# committed after the selection belongs to work deliberately discarded by the
# simulated crash.
for candidate in "$SNAPSHOT_ROOT"/snapshot_*; do
    if [[ -d "$candidate" && "$candidate" != "$SNAPSHOT_DIR" ]]; then
        rm -rf "$candidate"
    fi
done
for trainer_checkpoint in "$CHECKPOINT_DIR"/step_*; do
    if [[ -d "$trainer_checkpoint" ]]; then
        rm -rf "$trainer_checkpoint"
    fi
done

echo "=== Phase 2: restore Gym, TQ, replay, and the unfinished RL sibling ==="
timeout --signal=TERM --kill-after=30s "${PHASE2_TIMEOUT_S}s" \
    env \
        SC_TEST_ENTRYPOINT="$RECOVERY_HOOK" \
        SC_SIBLING_RECOVERY_TEST_EVENTS="$PHASE2_EVENTS" \
        RUN_CONVERGENCE_CHECKS=0 \
        NEMO_GYM_SOURCE_DIR="$GYM_ROOT" \
        NEMO_GYM_CHECKPOINT_CONTROL_TOKEN="$NEMO_GYM_CHECKPOINT_CONTROL_TOKEN" \
        bash "$BASE_TEST" "${COMMON_OVERRIDES[@]}"
cp "$BASE_RUN_LOG" "$PHASE2_LOG"

grep -Fq "Selected rollout recovery snapshot: $SNAPSHOT_DIR" "$PHASE2_LOG"
grep -q "Native TQ checkpoint restored and validated" "$PHASE2_LOG"
grep -q "/ng-control/v1/model-checkpoint/restore" "$PHASE2_LOG"
grep -q "/ng-control/v1/agent-checkpoint/restore" "$PHASE2_LOG"
grep -q "/ng-control/v1/resources-checkpoint/restore" "$PHASE2_LOG"
grep -q "train step $MAX_STEPS/$MAX_STEPS" "$PHASE2_LOG"

uv run --directory "$PROJECT_ROOT" --no-sync python "$SNAPSHOT_HELPER" \
    verify-restore "$SELECTION_FILE" "$PHASE2_EVENTS"

uv run --directory "$PROJECT_ROOT" --no-sync python - \
    "$CHECKPOINT_DIR/step_$MAX_STEPS/training_info.json" \
    "$MAX_STEPS" <<'PY'
import json
import sys
from pathlib import Path

training_info = json.loads(Path(sys.argv[1]).read_text())
max_steps = int(sys.argv[2])
assert training_info["current_step"] == max_steps, training_info
assert training_info["trainer_version"] == max_steps, training_info
PY

echo "Coordinated Gym turn-boundary + TQ crash/restore functional test passed."
