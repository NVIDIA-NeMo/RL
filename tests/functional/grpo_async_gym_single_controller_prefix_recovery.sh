#!/bin/bash
# Full-process recovery test for one model call cut during active vLLM decode.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
BASE_TEST=$SCRIPT_DIR/grpo_async_gym_single_controller.sh
BASE_RUN_LOG=$SCRIPT_DIR/grpo_async_gym_single_controller/run.log
RECOVERY_HOOK=$SCRIPT_DIR/_single_controller_sibling_recovery_hook.py
SNAPSHOT_HELPER=$SCRIPT_DIR/_gym_prefix_recovery_snapshot.py
PROFILE=${SC_GYM_PREFIX_RECOVERY_PROFILE:-basic}
if [[ "$PROFILE" != "basic" && "$PROFILE" != "workplace" ]]; then
    echo "[ERROR] Unsupported SC_GYM_PREFIX_RECOVERY_PROFILE=$PROFILE"
    exit 2
fi
TEST_SUFFIX=""
if [[ "$PROFILE" != "basic" ]]; then
    TEST_SUFFIX="_$PROFILE"
fi
TEST_DIR=$SCRIPT_DIR/grpo_async_gym_single_controller_prefix_recovery$TEST_SUFFIX
CHECKPOINT_DIR=$TEST_DIR/checkpoints
PHASE1_LOG=$TEST_DIR/phase1.log
PHASE2_LOG=$TEST_DIR/phase2.log
PHASE1_EVENTS=$TEST_DIR/phase1-events.jsonl
PHASE2_EVENTS=$TEST_DIR/phase2-events.jsonl
SELECTION_FILE=$TEST_DIR/selected_snapshot.json
TEST_DATA=$TEST_DIR/test_data.jsonl
AUDIT_EVENTS=$TEST_DIR/resource-audit.jsonl
PHASE1_PID=""

GYM_ROOT=${NEMO_GYM_SOURCE_DIR:-$PROJECT_ROOT/3rdparty/Gym-workspace/Gym}
DEFAULT_SNAPSHOT_INTERVAL_S=1
DEFAULT_MIN_GENERATION_TOKENS=384
DEFAULT_MAX_TOTAL_SEQUENCE_LENGTH=1024
if [[ "$PROFILE" == "workplace" ]]; then
    # The combined profile must first complete a tool turn. Give its second
    # call a longer decode window so a periodic cut reliably catches it.
    DEFAULT_SNAPSHOT_INTERVAL_S=0.25
    DEFAULT_MIN_GENERATION_TOKENS=1024
    DEFAULT_MAX_TOTAL_SEQUENCE_LENGTH=2048
fi
SNAPSHOT_INTERVAL_S=${SC_GYM_PREFIX_RECOVERY_INTERVAL_S:-$DEFAULT_SNAPSHOT_INTERVAL_S}
PHASE2_SNAPSHOT_INTERVAL_S=${SC_GYM_PREFIX_RECOVERY_PHASE2_INTERVAL_S:-600}
SNAPSHOT_TIMEOUT_S=${SC_GYM_PREFIX_RECOVERY_TIMEOUT_S:-2400}
PHASE2_TIMEOUT_S=${SC_GYM_PREFIX_RECOVERY_PHASE2_TIMEOUT_S:-2400}
NUM_PROMPTS=${SC_GYM_PREFIX_RECOVERY_NUM_PROMPTS:-2}
NUM_GENERATIONS=${SC_GYM_PREFIX_RECOVERY_NUM_GENERATIONS:-2}
MIN_GENERATION_TOKENS=${SC_GYM_PREFIX_RECOVERY_MIN_TOKENS:-$DEFAULT_MIN_GENERATION_TOKENS}
MAX_TOTAL_SEQUENCE_LENGTH=${SC_GYM_PREFIX_RECOVERY_MAX_TOTAL_SEQUENCE_LENGTH:-$DEFAULT_MAX_TOTAL_SEQUENCE_LENGTH}
TRAIN_GLOBAL_BATCH_SIZE=$((NUM_PROMPTS * NUM_GENERATIONS))

if [[ ! -f "$GYM_ROOT/nemo_gym/_checkpoint/model_control_contracts.py" ]]; then
    echo "[ERROR] $GYM_ROOT does not contain the Gym generation-prefix stack."
    echo "Set NEMO_GYM_SOURCE_DIR to the Gym token-prefix-recovery checkout."
    exit 2
fi

export NEMO_GYM_SOURCE_DIR=$GYM_ROOT
export NEMO_GYM_CHECKPOINT_CONTROL_TOKEN=${NEMO_GYM_CHECKPOINT_CONTROL_TOKEN:-functional-test-checkpoint-token}

rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

CHECKPOINT_TEST_ENV=()
if [[ "$PROFILE" == "basic" ]]; then
    # Keep the first policy call alive long enough for a periodic checkpoint to
    # cut a non-empty prefix. No tools are exposed so this profile isolates the
    # model-call continuation mechanism.
    jq -c -s \
        --argjson count "$NUM_PROMPTS" \
        --argjson min_tokens "$MIN_GENERATION_TOKENS" '
            limit($count; .[])
            | .task_source = "example_session_state_mgmt_simple_agent"
            | .responses_create_params.input = [{
                "role": "user",
                "content": "Write a long numbered list. Continue until the output limit and do not call tools."
              }]
            | .responses_create_params.tools = []
            | .responses_create_params.tool_choice = "none"
            | .responses_create_params.max_output_tokens = $min_tokens
            | .responses_create_params.metadata = ((.responses_create_params.metadata // {}) + {
                "extra_body": ({"min_tokens": $min_tokens} | tojson)
              })
        ' "$GYM_ROOT/resources_servers/example_session_state_mgmt/data/example.jsonl" \
        > "$TEST_DATA"
    GYM_CONFIG_PATHS='[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,responses_api_agents/checkpoint_test_agent/configs/example_session_state_mgmt.yaml]'
else
    # Call one deterministically mutates Workplace. The checkpoint test agent
    # rewrites call two into a long tool-free decode so the selected snapshot
    # contains both the committed turn and an active generation prefix.
    jq -c -s --argjson count "$NUM_PROMPTS" '
        limit($count; .[])
        | .task_source = "workplace_assistant_prefix_checkpoint_test_agent"
        | .responses_create_params.input = [{
            "role": "user",
            "content": "Call calendar_create_event exactly once with event_name NeMo RL checkpoint recovery sentinel, participant_email checkpoint-recovery@example.com, event_start 2025-01-15 10:00:00, and duration 30. Then explain that the event was created."
          }]
        | .responses_create_params.tools = [
            .responses_create_params.tools[]
            | select(.name == "calendar_create_event")
          ]
        | .responses_create_params.tool_choice = {
            "type": "function",
            "name": "calendar_create_event"
          }
        | .responses_create_params.parallel_tool_calls = false
        | .responses_create_params.max_output_tokens = 128
        | .ground_truth = [{
            "name": "calendar_create_event",
            "arguments": ({
              "event_name": "NeMo RL checkpoint recovery sentinel",
              "participant_email": "checkpoint-recovery@example.com",
              "event_start": "2025-01-15 10:00:00",
              "duration": "30"
            } | tojson)
          }]
        | .category = "workplace_assistant_calendar"
        | .environment_name = "workplace_assistant"
    ' "$GYM_ROOT/resources_servers/workplace_assistant/data/example.jsonl" \
        > "$TEST_DATA"
    GYM_CONFIG_PATHS='[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,responses_api_agents/checkpoint_test_agent/configs/workplace_assistant_prefix_recovery.yaml]'
    CHECKPOINT_TEST_ENV=(
        NEMO_GYM_TEST_WORKPLACE_PREFIX_AFTER_MUTATION=1
        NEMO_GYM_TEST_PREFIX_MIN_TOKENS="$MIN_GENERATION_TOKENS"
        NEMO_GYM_CHECKPOINT_TEST_EVENTS="$AUDIT_EVENTS"
    )
fi

export NEMO_GYM_TRAIN_DATA_PATH=$TEST_DATA
export NEMO_GYM_VALIDATION_DATA_PATH=$TEST_DATA

stop_phase1() {
    if [[ -z "$PHASE1_PID" ]]; then
        return
    fi
    kill -KILL -- "-$PHASE1_PID" 2>/dev/null || true
    wait "$PHASE1_PID" 2>/dev/null || true
    PHASE1_PID=""
}

cleanup() {
    local status=$?
    stop_phase1
    if [[ "$status" -eq 0 && "${SC_GYM_PREFIX_RECOVERY_KEEP_CHECKPOINTS:-0}" != "1" ]]; then
        rm -rf "$CHECKPOINT_DIR"
    else
        echo "Preserving prefix-recovery artifacts for inspection: $TEST_DIR"
    fi
    return "$status"
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
    ++rollout_checkpointing.keep_latest_k=16
    ++rollout_checkpointing.restore_mode=latest
    ++rollout_checkpointing.gym.capability_discovery_enabled=true
    ++rollout_checkpointing.gym.participant_checkpointing_enabled=true
    ++rollout_checkpointing.gym.generation_prefix_cuts_enabled=true
    ++rollout_checkpointing.gym.prepare_timeout_s=180
    ++env.nemo_gym.nemo_gym_log_dir="$TEST_DIR/gym_logs"
    async_rl.sampler.name=in_order
    async_rl.sampler.max_lookahead_versions=0
    async_rl.min_groups_for_streaming_train="$NUM_PROMPTS"
    async_rl.max_inflight_prompts="$NUM_PROMPTS"
    async_rl.max_buffered_rollouts="$NUM_PROMPTS"
    ++async_rl.rollout_failure.nemo_gym.rollout_timeout_s=300
    ++async_rl.rollout_failure.nemo_gym.max_row_attempts=1
    ++async_rl.stall_watchdog.interval_s=10
    ++async_rl.stall_watchdog.stall_timeout_s=600
    ++async_rl.stall_watchdog.stall_action=abort
    grpo.num_prompts_per_step="$NUM_PROMPTS"
    grpo.num_generations_per_prompt="$NUM_GENERATIONS"
    grpo.max_num_steps=1
    policy.max_total_sequence_length="$MAX_TOTAL_SEQUENCE_LENGTH"
    policy.generation.max_new_tokens="$MIN_GENERATION_TOKENS"
    policy.train_global_batch_size="$TRAIN_GLOBAL_BATCH_SIZE"
    policy.generation.temperature=1.0
    "env.nemo_gym.config_paths=$GYM_CONFIG_PATHS"
    '~env.nemo_gym.code_gen'
)

echo "=== Phase 1: checkpoint a non-empty active generation prefix ==="
command -v setsid >/dev/null
setsid env \
    "${CHECKPOINT_TEST_ENV[@]}" \
    SC_TEST_ENTRYPOINT="$RECOVERY_HOOK" \
    SC_SIBLING_RECOVERY_TEST_EVENTS="$PHASE1_EVENTS" \
    RUN_CONVERGENCE_CHECKS=0 \
    NEMO_GYM_SOURCE_DIR="$GYM_ROOT" \
    NEMO_GYM_CHECKPOINT_CONTROL_TOKEN="$NEMO_GYM_CHECKPOINT_CONTROL_TOKEN" \
    bash "$BASE_TEST" "${COMMON_OVERRIDES[@]}" "$@" &
PHASE1_PID=$!

uv run --directory "$PROJECT_ROOT" --no-sync python "$SNAPSHOT_HELPER" select \
    "$CHECKPOINT_DIR" \
    "$SELECTION_FILE" \
    "$PHASE1_PID" \
    "$BASE_RUN_LOG" \
    "$SNAPSHOT_TIMEOUT_S" \
    "$MIN_GENERATION_TOKENS" \
    --profile "$PROFILE"

stop_phase1
cp "$BASE_RUN_LOG" "$PHASE1_LOG"

SNAPSHOT_DIR=$(uv run --directory "$PROJECT_ROOT" --no-sync python -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["snapshot_path"])' \
    "$SELECTION_FILE")
SNAPSHOT_ROOT=$(dirname "$SNAPSHOT_DIR")

# Recover the exact cut chosen by the verifier. Later snapshots and trainer
# checkpoints represent work performed after the simulated crash point.
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

# Phase 1 checkpoints aggressively to catch an active generation. Give the
# restored request enough time to finish before another periodic cut begins.
for index in "${!COMMON_OVERRIDES[@]}"; do
    if [[ "${COMMON_OVERRIDES[$index]}" == ++rollout_checkpointing.snapshot_attempt_interval_s=* ]]; then
        COMMON_OVERRIDES[$index]="++rollout_checkpointing.snapshot_attempt_interval_s=$PHASE2_SNAPSHOT_INTERVAL_S"
    fi
done

echo "=== Phase 2: restore the prefix and generate only its remaining tail ==="
timeout --signal=TERM --kill-after=30s "${PHASE2_TIMEOUT_S}s" \
    env \
        "${CHECKPOINT_TEST_ENV[@]}" \
        SC_TEST_ENTRYPOINT="$RECOVERY_HOOK" \
        SC_SIBLING_RECOVERY_TEST_EVENTS="$PHASE2_EVENTS" \
        RUN_CONVERGENCE_CHECKS=0 \
        NEMO_GYM_SOURCE_DIR="$GYM_ROOT" \
        NEMO_GYM_CHECKPOINT_CONTROL_TOKEN="$NEMO_GYM_CHECKPOINT_CONTROL_TOKEN" \
        bash "$BASE_TEST" "${COMMON_OVERRIDES[@]}" "$@"
cp "$BASE_RUN_LOG" "$PHASE2_LOG"

grep -Fq "Selected rollout recovery snapshot: $SNAPSHOT_DIR" "$PHASE2_LOG"
grep -q "Native TQ checkpoint restored and validated" "$PHASE2_LOG"
grep -q \
    "Gym participant checkpoint restored and validated: .*components=resources_servers,responses_api_agents,responses_api_models" \
    "$PHASE2_LOG"
grep -q "train step 1/1" "$PHASE2_LOG"

uv run --directory "$PROJECT_ROOT" --no-sync python "$SNAPSHOT_HELPER" \
    verify-restore "$SELECTION_FILE" "$PHASE2_EVENTS" "$PHASE2_LOG" \
    --profile "$PROFILE" \
    --audit-events "$AUDIT_EVENTS"

echo "Single-controller Gym $PROFILE generation-prefix recovery functional test passed"
