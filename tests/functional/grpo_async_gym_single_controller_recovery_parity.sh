#!/bin/bash
# Compare an uninterrupted run with the same run recovered after two hard crashes.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
BASE_TEST=$SCRIPT_DIR/grpo_async_gym_single_controller.sh
BASE_RUN_LOG=$SCRIPT_DIR/grpo_async_gym_single_controller/run.log
RECOVERY_HOOK=$SCRIPT_DIR/_single_controller_sibling_recovery_hook.py
PARITY_HELPER=$SCRIPT_DIR/_gym_recovery_parity.py
TEST_DIR=$SCRIPT_DIR/grpo_async_gym_single_controller_recovery_parity
BASELINE_CHECKPOINT_DIR=$TEST_DIR/baseline-checkpoints
RECOVERY_CHECKPOINT_DIR=$TEST_DIR/recovery-checkpoints
BASELINE_LOG_DIR=$TEST_DIR/baseline-logs
RECOVERY_LOG_DIR=$TEST_DIR/recovery-logs
BASELINE_EVENTS=$TEST_DIR/baseline-events.jsonl
BASELINE_AUDIT=$TEST_DIR/baseline-resource-audit.jsonl
# Keep recovery stages separate: work completed after a selected snapshot but
# before the hard kill is intentionally replayed from that snapshot.
RECOVERY_EVENTS_STAGE1=$TEST_DIR/recovery-events-stage1.jsonl
RECOVERY_EVENTS_STAGE2=$TEST_DIR/recovery-events-stage2.jsonl
RECOVERY_EVENTS_STAGE3=$TEST_DIR/recovery-events-stage3.jsonl
RECOVERY_AUDIT_STAGE1=$TEST_DIR/recovery-resource-audit-stage1.jsonl
RECOVERY_AUDIT_STAGE2=$TEST_DIR/recovery-resource-audit-stage2.jsonl
RECOVERY_AUDIT_STAGE3=$TEST_DIR/recovery-resource-audit-stage3.jsonl
BASELINE_METRICS=$TEST_DIR/baseline-metrics.json
RECOVERY_METRICS=$TEST_DIR/recovery-metrics.json
ROLLOUT_TIMELINE=$TEST_DIR/rollout-timeline.json
FIRST_SELECTION=$TEST_DIR/workplace-prefix-cut.json
SECOND_SELECTION=$TEST_DIR/simple-agent-prefix-cut.json
TEST_DATA=$TEST_DIR/test-data.jsonl
ACTIVE_PID=""

GYM_ROOT=${NEMO_GYM_SOURCE_DIR:-$PROJECT_ROOT/3rdparty/Gym-workspace/Gym}
MAX_STEPS=${SC_GYM_RECOVERY_PARITY_STEPS:-5}
NUM_PROMPTS=${SC_GYM_RECOVERY_PARITY_PROMPTS_PER_STEP:-2}
NUM_GENERATIONS=${SC_GYM_RECOVERY_PARITY_GENERATIONS_PER_PROMPT:-2}
MIN_GENERATION_TOKENS=${SC_GYM_RECOVERY_PARITY_MIN_TOKENS:-256}
MAX_TOTAL_SEQUENCE_LENGTH=${SC_GYM_RECOVERY_PARITY_MAX_TOTAL_SEQUENCE_LENGTH:-2048}
CUT_INTERVAL_S=${SC_GYM_RECOVERY_PARITY_CUT_INTERVAL_S:-0.25}
FINAL_INTERVAL_S=${SC_GYM_RECOVERY_PARITY_FINAL_INTERVAL_S:-600}
CUT_TIMEOUT_S=${SC_GYM_RECOVERY_PARITY_CUT_TIMEOUT_S:-3600}
RUN_TIMEOUT_S=${SC_GYM_RECOVERY_PARITY_RUN_TIMEOUT_S:-7200}
REQUIRE_COMPLETION_ORDER_MATCH=${SC_GYM_RECOVERY_PARITY_REQUIRE_COMPLETION_ORDER_MATCH:-0}
TRAIN_GLOBAL_BATCH_SIZE=$((NUM_PROMPTS * NUM_GENERATIONS))

if [[ "$NUM_PROMPTS" -ne 2 ]]; then
    echo "[ERROR] This parity fixture requires two prompts per step (simple + Workplace)."
    exit 2
fi
if [[ "$MAX_STEPS" -lt 3 ]]; then
    echo "[ERROR] SC_GYM_RECOVERY_PARITY_STEPS must be at least 3."
    exit 2
fi
if [[ "$REQUIRE_COMPLETION_ORDER_MATCH" != "0" ]] && \
    [[ "$REQUIRE_COMPLETION_ORDER_MATCH" != "1" ]]; then
    echo "[ERROR] SC_GYM_RECOVERY_PARITY_REQUIRE_COMPLETION_ORDER_MATCH " \
        "must be 0 or 1."
    exit 2
fi
if [[ ! -f "$GYM_ROOT/nemo_gym/_checkpoint/model_control_contracts.py" ]]; then
    echo "[ERROR] $GYM_ROOT does not contain the Gym generation-prefix stack."
    echo "Set NEMO_GYM_SOURCE_DIR to the Gym token-prefix-recovery checkout."
    exit 2
fi

export NEMO_GYM_SOURCE_DIR=$GYM_ROOT
export NEMO_GYM_CHECKPOINT_CONTROL_TOKEN=${NEMO_GYM_CHECKPOINT_CONTROL_TOKEN:-functional-test-checkpoint-token}

rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

# Every step consumes one deterministic two-turn counter episode and one
# deterministic two-turn Workplace episode. The checkpoint test agent forces
# turn two to be a long, tool-free decode, making both agent types cuttable.
jq -n -c \
    --slurpfile simple "$GYM_ROOT/resources_servers/example_session_state_mgmt/data/example.jsonl" \
    --slurpfile workplace "$GYM_ROOT/resources_servers/workplace_assistant/data/example.jsonl" \
    --argjson steps "$MAX_STEPS" '
      range(0; $steps) as $step |
      (
        $simple[$step % ($simple | length)]
        | .id = (2 * $step)
        | .task_source = "example_session_state_mgmt_simple_agent"
        | .initial_count = (10 * $step)
        | .expected_count = (10 * $step + 1)
        | .responses_create_params.input = [{
            "role": "user",
            "content": ("Call increment_counter exactly once with count 1, then report the result. Case " + ($step | tostring))
          }]
        | .responses_create_params.tools = [
            .responses_create_params.tools[]
            | select(.name == "increment_counter")
          ]
        | .responses_create_params.tool_choice = {
            "type": "function",
            "name": "increment_counter"
          }
        | .responses_create_params.parallel_tool_calls = false
        | .responses_create_params.max_output_tokens = 64
      ),
      (
        $workplace[0]
        | .id = (2 * $step + 1)
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
      )
    ' > "$TEST_DATA"

export NEMO_GYM_TRAIN_DATA_PATH=$TEST_DATA
export NEMO_GYM_VALIDATION_DATA_PATH=$TEST_DATA

stop_active_run() {
    if [[ -z "$ACTIVE_PID" ]]; then
        return
    fi
    kill -KILL -- "-$ACTIVE_PID" 2>/dev/null || true
    wait "$ACTIVE_PID" 2>/dev/null || true
    ACTIVE_PID=""
}

cleanup() {
    local status=$?
    stop_active_run
    if [[ "$status" -eq 0 && "${SC_GYM_RECOVERY_PARITY_KEEP_ARTIFACTS:-0}" != "1" ]]; then
        rm -rf "$BASELINE_CHECKPOINT_DIR" "$RECOVERY_CHECKPOINT_DIR"
    else
        echo "Preserving recovery-parity artifacts for inspection: $TEST_DIR"
    fi
    return "$status"
}
trap cleanup EXIT

GYM_CONFIG_PATHS='[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,responses_api_agents/checkpoint_test_agent/configs/example_session_state_mgmt.yaml,responses_api_agents/checkpoint_test_agent/configs/workplace_assistant_prefix_recovery.yaml]'
COMMON_OVERRIDES=(
    checkpointing.enabled=true
    checkpointing.save_period=1
    checkpointing.metric_name=null
    +checkpointing.save_data_plane=true
    ++token_capture.enabled=true
    ++rollout_recovery.default_granularity=sibling
    ++rollout_checkpointing.keep_latest_k=16
    ++rollout_checkpointing.restore_mode=latest
    ++rollout_checkpointing.gym.capability_discovery_enabled=true
    ++rollout_checkpointing.gym.participant_checkpointing_enabled=true
    ++rollout_checkpointing.gym.generation_prefix_cuts_enabled=true
    ++rollout_checkpointing.gym.prepare_timeout_s=180
    async_rl.sampler.name=in_order
    async_rl.sampler.max_lookahead_versions=0
    async_rl.min_groups_for_streaming_train="$NUM_PROMPTS"
    async_rl.max_inflight_prompts="$NUM_PROMPTS"
    async_rl.max_buffered_rollouts="$NUM_PROMPTS"
    ++async_rl.rollout_failure.nemo_gym.rollout_timeout_s=600
    ++async_rl.stall_watchdog.interval_s=10
    ++async_rl.stall_watchdog.stall_timeout_s=900
    ++async_rl.stall_watchdog.stall_action=abort
    grpo.seed=1234
    grpo.num_prompts_per_step="$NUM_PROMPTS"
    grpo.num_generations_per_prompt="$NUM_GENERATIONS"
    grpo.max_num_steps="$MAX_STEPS"
    policy.max_total_sequence_length="$MAX_TOTAL_SEQUENCE_LENGTH"
    policy.generation.max_new_tokens="$MIN_GENERATION_TOKENS"
    policy.train_global_batch_size="$TRAIN_GLOBAL_BATCH_SIZE"
    # Keep train-time logit scaling finite while making generation effectively
    # greedy, including the tail regenerated after a restored prefix.
    policy.generation.temperature=1.0
    policy.generation.top_p=0.000001
    "env.nemo_gym.config_paths=$GYM_CONFIG_PATHS"
    env.should_log_nemo_gym_responses=false
    '~env.nemo_gym.code_gen'
)

run_environment=(
    NEMO_GYM_TEST_WORKPLACE_PREFIX_AFTER_MUTATION=1
    NEMO_GYM_TEST_PREFIX_MIN_TOKENS="$MIN_GENERATION_TOKENS"
    NEMO_GYM_SOURCE_DIR="$GYM_ROOT"
    NEMO_GYM_CHECKPOINT_CONTROL_TOKEN="$NEMO_GYM_CHECKPOINT_CONTROL_TOKEN"
    SC_TEST_ENTRYPOINT="$RECOVERY_HOOK"
    RUN_CONVERGENCE_CHECKS=0
)

echo "=== Recovery stage 1/3: cut a Workplace second-turn prefix ==="
command -v setsid >/dev/null
setsid env \
    "${run_environment[@]}" \
    NEMO_GYM_CHECKPOINT_TEST_EVENTS="$RECOVERY_AUDIT_STAGE1" \
    SC_SIBLING_RECOVERY_TEST_EVENTS="$RECOVERY_EVENTS_STAGE1" \
    bash "$BASE_TEST" \
    "${COMMON_OVERRIDES[@]}" \
    checkpointing.checkpoint_dir="$RECOVERY_CHECKPOINT_DIR" \
    ++rollout_checkpointing.snapshot_attempt_interval_s="$CUT_INTERVAL_S" \
    ++env.nemo_gym.nemo_gym_log_dir="$TEST_DIR/recovery-gym-logs" \
    logger.log_dir="$RECOVERY_LOG_DIR" \
    "$@" &
ACTIVE_PID=$!

uv run --directory "$PROJECT_ROOT" --no-sync python "$PARITY_HELPER" select-cut \
    "$RECOVERY_CHECKPOINT_DIR" "$FIRST_SELECTION" "$ACTIVE_PID" \
    "$BASE_RUN_LOG" "$CUT_TIMEOUT_S" 0 \
    workplace_assistant_prefix_checkpoint_test_agent "$MIN_GENERATION_TOKENS"
stop_active_run
cp "$BASE_RUN_LOG" "$TEST_DIR/recovery-crash-1.log"
uv run --directory "$PROJECT_ROOT" --no-sync python "$PARITY_HELPER" \
    prune-to-selection "$RECOVERY_CHECKPOINT_DIR" "$FIRST_SELECTION"

echo "=== Recovery stage 2/3: restore, train two steps, cut the simple agent ==="
setsid env \
    "${run_environment[@]}" \
    NEMO_GYM_CHECKPOINT_TEST_EVENTS="$RECOVERY_AUDIT_STAGE2" \
    SC_SIBLING_RECOVERY_TEST_EVENTS="$RECOVERY_EVENTS_STAGE2" \
    bash "$BASE_TEST" \
    "${COMMON_OVERRIDES[@]}" \
    checkpointing.checkpoint_dir="$RECOVERY_CHECKPOINT_DIR" \
    ++rollout_checkpointing.snapshot_attempt_interval_s="$CUT_INTERVAL_S" \
    ++env.nemo_gym.nemo_gym_log_dir="$TEST_DIR/recovery-gym-logs" \
    logger.log_dir="$RECOVERY_LOG_DIR" \
    "$@" &
ACTIVE_PID=$!

uv run --directory "$PROJECT_ROOT" --no-sync python "$PARITY_HELPER" select-cut \
    "$RECOVERY_CHECKPOINT_DIR" "$SECOND_SELECTION" "$ACTIVE_PID" \
    "$BASE_RUN_LOG" "$CUT_TIMEOUT_S" 2 \
    example_session_state_mgmt_simple_agent "$MIN_GENERATION_TOKENS"
stop_active_run
cp "$BASE_RUN_LOG" "$TEST_DIR/recovery-crash-2.log"
uv run --directory "$PROJECT_ROOT" --no-sync python "$PARITY_HELPER" \
    prune-to-selection "$RECOVERY_CHECKPOINT_DIR" "$SECOND_SELECTION"

echo "=== Recovery stage 3/3: restore again and finish all $MAX_STEPS steps ==="
timeout --signal=TERM --kill-after=30s "${RUN_TIMEOUT_S}s" \
    env \
        "${run_environment[@]}" \
        NEMO_GYM_CHECKPOINT_TEST_EVENTS="$RECOVERY_AUDIT_STAGE3" \
        SC_SIBLING_RECOVERY_TEST_EVENTS="$RECOVERY_EVENTS_STAGE3" \
        bash "$BASE_TEST" \
        "${COMMON_OVERRIDES[@]}" \
        checkpointing.checkpoint_dir="$RECOVERY_CHECKPOINT_DIR" \
        ++rollout_checkpointing.snapshot_attempt_interval_s="$FINAL_INTERVAL_S" \
        ++env.nemo_gym.nemo_gym_log_dir="$TEST_DIR/recovery-gym-logs" \
        logger.log_dir="$RECOVERY_LOG_DIR" \
        "$@"
cp "$BASE_RUN_LOG" "$TEST_DIR/recovery-final.log"

restored_prefixes=0
restarted_prefixes=0
for selection_and_log in \
    "$FIRST_SELECTION:$TEST_DIR/recovery-crash-2.log" \
    "$SECOND_SELECTION:$TEST_DIR/recovery-final.log"; do
    selection=${selection_and_log%%:*}
    run_log=${selection_and_log#*:}
    source_model_call_id=$(jq -r .model_call_id "$selection")
    restored_pattern="generation prefix restored: .*source_model_call_id=$source_model_call_id "
    completed_pattern="generation prefix completed: .*source_model_call_id=$source_model_call_id "
    restarted_pattern="generation prefix restart: .*source_model_call_id=$source_model_call_id "
    restored=0
    completed=0
    restarted=0
    grep -Eq "$restored_pattern" "$run_log" && restored=1
    grep -Eq "$completed_pattern" "$run_log" && completed=1
    grep -Eq "$restarted_pattern" "$run_log" && restarted=1

    if [[ "$restored" -eq 1 && "$completed" -eq 1 && "$restarted" -eq 0 ]]; then
        restored_prefixes=$((restored_prefixes + 1))
        continue
    fi
    if [[ "$restored" -eq 0 && "$completed" -eq 0 && "$restarted" -eq 1 ]]; then
        restarted_prefixes=$((restarted_prefixes + 1))
        continue
    fi
    echo "[ERROR] selected cut has an inconsistent recovery outcome: " \
        "source_model_call_id=$source_model_call_id restored=$restored " \
        "completed=$completed restarted=$restarted log=$run_log"
    exit 1
done
if [[ "$restored_prefixes" -lt 1 ]]; then
    echo "[ERROR] recovery parity test did not restore any safe generation prefix"
    exit 1
fi
if [[ "$restarted_prefixes" -lt 1 ]]; then
    echo "[ERROR] recovery parity test did not exercise structured-prefix fallback"
    exit 1
fi
grep -q "train step $MAX_STEPS/$MAX_STEPS" "$TEST_DIR/recovery-final.log"

echo "=== Reference: run $MAX_STEPS uninterrupted steps ==="
timeout --signal=TERM --kill-after=30s "${RUN_TIMEOUT_S}s" \
    env \
        "${run_environment[@]}" \
        NEMO_GYM_CHECKPOINT_TEST_EVENTS="$BASELINE_AUDIT" \
        SC_SIBLING_RECOVERY_TEST_EVENTS="$BASELINE_EVENTS" \
        bash "$BASE_TEST" \
        "${COMMON_OVERRIDES[@]}" \
        checkpointing.checkpoint_dir="$BASELINE_CHECKPOINT_DIR" \
        ++rollout_checkpointing.snapshot_attempt_interval_s="$FINAL_INTERVAL_S" \
        ++env.nemo_gym.nemo_gym_log_dir="$TEST_DIR/baseline-gym-logs" \
        logger.log_dir="$BASELINE_LOG_DIR" \
        "$@"
cp "$BASE_RUN_LOG" "$TEST_DIR/baseline-run.log"

uv run --directory "$PROJECT_ROOT" --no-sync python tests/json_dump_tb_logs.py \
    "$BASELINE_LOG_DIR" --output_path "$BASELINE_METRICS"
uv run --directory "$PROJECT_ROOT" --no-sync python tests/json_dump_tb_logs.py \
    "$RECOVERY_LOG_DIR" --output_path "$RECOVERY_METRICS"

completion_order_args=()
if [[ "$REQUIRE_COMPLETION_ORDER_MATCH" == "1" ]]; then
    completion_order_args+=(--require-completion-order-match)
fi

uv run --directory "$PROJECT_ROOT" --no-sync python "$PARITY_HELPER" compare \
    --baseline-events "$BASELINE_EVENTS" \
    --recovery-events "$RECOVERY_EVENTS_STAGE1" \
    --recovery-events "$RECOVERY_EVENTS_STAGE2" \
    --recovery-events "$RECOVERY_EVENTS_STAGE3" \
    --baseline-log-dir "$BASELINE_LOG_DIR" \
    --recovery-log-dir "$RECOVERY_LOG_DIR" \
    --baseline-metrics "$BASELINE_METRICS" \
    --recovery-metrics "$RECOVERY_METRICS" \
    --baseline-audit "$BASELINE_AUDIT" \
    --recovery-audit "$RECOVERY_AUDIT_STAGE1" \
    --recovery-audit "$RECOVERY_AUDIT_STAGE2" \
    --recovery-audit "$RECOVERY_AUDIT_STAGE3" \
    --timeline-output "$ROLLOUT_TIMELINE" \
    --steps "$MAX_STEPS" \
    --prompts-per-step "$NUM_PROMPTS" \
    --generations-per-prompt "$NUM_GENERATIONS" \
    --required-retried-task-source example_session_state_mgmt_simple_agent \
    --required-retried-task-source workplace_assistant_prefix_checkpoint_test_agent \
    "${completion_order_args[@]}"

echo "Single-controller Gym uninterrupted-vs-multi-crash recovery parity test passed"
