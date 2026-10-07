#!/bin/bash
# Functional parity coverage for a Gym-v2 checkpoint taken during active vLLM
# decode. It compares an uninterrupted reference run with a hard-kill/restore
# run, where the restored attempt must reuse the exact durable prefix and
# generate only the remaining suffix before the rollout is finalized/trained.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
BASE_TEST=$SCRIPT_DIR/grpo_async_gym_single_controller.sh
BASE_RUN_LOG=$SCRIPT_DIR/grpo_async_gym_single_controller/run.log
TEST_CONFIG=${SC_PREFIX_RECOVERY_TEST_CONFIG:-$PROJECT_ROOT/examples/nemo_gym/grpo_qwen3_30ba3b_instruct.yaml}
RECOVERY_HOOK=$SCRIPT_DIR/_single_controller_turn_recovery_hook.py
SNAPSHOT_HELPER=$SCRIPT_DIR/_gym_prefix_recovery_snapshot.py
PARITY_HELPER=$SCRIPT_DIR/_gym_prefix_recovery_parity.py
TEST_DIR=$SCRIPT_DIR/grpo_async_gym_single_controller_prefix_recovery
CHECKPOINT_DIR=$TEST_DIR/recovery-checkpoints
BASELINE_CHECKPOINT_DIR=$TEST_DIR/baseline-checkpoints
PHASE1_LOG=$TEST_DIR/phase1.log
PHASE2_LOG=$TEST_DIR/phase2.log
BASELINE_LOG=$TEST_DIR/baseline.log
PHASE1_EVENTS=$TEST_DIR/phase1-events.jsonl
PHASE2_EVENTS=$TEST_DIR/phase2-events.jsonl
BASELINE_EVENTS=$TEST_DIR/baseline-events.jsonl
BASELINE_TRAINING=$TEST_DIR/baseline-training.jsonl
DISCARDED_TRAINING=$TEST_DIR/discarded-phase1-training.jsonl
RECOVERY_TRAINING=$TEST_DIR/recovery-training.jsonl
BASELINE_LOG_DIR=$TEST_DIR/baseline-logs
RECOVERY_LOG_DIR=$TEST_DIR/recovery-logs
BASELINE_METRICS=$TEST_DIR/baseline-metrics.json
PHASE2_METRICS=$TEST_DIR/phase2-metrics.json
PARITY_REPORT=$TEST_DIR/parity-report.json
SELECTION_FILE=$TEST_DIR/selected-snapshot.json
SELECTED_BACKUP=$TEST_DIR/selected-snapshot
TEST_DATA=$TEST_DIR/test-data.jsonl
GYM_ROOT=$PROJECT_ROOT/3rdparty/Gym-workspace/Gym
PHASE1_PID=""
SENTINEL_EVENT="NeMo RL prefix recovery parity sentinel"

# Cuttable Workplace prompts per step, one per list length.
# Above ~500 the model overruns its list (stray tags, a second list) and hits a
# near-tied token where runs diverge; keep every list in the well-behaved range.
LIST_LENGTHS=(${SC_PREFIX_RECOVERY_LIST_LENGTHS:-150 250 350 450})
NUM_PROMPTS=${#LIST_LENGTHS[@]}
NUM_GENERATIONS=${SC_PREFIX_RECOVERY_NUM_GENERATIONS:-2}
# Qwen tokenizes each digit separately, so "N\n" costs at most digits + 1 tokens.
LIST_TOKENS_PER_ITEM=6
LIST_OUTPUT_SLACK_TOKENS=256
# Qwen3-0.6B answers the closing turn with a few tokens instead of the list.
MODEL_NAME=${SC_PREFIX_RECOVERY_MODEL:-Qwen/Qwen3-1.7B}
MAX_TOTAL_SEQUENCE_LENGTH=${SC_PREFIX_RECOVERY_MAX_TOTAL_SEQUENCE_LENGTH:-8192}
SNAPSHOT_INTERVAL_S=${SC_PREFIX_RECOVERY_INTERVAL_S:-0.05}
PHASE2_SNAPSHOT_INTERVAL_S=${SC_PREFIX_RECOVERY_PHASE2_INTERVAL_S:-600}
SNAPSHOT_TIMEOUT_S=${SC_PREFIX_RECOVERY_TIMEOUT_S:-2400}
PHASE2_TIMEOUT_S=${SC_PREFIX_RECOVERY_PHASE2_TIMEOUT_S:-2400}
MAX_STEPS=${SC_PREFIX_RECOVERY_MAX_STEPS:-3}
TRAIN_GLOBAL_BATCH_SIZE=$((NUM_PROMPTS * NUM_GENERATIONS))
# Gym refuses dispatches that land during a checkpoint and those rollouts start
# later, so require a recoverable prefix in every Workplace prompt group rather
# than in every sibling at once.
MIN_PREFIX_CUT_GROUPS=${#LIST_LENGTHS[@]}
POLICY_MAX_NEW_TOKENS=0
for list_length in "${LIST_LENGTHS[@]}"; do
    list_budget=$((list_length * LIST_TOKENS_PER_ITEM + LIST_OUTPUT_SLACK_TOKENS))
    if [[ "$list_budget" -gt "$POLICY_MAX_NEW_TOKENS" ]]; then
        POLICY_MAX_NEW_TOKENS=$list_budget
    fi
done
LIST_LENGTHS_JSON=$(printf '%s\n' "${LIST_LENGTHS[@]}" | jq -s -c 'map(tonumber)')

rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

# Each step has one Workplace prompt group per LIST_LENGTHS entry. Each episode
# makes a named calendar_create_event call, then closes by listing the integers
# 1..N. The checkpoint test agent switches the closing turn to tool_choice none
# without forcing a length, so that call is unconstrained (cuttable mid-decode),
# ends at its natural EOS, and the groups finish in an order fixed by N rather
# than by scheduling noise.
# The second Workplace prompt expects a different duration than requested, so
# its reward is 0 and reward parity is not satisfied by a constant.
jq -n -c \
    --slurpfile workplace "$GYM_ROOT/resources_servers/workplace_assistant/data/example.jsonl" \
    --argjson steps "$MAX_STEPS" \
    --argjson list_lengths "$LIST_LENGTHS_JSON" \
    --argjson tokens_per_item "$LIST_TOKENS_PER_ITEM" \
    --argjson output_slack "$LIST_OUTPUT_SLACK_TOKENS" \
    --arg event "$SENTINEL_EVENT" '
      range(0; $steps) as $step |
        range(0; $list_lengths | length) as $slot
        | $list_lengths[$slot] as $list_length
        | $workplace[0]
        | del(.agent_ref)
        | .task_source = "workplace_assistant_checkpoint_test_agent"
        | .responses_create_params.input = [{
            "role": "user",
            "content": ("Call calendar_create_event exactly once with event_name " + $event
                + ", participant_email checkpoint-recovery@example.com, event_start "
                + "2025-01-15 10:00:00, and duration 30. After the tool result, do not call any "
                + "more tools: list every integer from 1 to " + ($list_length | tostring)
                + " in increasing order, one per line, with no other text.")
          }]
        | .responses_create_params.tools = [
            .responses_create_params.tools[] | select(.name == "calendar_create_event")
          ]
        | .responses_create_params.tool_choice = {"type": "function", "name": "calendar_create_event"}
        | .responses_create_params.parallel_tool_calls = false
        | .responses_create_params.max_output_tokens = ($list_length * $tokens_per_item + $output_slack)
        | .ground_truth = [{
            "name": "calendar_create_event",
            "arguments": ({
              "event_name": $event,
              "participant_email": "checkpoint-recovery@example.com",
              "event_start": "2025-01-15 10:00:00",
              "duration": (if $slot == 1 then "60" else "30" end)
            } | tojson)
          }]
        | .category = "workplace_assistant_calendar"
        | .environment_name = "workplace_assistant"
    ' > "$TEST_DATA"

# MIN_TOKENS=0 stops further tool calls on the closing turn without forcing its
# length: forced min_tokens suppress EOS and leave near-tied argmax tokens that
# flip between otherwise identical runs.
WORKPLACE_PREFIX_ENV=(
    NEMO_GYM_TEST_WORKPLACE_PREFIX_AFTER_MUTATION=1
    NEMO_GYM_TEST_PREFIX_MIN_TOKENS=0
)

stop_phase1() {
    if [[ -z "$PHASE1_PID" ]]; then
        return
    fi
    kill -KILL -- "-$PHASE1_PID" 2>/dev/null || true
    wait "$PHASE1_PID" 2>/dev/null || true
    PHASE1_PID=""
    # Ray actors outlive the driver until their raylet notices the crash. Do
    # not start phase two while the old controller can still write snapshots.
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
    if [[ "$status" -eq 0 && "${SC_PREFIX_RECOVERY_KEEP_CHECKPOINTS:-0}" != "1" ]]; then
        rm -rf "$CHECKPOINT_DIR" "$BASELINE_CHECKPOINT_DIR" "$SELECTED_BACKUP"
    else
        echo "Preserving prefix-recovery artifacts for inspection: $TEST_DIR"
    fi
    return "$status"
}
trap cleanup EXIT

COMMON_OVERRIDES=(
    policy.model_name="$MODEL_NAME"
    policy.tokenizer.name="$MODEL_NAME"
    checkpointing.enabled=true
    checkpointing.checkpoint_dir="$CHECKPOINT_DIR"
    checkpointing.metric_name=null
    checkpointing.save_period=1
    ++checkpointing.save_data_plane=true
    ++token_capture.enabled=true
    ++rollout_recovery.target_level=prefix
    ++rollout_checkpointing.snapshot_attempt_interval_s="$SNAPSHOT_INTERVAL_S"
    ++rollout_checkpointing.keep_latest_k=16
    ++rollout_checkpointing.restore_mode=latest
    async_rl.sampler.name=in_order
    async_rl.sampler.max_lookahead_versions=0
    async_rl.min_groups_for_streaming_train="$NUM_PROMPTS"
    async_rl.max_inflight_prompts="$NUM_PROMPTS"
    async_rl.max_buffered_rollouts="$NUM_PROMPTS"
    ++async_rl.rollout_failure.nemo_gym.rollout_timeout_s=300
    ++async_rl.stall_watchdog.interval_s=10
    ++async_rl.stall_watchdog.stall_timeout_s=600
    ++async_rl.stall_watchdog.stall_action=abort
    grpo.seed=1234
    grpo.num_prompts_per_step="$NUM_PROMPTS"
    grpo.num_generations_per_prompt="$NUM_GENERATIONS"
    grpo.max_num_steps="$MAX_STEPS"
    policy.max_total_sequence_length="$MAX_TOTAL_SEQUENCE_LENGTH"
    policy.generation.max_new_tokens="$POLICY_MAX_NEW_TOKENS"
    policy.train_global_batch_size="$TRAIN_GLOBAL_BATCH_SIZE"
    # Make the suffix regenerated after recovery deterministic enough to
    # compare token-for-token with the uninterrupted reference run.
    policy.generation.temperature=1.0
    policy.generation.top_p=0.000001
    '~env.nemo_gym.code_gen'
    "env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,responses_api_agents/checkpoint_test_agent/configs/workplace_assistant.yaml]"
    # Keep each step's prompt membership fixed across runs.
    data.shuffle=false
    data.train.data_path="$TEST_DATA"
    data.validation.data_path="$TEST_DATA"
)

echo "=== Reference: run the same workload without interruption ==="
timeout --signal=TERM --kill-after=30s "${PHASE2_TIMEOUT_S}s" \
    env \
        "${WORKPLACE_PREFIX_ENV[@]}" \
        SC_TEST_ENTRYPOINT="$RECOVERY_HOOK" \
        SC_TEST_CONFIG="$TEST_CONFIG" \
        SC_GYM_RECOVERY_TEST_EVENTS="$BASELINE_EVENTS" \
        SC_PREFIX_RECOVERY_TRAINING_PAYLOAD="$BASELINE_TRAINING" \
        RUN_CONVERGENCE_CHECKS=0 \
    bash "$BASE_TEST" \
        "${COMMON_OVERRIDES[@]}" \
        checkpointing.checkpoint_dir="$BASELINE_CHECKPOINT_DIR" \
        ++rollout_checkpointing.snapshot_attempt_interval_s="$PHASE2_SNAPSHOT_INTERVAL_S" \
        logger.log_dir="$BASELINE_LOG_DIR" \
        "$@"
cp "$BASE_RUN_LOG" "$BASELINE_LOG"
uv run --directory "$PROJECT_ROOT" --no-sync tests/json_dump_tb_logs.py \
    "$BASELINE_LOG_DIR" \
    --output_path "$BASELINE_METRICS"

echo "=== Phase 1: publish a checkpoint containing an active generation prefix ==="
command -v setsid >/dev/null
setsid env \
    "${WORKPLACE_PREFIX_ENV[@]}" \
    SC_TEST_ENTRYPOINT="$RECOVERY_HOOK" \
    SC_TEST_CONFIG="$TEST_CONFIG" \
    SC_GYM_RECOVERY_TEST_EVENTS="$PHASE1_EVENTS" \
    SC_PREFIX_RECOVERY_TRAINING_PAYLOAD="$DISCARDED_TRAINING" \
    RUN_CONVERGENCE_CHECKS=0 \
    bash "$BASE_TEST" \
        "${COMMON_OVERRIDES[@]}" \
        logger.log_dir="$RECOVERY_LOG_DIR" \
        "$@" &
PHASE1_PID=$!

uv run --directory "$PROJECT_ROOT" --no-sync python "$SNAPSHOT_HELPER" select \
    "$CHECKPOINT_DIR/bootstrap/rollout_snapshots" \
    "$SELECTION_FILE" \
    "$PHASE1_PID" \
    "$BASE_RUN_LOG" \
    "$SNAPSHOT_TIMEOUT_S" \
    "$SELECTED_BACKUP" \
    "$SENTINEL_EVENT" \
    --min-cut-groups "$MIN_PREFIX_CUT_GROUPS"

stop_phase1
cp "$BASE_RUN_LOG" "$PHASE1_LOG"
SNAPSHOT_DIR=$(uv run --directory "$PROJECT_ROOT" --no-sync python -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["snapshot"])' \
    "$SELECTION_FILE")
SNAPSHOT_ROOT=$(dirname "$SNAPSHOT_DIR")

# Recover exactly the prefix selected above. A newer periodic snapshot or a
# trainer checkpoint would represent work performed after the simulated crash.
rm -rf "$SNAPSHOT_ROOT"
mkdir -p "$SNAPSHOT_ROOT"
cp -a "$SELECTED_BACKUP" "$SNAPSHOT_DIR"
for trainer_checkpoint in "$CHECKPOINT_DIR"/step_*; do
    if [[ -d "$trainer_checkpoint" ]]; then
        rm -rf "$trainer_checkpoint"
    fi
done

# Avoid cutting the replacement request again while verifying that it consumes
# the selected prefix and reaches normal completion.
for index in "${!COMMON_OVERRIDES[@]}"; do
    if [[ "${COMMON_OVERRIDES[$index]}" == ++rollout_checkpointing.snapshot_attempt_interval_s=* ]]; then
        COMMON_OVERRIDES[$index]="++rollout_checkpointing.snapshot_attempt_interval_s=$PHASE2_SNAPSHOT_INTERVAL_S"
    fi
done

echo "=== Phase 2: restore the prefix and generate its remaining suffix ==="
timeout --signal=TERM --kill-after=30s "${PHASE2_TIMEOUT_S}s" \
    env \
        "${WORKPLACE_PREFIX_ENV[@]}" \
        SC_TEST_ENTRYPOINT="$RECOVERY_HOOK" \
        SC_TEST_CONFIG="$TEST_CONFIG" \
        SC_GYM_RECOVERY_TEST_EVENTS="$PHASE2_EVENTS" \
        SC_PREFIX_RECOVERY_TRAINING_PAYLOAD="$RECOVERY_TRAINING" \
        RUN_CONVERGENCE_CHECKS=0 \
    bash "$BASE_TEST" \
        "${COMMON_OVERRIDES[@]}" \
        logger.log_dir="$RECOVERY_LOG_DIR" \
        "$@"
cp "$BASE_RUN_LOG" "$PHASE2_LOG"

grep -Fq "Selected rollout recovery snapshot: $SNAPSHOT_DIR" "$PHASE2_LOG"
grep -q "Native TQ checkpoint restored and validated" "$PHASE2_LOG"
grep -q "Loaded .* unfinished rollout group(s)" "$PHASE2_LOG"
grep -q "generation prefix restored:" "$PHASE2_LOG"
grep -q "train step $MAX_STEPS/$MAX_STEPS" "$PHASE2_LOG"

uv run --directory "$PROJECT_ROOT" --no-sync tests/json_dump_tb_logs.py \
    "$RECOVERY_LOG_DIR" \
    --output_path "$PHASE2_METRICS"
uv run --directory "$PROJECT_ROOT" --no-sync tests/check_metrics.py "$PHASE2_METRICS" \
    'max(data["train/finalize/invalid_row_rate"]) == 0' \
    'max(data["train/finalize/capture_poisoned_rollouts"]) == 0'

uv run --directory "$PROJECT_ROOT" --no-sync python "$SNAPSHOT_HELPER" verify-restore \
    "$SELECTION_FILE" \
    "$PHASE1_EVENTS" \
    "$PHASE2_EVENTS" \
    "$PHASE2_LOG" \
    "$CHECKPOINT_DIR/step_$MAX_STEPS/training_info.json" \
    "$MAX_STEPS"

uv run --directory "$PROJECT_ROOT" --no-sync python "$PARITY_HELPER" \
    --baseline-training "$BASELINE_TRAINING" \
    --recovery-training "$RECOVERY_TRAINING" \
    --baseline-metrics "$BASELINE_METRICS" \
    --recovery-metrics "$PHASE2_METRICS" \
    --selection "$SELECTION_FILE" \
    --recovery-log "$PHASE2_LOG" \
    --successor-checkpoint "$CHECKPOINT_DIR/step_$MAX_STEPS" \
    --steps "$MAX_STEPS" \
    --prompts-per-step "$NUM_PROMPTS" \
    --generations-per-prompt "$NUM_GENERATIONS" \
    --baseline-events "$BASELINE_EVENTS" \
    --phase1-events "$PHASE1_EVENTS" \
    --recovery-events "$PHASE2_EVENTS" \
    --report-output "$PARITY_REPORT" \
    --require-prompt-group-order

echo "Single-controller Gym Workplace generation-prefix recovery parity test passed"
