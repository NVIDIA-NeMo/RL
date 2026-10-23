#!/bin/bash
# Two-process functional coverage for a Gym-v2 checkpoint taken during active
# vLLM decode. The restored attempt must reuse the exact durable prefix and
# generate only the remaining suffix before the rollout is finalized/trained.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
BASE_TEST=$SCRIPT_DIR/grpo_async_gym_single_controller.sh
BASE_RUN_LOG=$SCRIPT_DIR/grpo_async_gym_single_controller/run.log
TEST_CONFIG=${SC_PREFIX_RECOVERY_TEST_CONFIG:-$PROJECT_ROOT/examples/nemo_gym/grpo_qwen3_30ba3b_instruct.yaml}
RECOVERY_HOOK=$SCRIPT_DIR/_single_controller_turn_recovery_hook.py
SNAPSHOT_HELPER=$SCRIPT_DIR/_gym_prefix_recovery_snapshot.py
TEST_DIR=$SCRIPT_DIR/grpo_async_gym_single_controller_prefix_recovery
CHECKPOINT_DIR=$TEST_DIR/checkpoints
PHASE1_LOG=$TEST_DIR/phase1.log
PHASE2_LOG=$TEST_DIR/phase2.log
PHASE1_EVENTS=$TEST_DIR/phase1-events.jsonl
PHASE2_EVENTS=$TEST_DIR/phase2-events.jsonl
SELECTION_FILE=$TEST_DIR/selected-snapshot.json
SELECTED_BACKUP=$TEST_DIR/selected-snapshot
TEST_DATA=$TEST_DIR/test-data.jsonl
GYM_ROOT=$PROJECT_ROOT/3rdparty/Gym-workspace/Gym
PHASE1_PID=""

NUM_PROMPTS=${SC_PREFIX_RECOVERY_NUM_PROMPTS:-2}
NUM_GENERATIONS=${SC_PREFIX_RECOVERY_NUM_GENERATIONS:-2}
MIN_GENERATION_TOKENS=${SC_PREFIX_RECOVERY_MIN_TOKENS:-4096}
MAX_TOTAL_SEQUENCE_LENGTH=${SC_PREFIX_RECOVERY_MAX_TOTAL_SEQUENCE_LENGTH:-8192}
SNAPSHOT_INTERVAL_S=${SC_PREFIX_RECOVERY_INTERVAL_S:-0.05}
PHASE2_SNAPSHOT_INTERVAL_S=${SC_PREFIX_RECOVERY_PHASE2_INTERVAL_S:-600}
SNAPSHOT_TIMEOUT_S=${SC_PREFIX_RECOVERY_TIMEOUT_S:-2400}
PHASE2_TIMEOUT_S=${SC_PREFIX_RECOVERY_PHASE2_TIMEOUT_S:-2400}
MAX_STEPS=${SC_PREFIX_RECOVERY_MAX_STEPS:-1}
TRAIN_GLOBAL_BATCH_SIZE=$((NUM_PROMPTS * NUM_GENERATIONS))

rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

# A long, tool-free first policy call isolates active decode recovery. min_tokens
# keeps the request alive until a periodic checkpoint can cut a non-empty,
# nonterminal prefix.
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
        rm -rf "$CHECKPOINT_DIR" "$SELECTED_BACKUP"
    else
        echo "Preserving prefix-recovery artifacts for inspection: $TEST_DIR"
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
    grpo.num_prompts_per_step="$NUM_PROMPTS"
    grpo.num_generations_per_prompt="$NUM_GENERATIONS"
    grpo.max_num_steps="$MAX_STEPS"
    policy.max_total_sequence_length="$MAX_TOTAL_SEQUENCE_LENGTH"
    policy.generation.max_new_tokens="$MIN_GENERATION_TOKENS"
    policy.train_global_batch_size="$TRAIN_GLOBAL_BATCH_SIZE"
    policy.generation.temperature=1.0
    '~env.nemo_gym.code_gen'
    "env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/example_session_state_mgmt/configs/example_session_state_mgmt.yaml]"
    data.train.data_path="$TEST_DATA"
    data.validation.data_path="$TEST_DATA"
)

echo "=== Phase 1: publish a checkpoint containing an active generation prefix ==="
command -v setsid >/dev/null
setsid env \
    SC_TEST_ENTRYPOINT="$RECOVERY_HOOK" \
    SC_TEST_CONFIG="$TEST_CONFIG" \
    SC_GYM_RECOVERY_TEST_EVENTS="$PHASE1_EVENTS" \
    RUN_CONVERGENCE_CHECKS=0 \
    bash "$BASE_TEST" "${COMMON_OVERRIDES[@]}" "$@" &
PHASE1_PID=$!

uv run --directory "$PROJECT_ROOT" --no-sync python "$SNAPSHOT_HELPER" select \
    "$CHECKPOINT_DIR/bootstrap/rollout_snapshots" \
    "$SELECTION_FILE" \
    "$PHASE1_PID" \
    "$BASE_RUN_LOG" \
    "$SNAPSHOT_TIMEOUT_S" \
    "$SELECTED_BACKUP"

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
        SC_TEST_ENTRYPOINT="$RECOVERY_HOOK" \
        SC_TEST_CONFIG="$TEST_CONFIG" \
        SC_GYM_RECOVERY_TEST_EVENTS="$PHASE2_EVENTS" \
        RUN_CONVERGENCE_CHECKS=0 \
    bash "$BASE_TEST" "${COMMON_OVERRIDES[@]}" "$@"
cp "$BASE_RUN_LOG" "$PHASE2_LOG"

grep -Fq "Selected rollout recovery snapshot: $SNAPSHOT_DIR" "$PHASE2_LOG"
grep -q "Native TQ checkpoint restored and validated" "$PHASE2_LOG"
grep -q "Loaded .* unfinished rollout group(s)" "$PHASE2_LOG"
grep -q "generation prefix restored:" "$PHASE2_LOG"
grep -q "train step $MAX_STEPS/$MAX_STEPS" "$PHASE2_LOG"

PHASE2_METRICS=$TEST_DIR/phase2-metrics.json
uv run --directory "$PROJECT_ROOT" --no-sync tests/json_dump_tb_logs.py \
    "$SCRIPT_DIR/grpo_async_gym_single_controller/logs" \
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

echo "Single-controller Gym generation-prefix recovery functional test passed"
