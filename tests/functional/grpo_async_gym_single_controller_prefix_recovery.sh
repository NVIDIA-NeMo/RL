#!/bin/bash
# Full-process recovery test for one model call cut during active vLLM decode.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
BASE_TEST=$SCRIPT_DIR/grpo_async_gym_single_controller.sh
BASE_RUN_LOG=$SCRIPT_DIR/grpo_async_gym_single_controller/run.log
RECOVERY_HOOK=$SCRIPT_DIR/_single_controller_sibling_recovery_hook.py
SNAPSHOT_HELPER=$SCRIPT_DIR/_gym_prefix_recovery_snapshot.py
TEST_DIR=$SCRIPT_DIR/grpo_async_gym_single_controller_prefix_recovery
CHECKPOINT_DIR=$TEST_DIR/checkpoints
PHASE1_LOG=$TEST_DIR/phase1.log
PHASE2_LOG=$TEST_DIR/phase2.log
PHASE1_EVENTS=$TEST_DIR/phase1-events.jsonl
PHASE2_EVENTS=$TEST_DIR/phase2-events.jsonl
SELECTION_FILE=$TEST_DIR/selected_snapshot.json
TEST_DATA=$TEST_DIR/test_data.jsonl
PHASE1_PID=""

GYM_ROOT=${NEMO_GYM_SOURCE_DIR:-$PROJECT_ROOT/3rdparty/Gym-workspace/Gym}
SNAPSHOT_INTERVAL_S=${SC_GYM_PREFIX_RECOVERY_INTERVAL_S:-1}
SNAPSHOT_TIMEOUT_S=${SC_GYM_PREFIX_RECOVERY_TIMEOUT_S:-2400}
PHASE2_TIMEOUT_S=${SC_GYM_PREFIX_RECOVERY_PHASE2_TIMEOUT_S:-2400}
NUM_PROMPTS=${SC_GYM_PREFIX_RECOVERY_NUM_PROMPTS:-2}
NUM_GENERATIONS=${SC_GYM_PREFIX_RECOVERY_NUM_GENERATIONS:-2}
MIN_GENERATION_TOKENS=${SC_GYM_PREFIX_RECOVERY_MIN_TOKENS:-384}
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

# Keep the first policy call alive long enough for a periodic checkpoint to cut
# a non-empty prefix. No tools are exposed: this test isolates model-call
# continuation from turn-level resource mutation recovery.
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
    ++async_rl.stall_watchdog.interval_s=10
    ++async_rl.stall_watchdog.stall_timeout_s=600
    ++async_rl.stall_watchdog.stall_action=abort
    grpo.num_prompts_per_step="$NUM_PROMPTS"
    grpo.num_generations_per_prompt="$NUM_GENERATIONS"
    grpo.max_num_steps=1
    policy.max_total_sequence_length=1024
    policy.generation.max_new_tokens="$MIN_GENERATION_TOKENS"
    policy.train_global_batch_size="$TRAIN_GLOBAL_BATCH_SIZE"
    policy.generation.temperature=1.0
    'env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,responses_api_agents/checkpoint_test_agent/configs/example_session_state_mgmt.yaml]'
    '~env.nemo_gym.code_gen'
)

echo "=== Phase 1: checkpoint a non-empty active generation prefix ==="
command -v setsid >/dev/null
setsid env \
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
    "$MIN_GENERATION_TOKENS"

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

echo "=== Phase 2: restore the prefix and generate only its remaining tail ==="
timeout --signal=TERM --kill-after=30s "${PHASE2_TIMEOUT_S}s" \
    env \
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
    verify-restore "$SELECTION_FILE" "$PHASE2_EVENTS" "$PHASE2_LOG"

echo "Single-controller Gym generation-prefix recovery functional test passed"
