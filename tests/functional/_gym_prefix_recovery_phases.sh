# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Phase control shared by the prefix-recovery functional tests. Source it after
# setting SCRIPT_DIR and TEST_DIR; a test starts phase one in its own process
# group and keeps its PID in PHASE1_PID. The reference-free tests (router
# replay, Omni) also run both phases and their checks through the functions at
# the end.

PHASE1_PID=""

# Hard-kill phase one and wait until its controller can no longer write
# snapshots.
stop_phase1() {
    if [[ -z "${PHASE1_PID}" ]]; then
        return
    fi
    kill -KILL -- "-${PHASE1_PID}" 2>/dev/null || true
    wait "${PHASE1_PID}" 2>/dev/null || true
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

# On exit, stop phase one and delete the given checkpoint artifacts if the test
# passed, unless SC_PREFIX_RECOVERY_KEEP_CHECKPOINTS=1.
install_phase_cleanup() {
    PHASE_CLEANUP_PATHS=("$@")
    trap _phase_cleanup EXIT
}

_phase_cleanup() {
    local status=$?
    stop_phase1
    if [[ "${status}" -eq 0 && "${SC_PREFIX_RECOVERY_KEEP_CHECKPOINTS:-0}" != "1" ]]; then
        rm -rf "${PHASE_CLEANUP_PATHS[@]}"
    else
        echo "Preserving prefix-recovery artifacts for inspection: ${TEST_DIR}"
    fi
    return "${status}"
}

# Paths of a reference-free test, all under TEST_DIR.
set_prefix_recovery_paths() {
    RECOVERY_HOOK="${SCRIPT_DIR}/_single_controller_turn_recovery_hook.py"
    SNAPSHOT_HELPER="${SCRIPT_DIR}/_gym_prefix_recovery_snapshot.py"
    CHECKPOINT_DIR="${TEST_DIR}/recovery-checkpoints"
    PHASE1_LOG="${TEST_DIR}/phase1.log"
    PHASE2_LOG="${TEST_DIR}/phase2.log"
    PHASE1_EVENTS="${TEST_DIR}/phase1-events.jsonl"
    PHASE2_EVENTS="${TEST_DIR}/phase2-events.jsonl"
    # Each phase writes into its own logger.log_dir: phase two's trained rows
    # are read back from the only experiment directory under RECOVERY_LOG_DIR.
    PHASE1_LOG_DIR="${TEST_DIR}/phase1-logs"
    RECOVERY_LOG_DIR="${TEST_DIR}/recovery-logs"
    RECOVERY_STAGING_KEYS="${TEST_DIR}/recovery-staging-keys.jsonl"
    RECOVERY_FLUSH_KEYS="${TEST_DIR}/recovery-flush-keys.jsonl"
    PHASE2_METRICS="${TEST_DIR}/phase2-metrics.json"
    SELECTION_FILE="${TEST_DIR}/selected-snapshot.json"
    SELECTED_BACKUP="${TEST_DIR}/selected-snapshot"
    TEST_DATA="${TEST_DIR}/test-data.jsonl"
}

# Overrides every reference-free test shares, into PREFIX_RECOVERY_OVERRIDES:
# fixed prompt order, one prompt group per prompt, rollout checkpoints with
# prefix recovery, and a dump of the trained rows for verify-completion. Call
# after setting NUM_PROMPTS, NUM_GENERATIONS, MAX_STEPS, MAX_MODEL_LEN,
# TRAIN_GLOBAL_BATCH_SIZE and SNAPSHOT_INTERVAL_S. Phase two trains
# PHASE2_STEPS, one step past phase one, so the snapshot after step MAX_STEPS
# deletes the restored calls' retired prefix rows while the run is still alive
# to record it; the test data must hold prompts for every one of its steps.
prefix_recovery_overrides() {
    PREFIX_RECOVERY_OVERRIDES=(
        policy.max_total_sequence_length="${MAX_MODEL_LEN}"
        policy.train_global_batch_size="${TRAIN_GLOBAL_BATCH_SIZE}"
        policy.train_micro_batch_size=1
        data.shuffle=false
        data.train.data_path="${TEST_DATA}"
        data.validation.data_path="${TEST_DATA}"
        grpo.seed=1234
        grpo.num_prompts_per_step="${NUM_PROMPTS}"
        grpo.num_generations_per_prompt="${NUM_GENERATIONS}"
        grpo.max_num_steps="${MAX_STEPS}"
        grpo.val_period=0
        grpo.val_at_start=false
        ++grpo.val_at_end=false
        ++async_rl.sampler.name=in_order
        ++async_rl.sampler.max_lookahead_versions=0
        ++async_rl.min_groups_for_streaming_train="${NUM_PROMPTS}"
        ++async_rl.max_inflight_prompts="${NUM_PROMPTS}"
        ++async_rl.max_buffered_rollouts="${NUM_PROMPTS}"
        ++async_rl.stall_watchdog.interval_s=10
        ++async_rl.stall_watchdog.stall_action=abort
        ++token_capture.enabled=true
        ++rollout_recovery.target_level=prefix
        ++rollout_checkpointing.snapshot_attempt_interval_s="${SNAPSHOT_INTERVAL_S}"
        ++rollout_checkpointing.keep_latest_k=16
        ++rollout_checkpointing.restore_mode=latest
        # The step count only decides when a run stops, so phase two's
        # restore must accept its longer run.
        '++rollout_checkpointing.extra_fingerprint_excluded_paths=[grpo.max_num_steps]'
        checkpointing.enabled=true
        checkpointing.checkpoint_dir="${CHECKPOINT_DIR}"
        checkpointing.metric_name=null
        # Phase two saves after step MAX_STEPS, whose snapshot deletes the
        # retired prefix rows, and after its last step, whose training_info is
        # read.
        checkpointing.save_period="${MAX_STEPS}"
        ++checkpointing.save_data_plane=true
        # Each run dumps its trained rows (train_data_step*.jsonl in its log
        # dir) for verify-completion.
        ++async_rl.log_full_train_data=true
        logger.tensorboard_enabled=true
        logger.wandb_enabled=false
        logger.monitor_gpus=false
    )
}

# Run phase one until a snapshot cuts every prompt group, hard-kill it, keep
# only that snapshot, then restore from it in phase two. Arguments: the two
# phase banners, the name of the command array, then extra overrides for both
# phases.
run_prefix_recovery_phases() {
    local phase1_banner=$1 phase2_banner=$2
    local -n phase_cmd=$3
    shift 3

    echo "=== Phase 1: ${phase1_banner} ==="
    command -v setsid >/dev/null
    setsid env \
        SC_GYM_RECOVERY_TEST_EVENTS="${PHASE1_EVENTS}" \
        "${phase_cmd[@]}" \
            logger.log_dir="${PHASE1_LOG_DIR}" \
            "$@" >"${PHASE1_LOG}" 2>&1 &
    PHASE1_PID=$!

    # Every prompt group must hold a recoverable prefix.
    uv run --no-sync python "${SNAPSHOT_HELPER}" select \
        "${CHECKPOINT_DIR}/bootstrap/rollout_snapshots" \
        "${SELECTION_FILE}" \
        "${PHASE1_PID}" \
        "${PHASE1_LOG}" \
        "${SNAPSHOT_TIMEOUT_S}" \
        "${SELECTED_BACKUP}" \
        --stateless-resources \
        --min-cut-groups "${NUM_PROMPTS}"

    stop_phase1
    local snapshot_dir snapshot_root trainer_checkpoint
    snapshot_dir=$(uv run --no-sync python -c \
        'import json, sys; print(json.load(open(sys.argv[1]))["snapshot"])' \
        "${SELECTION_FILE}")
    snapshot_root=$(dirname "${snapshot_dir}")

    # Recover exactly the prefix selected above. A newer periodic snapshot or a
    # trainer checkpoint would represent work performed after the simulated
    # crash.
    rm -rf "${snapshot_root}"
    mkdir -p "${snapshot_root}"
    cp -a "${SELECTED_BACKUP}" "${snapshot_dir}"
    for trainer_checkpoint in "${CHECKPOINT_DIR}"/step_*; do
        if [[ -d "${trainer_checkpoint}" ]]; then
            rm -rf "${trainer_checkpoint}"
        fi
    done

    echo "=== Phase 2: ${phase2_banner} ==="
    # Do not cut the replacement requests again while they consume the prefix.
    : >"${RECOVERY_FLUSH_KEYS}"
    timeout --signal=TERM --kill-after=30s "${RUN_TIMEOUT_S}s" \
        env \
            SC_GYM_RECOVERY_TEST_EVENTS="${PHASE2_EVENTS}" \
            SC_PREFIX_RECOVERY_STAGING_KEYS="${RECOVERY_STAGING_KEYS}" \
            SC_PREFIX_RECOVERY_FLUSH_KEYS="${RECOVERY_FLUSH_KEYS}" \
        "${phase_cmd[@]}" \
            ++rollout_checkpointing.snapshot_attempt_interval_s="${PHASE2_SNAPSHOT_INTERVAL_S}" \
            grpo.max_num_steps="${PHASE2_STEPS}" \
            logger.log_dir="${RECOVERY_LOG_DIR}" \
            "$@" 2>&1 | tee "${PHASE2_LOG}"

    grep -Fq "Selected rollout recovery snapshot: ${snapshot_dir}" "${PHASE2_LOG}"
    grep -q "Native TQ checkpoint restored and validated" "${PHASE2_LOG}"
    grep -q "Loaded .* unfinished rollout group(s)" "${PHASE2_LOG}"
    grep -q "generation prefix restored:" "${PHASE2_LOG}"
    grep -q "train step ${PHASE2_STEPS}/${PHASE2_STEPS}" "${PHASE2_LOG}"
}

# Check the restore, then that each restored call completed exactly once, its
# trained row begins that call with the exact restored prefix, and its restored
# prefix rows outlived training only until the next snapshot. Extra arguments
# go to verify-completion (router replay passes --require-routes).
verify_prefix_recovery() {
    uv run --no-sync python "${SNAPSHOT_HELPER}" verify-restore \
        "${SELECTION_FILE}" \
        "${PHASE1_EVENTS}" \
        "${PHASE2_EVENTS}" \
        "${PHASE2_LOG}" \
        "${CHECKPOINT_DIR}/step_${PHASE2_STEPS}/training_info.json" \
        "${PHASE2_STEPS}"
    uv run --no-sync python "${SNAPSHOT_HELPER}" verify-completion \
        "${SELECTION_FILE}" \
        "${RECOVERY_LOG_DIR}" \
        "${RECOVERY_STAGING_KEYS}" \
        "${RECOVERY_FLUSH_KEYS}" \
        "${PHASE2_LOG}" \
        "$@"
}
