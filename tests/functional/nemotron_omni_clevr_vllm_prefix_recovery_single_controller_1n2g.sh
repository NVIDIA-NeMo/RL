#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Multimodal smoke test for generation-prefix recovery on the Omni CLEVR vLLM
# token-capture setup. A Gym-v2 checkpoint is taken while vLLM is mid-decode on
# image prompts; the process is hard-killed and restored. Each restored call
# must reprocess its image, verify it against the durable TQ media copy, resume
# from the exact durable token prefix, and finish and train.
#
# Token-for-token parity against an uninterrupted run is the Workplace text
# test's job (grpo_async_gym_single_controller_prefix_recovery.sh). Here the
# Reasoning model's free-form thinking flips near-tied tokens between runs, so
# parity would be flaky; this test instead checks each restored prefix appears
# verbatim in its trained row. Other differences from the Workplace test:
# - Single-turn string_match rows: the cut lands in the only model call, so the
#   first durable prefix row is the one that owns the image tensors.
# - string_match is stateless, so snapshot selection runs with
#   --stateless-resources instead of a calendar-sentinel check.
# - Each prompt group gets a distinct image, so a restore that verified against
#   the wrong group's pixels would fail (retained_media_changed) instead of
#   passing on identical fixtures.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "SKIP: HF_TOKEN is required for the Omni checkpoint"
    exit 0
fi

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( GPU_COUNT < 2 )); then
    echo "SKIP: Omni CLEVR vLLM prefix recovery requires at least two visible GPUs"
    exit 0
fi
DETECTED_CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${DETECTED_CUDA_ARCH}}"

EXP_NAME=$(basename "$0" .sh)
TEST_DIR="${SCRIPT_DIR}/${EXP_NAME}"
OMNI_CONFIG="${PROJECT_ROOT}/examples/nemo_gym/vlm_grpo_nemotron_omni_clevr_vllm_token_capture_single_controller.yaml"
RECOVERY_HOOK="${SCRIPT_DIR}/_single_controller_turn_recovery_hook.py"
SNAPSHOT_HELPER="${SCRIPT_DIR}/_gym_prefix_recovery_snapshot.py"
CHECKPOINT_DIR="${TEST_DIR}/recovery-checkpoints"
PHASE1_LOG="${TEST_DIR}/phase1.log"
PHASE2_LOG="${TEST_DIR}/phase2.log"
PHASE1_EVENTS="${TEST_DIR}/phase1-events.jsonl"
PHASE2_EVENTS="${TEST_DIR}/phase2-events.jsonl"
# Each phase writes into its own logger.log_dir: phase two's trained rows are
# read back from the only experiment directory under RECOVERY_LOG_DIR.
PHASE1_LOG_DIR="${TEST_DIR}/phase1-logs"
RECOVERY_LOG_DIR="${TEST_DIR}/recovery-logs"
RECOVERY_STAGING_KEYS="${TEST_DIR}/recovery-staging-keys.jsonl"
RECOVERY_FLUSH_KEYS="${TEST_DIR}/recovery-flush-keys.jsonl"
PHASE2_METRICS="${TEST_DIR}/phase2-metrics.json"
SELECTION_FILE="${TEST_DIR}/selected-snapshot.json"
SELECTED_BACKUP="${TEST_DIR}/selected-snapshot"
TEST_DATA="${TEST_DIR}/test-data.jsonl"

# One prompt group per list length, each with its own solid-color image. The
# list keeps decode long enough to be cut mid-generation; distinct images mean
# a restore verified against another group's pixels would be rejected.
LIST_LENGTHS=(${SC_MM_PREFIX_RECOVERY_LIST_LENGTHS:-120 240})
NUM_PROMPTS=${#LIST_LENGTHS[@]}
NUM_GENERATIONS=${SC_MM_PREFIX_RECOVERY_NUM_GENERATIONS:-2}
# Conservative per-item budget: the Omni tokenizer's digit grouping is not
# assumed, so "N\n" is budgeted like Qwen's one-token-per-digit.
LIST_TOKENS_PER_ITEM=6
LIST_OUTPUT_SLACK_TOKENS=256
MAX_MODEL_LEN=${SC_MM_PREFIX_RECOVERY_MAX_MODEL_LEN:-4096}
MAX_STEPS=${SC_MM_PREFIX_RECOVERY_MAX_STEPS:-2}
# Phase two trains one step past phase one, so the snapshot after step
# MAX_STEPS deletes the restored calls' retired prefix rows while the run is
# still alive to record it.
PHASE2_STEPS=$((MAX_STEPS + 1))
SNAPSHOT_INTERVAL_S=${SC_MM_PREFIX_RECOVERY_INTERVAL_S:-0.05}
PHASE2_SNAPSHOT_INTERVAL_S=${SC_MM_PREFIX_RECOVERY_PHASE2_INTERVAL_S:-600}
SNAPSHOT_TIMEOUT_S=${SC_MM_PREFIX_RECOVERY_TIMEOUT_S:-3600}
RUN_TIMEOUT_S=${SC_MM_PREFIX_RECOVERY_RUN_TIMEOUT_S:-3600}
TRAIN_GLOBAL_BATCH_SIZE=$((NUM_PROMPTS * NUM_GENERATIONS))
POLICY_MAX_NEW_TOKENS=0
for list_length in "${LIST_LENGTHS[@]}"; do
    list_budget=$((list_length * LIST_TOKENS_PER_ITEM + LIST_OUTPUT_SLACK_TOKENS))
    if (( list_budget > POLICY_MAX_NEW_TOKENS )); then
        POLICY_MAX_NEW_TOKENS=$list_budget
    fi
done

rm -rf "${TEST_DIR}"
mkdir -p "${TEST_DIR}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# The test config trains one epoch, so the data holds a step of prompts for
# every phase-two step; phase one stops after MAX_STEPS of them.
TEST_DATA="${TEST_DATA}" DATA_STEPS="${PHASE2_STEPS}" \
LIST_LENGTHS="${LIST_LENGTHS[*]}" LIST_TOKENS_PER_ITEM="${LIST_TOKENS_PER_ITEM}" \
LIST_OUTPUT_SLACK_TOKENS="${LIST_OUTPUT_SLACK_TOKENS}" \
uv run --no-sync python - <<'PY'
import base64
import io
import json
import os

from PIL import Image

COLORS = ["red", "green", "blue", "yellow"]
list_lengths = [int(value) for value in os.environ["LIST_LENGTHS"].split()]
if not 1 <= len(list_lengths) <= len(COLORS):
    raise SystemExit(f"need 1 to {len(COLORS)} list lengths, got {len(list_lengths)}")
per_item = int(os.environ["LIST_TOKENS_PER_ITEM"])
slack = int(os.environ["LIST_OUTPUT_SLACK_TOKENS"])


def image_url(color: str) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), color=color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


with open(os.environ["TEST_DATA"], "w") as output:
    for step in range(int(os.environ["DATA_STEPS"])):
        for slot, list_length in enumerate(list_lengths):
            color = COLORS[slot]
            text = (
                f"Step {step} item {slot}: list every integer from 1 to {list_length} "
                "in increasing order, one per line, with no other text. Then give "
                "the color of the image as one word inside \\boxed{}."
            )
            row = {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": image_url(color),
                                    "detail": "auto",
                                },
                                {"type": "input_text", "text": text},
                            ],
                        }
                    ],
                    "max_output_tokens": list_length * per_item + slack,
                },
                "expected_answer": color,
                "extraction_mode": "boxed",
                "case_sensitive": False,
                "agent_ref": {
                    "type": "responses_api_agents",
                    "name": "string_match_simple_agent",
                },
            }
            output.write(json.dumps(row) + "\n")
PY

source "${SCRIPT_DIR}/_gym_prefix_recovery_phases.sh"
install_phase_cleanup "${CHECKPOINT_DIR}" "${SELECTED_BACKUP}"

# The text prefix-recovery settings, on the YAML's Omni CLEVR topology (one
# Megatron training GPU with a frozen language model, one async vLLM GPU).
COMMON_OVERRIDES=(
    # Quoted for Hydra: a bare compute capability (9.0, 10.0) parses as a
    # float, which the string-typed env_vars field rejects.
    policy.megatron_cfg.env_vars.TORCH_CUDA_ARCH_LIST="'${TORCH_CUDA_ARCH_LIST}'"
    policy.generation.max_new_tokens="${POLICY_MAX_NEW_TOKENS}"
    policy.generation.vllm_cfg.max_model_len="${MAX_MODEL_LEN}"
    ++policy.generation.vllm_kwargs.max_num_batched_tokens="${MAX_MODEL_LEN}"
    # Near-greedy decoding keeps each list (and so each decode) predictable.
    policy.generation.temperature=1.0
    policy.generation.top_p=0.000001
    # The trainer replays the top_p filter, so it needs vLLM's logprobs after
    # that filter.
    ++policy.generation.vllm_cfg.logprobs_mode=processed_logprobs
    policy.max_total_sequence_length="${MAX_MODEL_LEN}"
    policy.train_global_batch_size="${TRAIN_GLOBAL_BATCH_SIZE}"
    # Keep each step's prompt membership fixed across runs.
    data.shuffle=false
    data.train.data_path="${TEST_DATA}"
    data.validation.data_path="${TEST_DATA}"
    grpo.seed=1234
    grpo.num_prompts_per_step="${NUM_PROMPTS}"
    grpo.num_generations_per_prompt="${NUM_GENERATIONS}"
    grpo.max_num_steps="${MAX_STEPS}"
    async_rl.sampler.name=in_order
    async_rl.sampler.max_lookahead_versions=0
    async_rl.min_groups_for_streaming_train="${NUM_PROMPTS}"
    async_rl.max_inflight_prompts="${NUM_PROMPTS}"
    async_rl.max_buffered_rollouts="${NUM_PROMPTS}"
    ++async_rl.rollout_failure.nemo_gym.rollout_timeout_s=600
    ++async_rl.stall_watchdog.interval_s=10
    ++async_rl.stall_watchdog.stall_timeout_s=900
    ++async_rl.stall_watchdog.stall_action=abort
    ++token_capture.enabled=true
    ++rollout_recovery.target_level=prefix
    ++rollout_checkpointing.snapshot_attempt_interval_s="${SNAPSHOT_INTERVAL_S}"
    ++rollout_checkpointing.keep_latest_k=16
    ++rollout_checkpointing.restore_mode=latest
    # Phase two trains one step longer than phase one (PHASE2_STEPS); the step
    # count only decides when the run stops, so restore must accept it.
    '++rollout_checkpointing.extra_fingerprint_excluded_paths=[grpo.max_num_steps]'
    checkpointing.enabled=true
    checkpointing.checkpoint_dir="${CHECKPOINT_DIR}"
    checkpointing.metric_name=null
    # Phase two saves after step MAX_STEPS, whose snapshot deletes the retired
    # prefix rows, and after its last step, whose training_info is read; every
    # Omni save writes the full 30B model.
    checkpointing.save_period="${MAX_STEPS}"
    ++checkpointing.save_data_plane=true
    # Each run dumps its trained rows (train_data_step*.jsonl in its log dir)
    # for verify-completion.
    ++async_rl.log_full_train_data=true
)

OMNI_CMD=(
    uv run --no-sync python "${RECOVERY_HOOK}"
    --config "${OMNI_CONFIG}"
    "${COMMON_OVERRIDES[@]}"
)

echo "=== Phase 1: publish a checkpoint holding an active multimodal generation prefix ==="
command -v setsid >/dev/null
setsid env \
    SC_GYM_RECOVERY_TEST_EVENTS="${PHASE1_EVENTS}" \
    "${OMNI_CMD[@]}" \
        logger.log_dir="${PHASE1_LOG_DIR}" \
        "$@" >"${PHASE1_LOG}" 2>&1 &
PHASE1_PID=$!

# Every prompt group must hold a recoverable prefix, so each distinct image is
# restored and verified at least once.
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
restore_selected_snapshot "${SELECTION_FILE}" "${SELECTED_BACKUP}" "${CHECKPOINT_DIR}"

echo "=== Phase 2: reprocess the images, verify them, and finish each suffix ==="
# Do not cut the replacement requests again while they consume the prefix.
: > "${RECOVERY_FLUSH_KEYS}"
timeout --signal=TERM --kill-after=30s "${RUN_TIMEOUT_S}s" \
    env \
        SC_GYM_RECOVERY_TEST_EVENTS="${PHASE2_EVENTS}" \
        SC_PREFIX_RECOVERY_STAGING_KEYS="${RECOVERY_STAGING_KEYS}" \
        SC_PREFIX_RECOVERY_FLUSH_KEYS="${RECOVERY_FLUSH_KEYS}" \
    "${OMNI_CMD[@]}" \
        ++rollout_checkpointing.snapshot_attempt_interval_s="${PHASE2_SNAPSHOT_INTERVAL_S}" \
        grpo.max_num_steps="${PHASE2_STEPS}" \
        logger.log_dir="${RECOVERY_LOG_DIR}" \
        "$@" 2>&1 | tee "${PHASE2_LOG}"

grep -Fq "Selected rollout recovery snapshot: ${SNAPSHOT_DIR}" "${PHASE2_LOG}"
grep -q "Native TQ checkpoint restored and validated" "${PHASE2_LOG}"
grep -q "Loaded .* unfinished rollout group(s)" "${PHASE2_LOG}"
grep -q "generation prefix restored:" "${PHASE2_LOG}"
grep -q "train step ${PHASE2_STEPS}/${PHASE2_STEPS}" "${PHASE2_LOG}"
# A media mismatch is rejected with retained_media_changed and the call is
# retried from the prompt, which would otherwise pass as a fresh rollout.
if grep -q "retained_media_changed" "${PHASE2_LOG}"; then
    echo "[ERROR] a restored multimodal prefix failed media verification"
    exit 1
fi

uv run --no-sync tests/json_dump_tb_logs.py "${RECOVERY_LOG_DIR}" --output_path "${PHASE2_METRICS}"
# media_row_rate == 1 proves every learner row, including the restored ones,
# still carried captured pixels after recovery.
uv run --no-sync tests/check_metrics.py "${PHASE2_METRICS}" \
    'max(data["train/finalize/invalid_row_rate"]) == 0' \
    'max(data["train/finalize/capture_poisoned_rollouts"]) == 0' \
    'min(data["train/finalize/media_row_rate"]) == 1' \
    'all_finite(data["train/reward"])'

uv run --no-sync python "${SNAPSHOT_HELPER}" verify-restore \
    "${SELECTION_FILE}" \
    "${PHASE1_EVENTS}" \
    "${PHASE2_EVENTS}" \
    "${PHASE2_LOG}" \
    "${CHECKPOINT_DIR}/step_${PHASE2_STEPS}/training_info.json" \
    "${PHASE2_STEPS}"

# Each restored call completed exactly once, its trained row begins that call
# with the exact restored prefix, and its restored prefix rows outlived
# training only until the next snapshot.
uv run --no-sync python "${SNAPSHOT_HELPER}" verify-completion \
    "${SELECTION_FILE}" \
    "${RECOVERY_LOG_DIR}" \
    "${RECOVERY_STAGING_KEYS}" \
    "${RECOVERY_FLUSH_KEYS}" \
    "${PHASE2_LOG}"

echo "Omni CLEVR multimodal generation-prefix recovery test passed"
