#!/bin/bash
# External capture mode: exact tokens travel through staging, Gym carries refs.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
export EXTERNAL_VLLM_TEST_RUN_DIR=${EXTERNAL_VLLM_TEST_RUN_DIR:-$SCRIPT_DIR/grpo_external_vllm_capture_gym/run}
bash "$SCRIPT_DIR/grpo_external_vllm_prefix_gym.sh" \
    token_capture.enabled=true \
    ++env.nemo_gym.policy_model.responses_api_models.vllm_model.return_token_id_information=false \
    ++env.nemo_gym.policy_model.responses_api_models.vllm_model.supply_prefix_token_ids=false \
    "$@"

uv run "$PROJECT_ROOT/tests/check_metrics.py" "$EXTERNAL_VLLM_TEST_RUN_DIR/train/metrics.json" \
    'min(data["train/finalize/calls_per_rollout"]) > 0' \
    'max(data["train/finalize/calls_per_rollout"]) > 1' \
    'max(data["train/finalize/token_in_calls"]) > 0' \
    'max(data["train/finalize/invalid_row_rate"]) == 0' \
    'max(data["train/finalize/capture_poisoned_rollouts"]) == 0'
