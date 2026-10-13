#!/bin/bash
# Exact inline token-prefix mode for the external-vLLM Gym E2E test.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
PLUGIN_DIR=${EXTERNAL_VLLM_TEST_PLUGIN_DIR:-$SCRIPT_DIR/grpo_external_vllm_prefix_gym/plugin}
PLUGIN_WHEEL=$(bash "$PROJECT_ROOT/tools/external_rollout_vllm/build_prefix_plugin.sh" "$PLUGIN_DIR")
export PYTHONPATH="$PLUGIN_WHEEL:${PYTHONPATH:-}"
export VLLM_PLUGINS=nemo_rl_prefix_api
export EXTERNAL_VLLM_TEST_REQUIRE_PREFIX=1
export EXTERNAL_VLLM_TEST_MAX_AGENT_STEPS=${EXTERNAL_VLLM_TEST_MAX_AGENT_STEPS:-6}
export EXTERNAL_VLLM_TEST_RUN_DIR=${EXTERNAL_VLLM_TEST_RUN_DIR:-$SCRIPT_DIR/grpo_external_vllm_prefix_gym/run}
bash "$SCRIPT_DIR/grpo_external_vllm_gym_single_controller.sh" \
    ++env.nemo_gym.policy_model.responses_api_models.vllm_model.supply_prefix_token_ids=true \
    "$@"

uv run "$PROJECT_ROOT/tests/check_metrics.py" "$EXTERNAL_VLLM_TEST_RUN_DIR/train/metrics.json" \
    'max(data["train/turns_per_sample/max"]) >= 2'
