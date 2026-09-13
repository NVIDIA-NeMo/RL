#!/bin/bash
# vLLM reload_weights refit smoke for the SingleController entrypoint.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

set -eou pipefail

EXP_NAME=grpo_dp_single_controller_reload_refit \
    bash "$SCRIPT_DIR/grpo_dp_single_controller.sh" \
    policy.generation.vllm_cfg.refit_with_reload_api=true \
    "$@"
