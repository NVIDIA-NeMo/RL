#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

set -eou pipefail

EXP_NAME=grpo_non_colocated_reload_refit \
    bash "$SCRIPT_DIR/grpo_non_colocated.sh" \
    policy.generation.vllm_cfg.refit_with_reload_api=true \
    "$@"
