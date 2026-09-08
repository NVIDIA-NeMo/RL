#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

set -eou pipefail

EXP_NAME=grpo_nixl_non_colocated \
    bash "$SCRIPT_DIR/grpo_non_colocated.sh" \
    policy.generation.refit_transport=nixl
