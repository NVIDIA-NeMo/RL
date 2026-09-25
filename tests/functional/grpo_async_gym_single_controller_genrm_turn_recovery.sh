#!/bin/bash
# Crash/recovery coverage for a two-sibling stateless GenRM cohort wait.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

SC_GYM_TURN_RECOVERY_PROFILE=genrm \
    exec bash "$SCRIPT_DIR/grpo_async_gym_single_controller_turn_recovery.sh" "$@"
