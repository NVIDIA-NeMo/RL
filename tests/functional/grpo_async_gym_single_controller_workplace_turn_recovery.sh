#!/bin/bash
# Deterministic Workplace Assistant state recovery through the coordinated
# Single Controller + Gym + TQ turn-checkpoint path.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

SC_GYM_TURN_RECOVERY_PROFILE=workplace \
    exec bash "$SCRIPT_DIR/grpo_async_gym_single_controller_turn_recovery.sh" "$@"
