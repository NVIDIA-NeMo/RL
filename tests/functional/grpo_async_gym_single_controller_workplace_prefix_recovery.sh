#!/bin/bash
# Combined Workplace turn-state and active generation-prefix crash recovery.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

SC_GYM_PREFIX_RECOVERY_PROFILE=workplace \
    exec bash "$SCRIPT_DIR/grpo_async_gym_single_controller_prefix_recovery.sh" "$@"
