#!/bin/bash
# Crash/recovery coverage for two distinct one-replica Gym shards. Prefix cuts
# and replicated shards are intentionally outside this first integration slice.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

SC_GYM_TURN_RECOVERY_PROFILE=sharded \
    exec bash "$SCRIPT_DIR/grpo_async_gym_single_controller_turn_recovery.sh" "$@"
