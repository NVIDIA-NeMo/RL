#!/bin/bash
# Full-process active-prefix crash recovery across two distinct one-replica Gym
# shards. The selected cut is owned by the non-leader shard.

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

SC_GYM_PREFIX_RECOVERY_PROFILE=sharded \
    exec bash "$SCRIPT_DIR/grpo_async_gym_single_controller_prefix_recovery.sh" "$@"
