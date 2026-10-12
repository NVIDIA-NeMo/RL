#!/bin/bash
# Streaming counterpart of the full-batch SingleController PPO smoke.
# Reuse its critic warmup, checkpoint/restore and metric checks with one policy
# epoch and a readiness threshold of one prompt group out of two.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXP_NAME=$(basename "$0" .sh)
export EXP_NAME

# Keep dispatch bounded while exercising the streaming configuration. The
# number of chunks in a step depends on timing and is not asserted here.
bash "${SCRIPT_DIR}/ppo_async_single_controller.sh" \
    ppo.ppo_epochs=1 \
    ppo.critic_ppo_epochs=2 \
    async_rl.min_groups_for_streaming_train=1 \
    async_rl.sampler.max_lookahead_versions=0 \
    async_rl.sampler.warmup_lookahead_versions=0 \
    async_rl.max_inflight_prompts=1 \
    "$@"
