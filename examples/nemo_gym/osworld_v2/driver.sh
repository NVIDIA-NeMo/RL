#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/runtime_env.sh"
cd "$OSWORLD_RL_ROOT"

OVERRIDES=(
  "grpo.max_num_steps=${OSWORLD_GRPO_MAX_STEPS:-300}"
  "logger.log_dir=$OSWORLD_RUN_ROOT/exp_001"
  "logger.wandb_enabled=false"
  "checkpointing.checkpoint_must_save_by=${OSWORLD_CHECKPOINT_MUST_SAVE_BY:-00:03:40:00}"
)
if [[ -n "${OSWORLD_SEED:-}" ]]; then
  OVERRIDES+=("grpo.seed=$OSWORLD_SEED")
fi

exec "$OSWORLD_DRIVER_PYTHON" \
  examples/nemo_gym/launch_osworld_v2_cc.py \
  --recipe "${OSWORLD_RECIPE:-flash-b8n8-dr-grpo}" \
  "${OVERRIDES[@]}"
