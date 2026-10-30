#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/runtime_env.sh"
"$SCRIPT_DIR/prepare_gym_agent.sh"
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
if [[ -n "${OSWORLD_LR_WARMUP_ITERS:-}" ]]; then
  OVERRIDES+=(
    "policy.megatron_cfg.scheduler.lr_warmup_iters=$OSWORLD_LR_WARMUP_ITERS"
  )
fi
if [[ -n "${OSWORLD_MAX_TOTAL_SEQUENCE_LENGTH:-}" ]]; then
  OVERRIDES+=(
    "policy.max_total_sequence_length=$OSWORLD_MAX_TOTAL_SEQUENCE_LENGTH"
  )
fi
if [[ -n "${OSWORLD_TRAIN_MB_TOKENS:-}" ]]; then
  OVERRIDES+=(
    "policy.sequence_packing.train_mb_tokens=$OSWORLD_TRAIN_MB_TOKENS"
  )
fi
if [[ -n "${OSWORLD_LOGPROB_MB_TOKENS:-}" ]]; then
  OVERRIDES+=(
    "policy.sequence_packing.logprob_mb_tokens=$OSWORLD_LOGPROB_MB_TOKENS"
  )
fi
if [[ -n "${OSWORLD_TEMPERATURE:-}" ]]; then
  OVERRIDES+=(
    "policy.generation.temperature=$OSWORLD_TEMPERATURE"
    "policy.generation.val_temperature=$OSWORLD_TEMPERATURE"
  )
fi

exec "$OSWORLD_DRIVER_PYTHON" \
  examples/nemo_gym/launch_osworld_v2_cc.py \
  --molt-b8k8-checkpoint \
  "${OVERRIDES[@]}"
