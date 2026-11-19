#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/runtime_env.sh"

# Context-compacted rollouts can contain more physical TQ rows than the logical
# num_generations_per_prompt group size, so this branch cannot restore their
# buffered replay sidecars. Opt in to a replay-free resume while preserving the
# artifacts for diagnosis instead of failing after model initialization.
if [[ "${OSWORLD_DISCARD_REPLAY_ON_RESUME:-0}" == "1" ]]; then
  latest_status="$OSWORLD_CHECKPOINT_DIR/latest_checkpoint_status.json"
  if [[ -s "$latest_status" ]]; then
    latest_step="$(
      "$OSWORLD_COMPONENT_PYTHON" - "$latest_status" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as status_file:
    print(int(json.load(status_file)["last_checkpoint_step"]))
PY
    )"
    latest_checkpoint="$OSWORLD_CHECKPOINT_DIR/step_$latest_step"
    quarantine_suffix="discarded-cc-replay-${SLURM_JOB_ID:-manual}"
    for sidecar in replay_buffer_metadata.pt rollout_recovery.pt; do
      sidecar_path="$latest_checkpoint/$sidecar"
      if [[ -f "$sidecar_path" ]]; then
        mv -- "$sidecar_path" "$sidecar_path.$quarantine_suffix"
        echo "Quarantined unsupported CC replay sidecar: $sidecar_path"
      fi
    done
  fi
fi

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
    # NeMo-Gym delegates truncation to the generation engine, so keep its
    # context limit aligned with the policy/packing limit.
    "policy.generation.vllm_cfg.max_model_len=$OSWORLD_MAX_TOTAL_SEQUENCE_LENGTH"
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
if [[ -n "${OSWORLD_OPTIMIZER_OFFLOAD_FRACTION:-}" ]]; then
  OVERRIDES+=(
    "policy.megatron_cfg.optimizer.optimizer_offload_fraction=$OSWORLD_OPTIMIZER_OFFLOAD_FRACTION"
  )
fi
if [[ -n "${OSWORLD_PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  OVERRIDES+=(
    "policy.megatron_cfg.env_vars={PYTORCH_CUDA_ALLOC_CONF: \"$OSWORLD_PYTORCH_CUDA_ALLOC_CONF\"}"
  )
fi
if [[ -n "${OSWORLD_STORAGE_UNIT_PLACEMENT:-}" ]]; then
  OVERRIDES+=(
    "+data_plane.simple.storage_unit_placement=[$OSWORLD_STORAGE_UNIT_PLACEMENT]"
  )
fi
if [[ -n "${OSWORLD_TEMPERATURE:-}" ]]; then
  OVERRIDES+=(
    "policy.generation.temperature=$OSWORLD_TEMPERATURE"
    "policy.generation.val_temperature=$OSWORLD_TEMPERATURE"
  )
fi
if [[ -n "${OSWORLD_MAX_INFLIGHT_PROMPTS:-}" ]]; then
  OVERRIDES+=(
    "async_rl.max_inflight_prompts=$OSWORLD_MAX_INFLIGHT_PROMPTS"
  )
fi
if [[ -n "${OSWORLD_MAX_BUFFERED_ROLLOUTS:-}" ]]; then
  OVERRIDES+=(
    "async_rl.max_buffered_rollouts=$OSWORLD_MAX_BUFFERED_ROLLOUTS"
  )
fi
if [[ -n "${OSWORLD_NUM_PROMPTS_PER_STEP:-}" ]]; then
  OVERRIDES+=(
    "grpo.num_prompts_per_step=$OSWORLD_NUM_PROMPTS_PER_STEP"
  )
fi
if [[ -n "${OSWORLD_MIN_GROUPS_FOR_STREAMING_TRAIN:-}" ]]; then
  OVERRIDES+=(
    "async_rl.min_groups_for_streaming_train=$OSWORLD_MIN_GROUPS_FOR_STREAMING_TRAIN"
  )
fi
if [[ -n "${OSWORLD_NUM_GENERATIONS_PER_PROMPT:-}" ]]; then
  OVERRIDES+=(
    "grpo.num_generations_per_prompt=$OSWORLD_NUM_GENERATIONS_PER_PROMPT"
  )
fi
if [[ -n "${OSWORLD_TRAIN_GLOBAL_BATCH_SIZE:-}" ]]; then
  OVERRIDES+=(
    "policy.train_global_batch_size=$OSWORLD_TRAIN_GLOBAL_BATCH_SIZE"
  )
fi
if [[ -n "${OSWORLD_GENERATION_BATCH_SIZE:-}" ]]; then
  OVERRIDES+=(
    "policy.generation_batch_size=$OSWORLD_GENERATION_BATCH_SIZE"
  )
fi
if [[ -n "${OSWORLD_TRAIN_MAX_PHYSICAL_ROWS_PER_DISPATCH:-}" ]]; then
  OVERRIDES+=(
    "policy.train_max_physical_rows_per_dispatch=$OSWORLD_TRAIN_MAX_PHYSICAL_ROWS_PER_DISPATCH"
  )
fi
if [[ -n "${OSWORLD_LOGPROB_MAX_PHYSICAL_ROWS_PER_DISPATCH:-}" ]]; then
  OVERRIDES+=(
    "policy.logprob_max_physical_rows_per_dispatch=$OSWORLD_LOGPROB_MAX_PHYSICAL_ROWS_PER_DISPATCH"
  )
fi

exec "$OSWORLD_DRIVER_PYTHON" \
  examples/nemo_gym/launch_osworld_v2_cc.py \
  "--$OSWORLD_RECIPE" \
  "${OVERRIDES[@]}"
