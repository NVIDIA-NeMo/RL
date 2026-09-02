#!/usr/bin/env bash
set -Eeuo pipefail

physical_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if [[ "${physical_root}" == /lustre/fs?/* ]]; then
  ROOT="/lustre/fsw/${physical_root#/*/*/}"
else
  ROOT="${physical_root}"
fi
RUNTIME_ROOT="${OSWORLD_RUNTIME_ROOT:-$(dirname "${ROOT}")/osworld-cc-runtime}"
DATA_DIR="${OSWORLD_OVERFIT_DATA_DIR:-${RUNTIME_ROOT}/data/overfit361-cc-v2}"

export OSWORLD_GRPO_TRAIN_DATA="${OSWORLD_GRPO_TRAIN_DATA:-${DATA_DIR}/train-5x.jsonl}"
export OSWORLD_GRPO_VAL_DATA="${OSWORLD_GRPO_VAL_DATA:-${DATA_DIR}/validation-1x.jsonl}"
: "${NANO_OMNI_MODEL_NAME:?Set NANO_OMNI_MODEL_NAME to the SFT checkpoint}"
export NANO_OMNI_MODEL_NAME

# Molt Omni schedule: five dataset episodes, eight logical rollouts per prompt,
# and one optimizer epoch per consumed prompt group. OSWorld uses one prompt
# group per optimizer step to keep 48K multi-image batches within memory.
export MOLT_RUN_NAME="${MOLT_RUN_NAME:-osworld361-overfit5-molt-step400-lr5e6-r3-3to10to3}"
export MOLT_MAX_STEPS="${MOLT_MAX_STEPS:-1805}"
export OSWORLD_NUM_PROMPTS_PER_STEP="${OSWORLD_NUM_PROMPTS_PER_STEP:-1}"
export OSWORLD_NUM_GENERATIONS="${OSWORLD_NUM_GENERATIONS:-8}"
export GRPO_MAX_NUM_EPOCHS=1

export OSWORLD_LEARNING_RATE="${OSWORLD_LEARNING_RATE:-5e-6}"
export MOLT_MAX_STALENESS="${MOLT_MAX_STALENESS:-1}"
export MOLT_ASYNC_QUEUE_SIZE="${MOLT_ASYNC_QUEUE_SIZE:-1}"
export MOLT_IS_RATIO_MIN="${MOLT_IS_RATIO_MIN:-0.95}"
export MOLT_IS_RATIO_MAX="${MOLT_IS_RATIO_MAX:-1.05}"

# Jeff-style visual-history sawtooth: retain three images after compaction,
# grow to ten active images, and compact before admitting the eleventh.
export OSWORLD_MAX_MODEL_LEN="${OSWORLD_MAX_MODEL_LEN:-49152}"
export OSWORLD_MAX_ACTIVE_IMAGES="${OSWORLD_MAX_ACTIVE_IMAGES:-10}"
export OSWORLD_VLLM_MAX_IMAGES="${OSWORLD_VLLM_MAX_IMAGES:-11}"
export OSWORLD_CC_KEEP_LAST_IMAGE_GROUPS="${OSWORLD_CC_KEEP_LAST_IMAGE_GROUPS:-2}"
export OSWORLD_CC_ACTIONS_PER_CHUNK="${OSWORLD_CC_ACTIONS_PER_CHUNK:-100}"
export OSWORLD_MAX_STEPS="${OSWORLD_MAX_STEPS:-11}"

export OSWORLD_NEMO_GYM_NUM_WORKERS="${OSWORLD_NEMO_GYM_NUM_WORKERS:-8}"
export OSWORLD_MAX_PARALLEL_ROLLOUTS="${OSWORLD_MAX_PARALLEL_ROLLOUTS:-8}"
export MOLT_OPTIMIZER_CPU_OFFLOAD="${MOLT_OPTIMIZER_CPU_OFFLOAD:-true}"
export MOLT_OPTIMIZER_OFFLOAD_FRACTION="${MOLT_OPTIMIZER_OFFLOAD_FRACTION:-1.0}"

# The batch partition is capped at four hours. A subsequent submission with
# the same run/checkpoint directory resumes the atomic Molt buffer and trainer
# state from the latest completed optimizer step.
export CHECKPOINTING_ENABLED=true
export CHECKPOINT_SAVE_PERIOD="${CHECKPOINT_SAVE_PERIOD:-10}"
export CHECKPOINT_KEEP_TOP_K="${CHECKPOINT_KEEP_TOP_K:-2}"
export OSWORLD_VAL_PERIOD="${OSWORLD_VAL_PERIOD:-100000}"
export OSWORLD_VAL_AT_START="${OSWORLD_VAL_AT_START:-false}"
export OSWORLD_VAL_AT_END="${OSWORLD_VAL_AT_END:-true}"
export OSWORLD_VAL_BATCH_SIZE="${OSWORLD_VAL_BATCH_SIZE:-361}"
export SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
export SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
if [[ "${SBATCH_PARTITION}" == "batch_short" ]]; then
  export OSWORLD_CHECKPOINT_MUST_SAVE_BY="${OSWORLD_CHECKPOINT_MUST_SAVE_BY:-00:01:40:00}"
else
  export OSWORLD_CHECKPOINT_MUST_SAVE_BY="${OSWORLD_CHECKPOINT_MUST_SAVE_BY:-00:03:40:00}"
fi
# Worker nodes may carry container-baked venvs whose editable Megatron-Bridge
# points at /opt/nemo-rl instead of this checkout. Rebuild by default so local
# Molt and chunked-vision patches are present on every newly allocated node.
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-true}"

exec "${ROOT}/examples/nemo_gym/submit_osworld_molt_async.sh"
