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

# Match the validated RFC0037 step-400 Molt recipe while keeping checkpoint
# locations portable across clusters.
: "${NANO_OMNI_MODEL_NAME:?Set NANO_OMNI_MODEL_NAME to the SFT checkpoint}"
export NANO_OMNI_MODEL_NAME
export OSWORLD_GRPO_TRAIN_DATA="${OSWORLD_GRPO_TRAIN_DATA:-${DATA_DIR}/validation-1x.jsonl}"
export OSWORLD_GRPO_VAL_DATA="${OSWORLD_GRPO_VAL_DATA:-${DATA_DIR}/validation-1x.jsonl}"
export MOLT_RUN_NAME="${MOLT_RUN_NAME:-osworld361-molt-aligned-s400-rbaseline-r3}"
export MOLT_MAX_STEPS="${MOLT_MAX_STEPS:-300}"

export OSWORLD_NUM_PROMPTS_PER_STEP="${OSWORLD_NUM_PROMPTS_PER_STEP:-8}"
export OSWORLD_NUM_GENERATIONS="${OSWORLD_NUM_GENERATIONS:-8}"
# Resumed async jobs must be able to cycle the finite 361-task prompt set.
# One epoch can be exhausted before max_steps after changing the train-group
# size or restoring the replay/dataloader cursor from a later checkpoint.
export GRPO_MAX_NUM_EPOCHS="${GRPO_MAX_NUM_EPOCHS:-20}"
export MOLT_ASYNC_QUEUE_SIZE="${MOLT_ASYNC_QUEUE_SIZE:-2}"
export MOLT_VLLM_GENERATE_BATCH_SIZE="${MOLT_VLLM_GENERATE_BATCH_SIZE:-8}"
export MOLT_MAX_STALENESS="${MOLT_MAX_STALENESS:-1}"

export NUM_NODES="${NUM_NODES:-4}"
export MOLT_INFERENCE_NODES="${MOLT_INFERENCE_NODES:-3}"
export MOLT_TENSOR_PARALLEL_SIZE="${MOLT_TENSOR_PARALLEL_SIZE:-1}"
export MOLT_EXPERT_MODEL_PARALLEL_SIZE="${MOLT_EXPERT_MODEL_PARALLEL_SIZE:-8}"
export MOLT_CONTEXT_PARALLEL_SIZE="${MOLT_CONTEXT_PARALLEL_SIZE:-8}"
export MOLT_PIPELINE_PARALLEL_SIZE="${MOLT_PIPELINE_PARALLEL_SIZE:-1}"
export MOLT_SEQUENCE_LENGTH_DIVISOR="${MOLT_SEQUENCE_LENGTH_DIVISOR:-32}"

export OSWORLD_LEARNING_RATE="${OSWORLD_LEARNING_RATE:-5e-6}"
export MOLT_ROUTER_REPLAY_ENABLED="${MOLT_ROUTER_REPLAY_ENABLED:-true}"
export MOLT_IS_RATIO_MIN="${MOLT_IS_RATIO_MIN:-0.99}"
export MOLT_IS_RATIO_MAX="${MOLT_IS_RATIO_MAX:-1.01}"

export OSWORLD_MAX_MODEL_LEN="${OSWORLD_MAX_MODEL_LEN:-49152}"
export OSWORLD_MAX_NEW_TOKENS="${OSWORLD_MAX_NEW_TOKENS:-16384}"
export OSWORLD_AGENT_MAX_TOKENS="${OSWORLD_AGENT_MAX_TOKENS:-16384}"
export OSWORLD_MAX_IMAGE_HISTORY_LENGTH="${OSWORLD_MAX_IMAGE_HISTORY_LENGTH:-3}"
export OSWORLD_MAX_ACTIVE_IMAGES="${OSWORLD_MAX_ACTIVE_IMAGES:-10}"
export OSWORLD_VLLM_MAX_IMAGES="${OSWORLD_VLLM_MAX_IMAGES:-20}"
export OSWORLD_CC_KEEP_LAST_IMAGE_GROUPS="${OSWORLD_CC_KEEP_LAST_IMAGE_GROUPS:-2}"
export OSWORLD_CC_ACTIONS_PER_CHUNK="${OSWORLD_CC_ACTIONS_PER_CHUNK:-100}"
export OSWORLD_CC_MAX_TOTAL_TOKENS="${OSWORLD_CC_MAX_TOTAL_TOKENS:-49152}"
export OSWORLD_CC_RESERVED_GENERATION_TOKENS="${OSWORLD_CC_RESERVED_GENERATION_TOKENS:-11152}"
export OSWORLD_MAX_STEPS="${OSWORLD_MAX_STEPS:-150}"
export OSWORLD_ROLLOUT_TIMEOUT_S="${OSWORLD_ROLLOUT_TIMEOUT_S:-1200}"
export OSWORLD_ACTION_TIMEOUT_S="${OSWORLD_ACTION_TIMEOUT_S:-60}"
export OSWORLD_LLM_TIMEOUT_S="${OSWORLD_LLM_TIMEOUT_S:-900}"
export OSWORLD_SLEEP_AFTER_EXECUTION="${OSWORLD_SLEEP_AFTER_EXECUTION:-5}"

export OSWORLD_NEMO_GYM_NUM_WORKERS="${OSWORLD_NEMO_GYM_NUM_WORKERS:-32}"
export OSWORLD_MAX_PARALLEL_ROLLOUTS="${OSWORLD_MAX_PARALLEL_ROLLOUTS:-32}"

export MOLT_SEQUENCE_PACKING_ENABLED="${MOLT_SEQUENCE_PACKING_ENABLED:-true}"
export MOLT_TRAIN_MB_TOKENS="${MOLT_TRAIN_MB_TOKENS:-49152}"
export MOLT_LOGPROB_MB_TOKENS="${MOLT_LOGPROB_MB_TOKENS:-49152}"
export NEMOTRON_OMNI_VISION_CHUNK_SIZE="${NEMOTRON_OMNI_VISION_CHUNK_SIZE:-1}"
export NEMOTRON_OMNI_VISION_CACHE_MAX_ENTRIES="${NEMOTRON_OMNI_VISION_CACHE_MAX_ENTRIES:-0}"
export RAY_ENABLE_ZERO_COPY_TORCH_TENSORS="${RAY_ENABLE_ZERO_COPY_TORCH_TENSORS:-1}"
export MOLT_OPTIMIZER_CPU_OFFLOAD="${MOLT_OPTIMIZER_CPU_OFFLOAD:-true}"
export MOLT_OPTIMIZER_OFFLOAD_FRACTION="${MOLT_OPTIMIZER_OFFLOAD_FRACTION:-1.0}"
export MOLT_OFFLOAD_OPTIMIZER_FOR_LOGPROB="${MOLT_OFFLOAD_OPTIMIZER_FOR_LOGPROB:-true}"

export CHECKPOINTING_ENABLED=true
export CHECKPOINT_SAVE_PERIOD="${CHECKPOINT_SAVE_PERIOD:-1}"
export CHECKPOINT_KEEP_TOP_K="${CHECKPOINT_KEEP_TOP_K:-2}"
export MOLT_SAVE_OPTIMIZER="${MOLT_SAVE_OPTIMIZER:-true}"
export OSWORLD_VAL_PERIOD="${OSWORLD_VAL_PERIOD:-100000}"
export OSWORLD_VAL_AT_START="${OSWORLD_VAL_AT_START:-false}"
export OSWORLD_VAL_AT_END="${OSWORLD_VAL_AT_END:-false}"
export SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
export SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
export OSWORLD_CHECKPOINT_MUST_SAVE_BY="${OSWORLD_CHECKPOINT_MUST_SAVE_BY:-00:03:40:00}"
# Shared Gym environments are already built and protected by flock. Rebuilding
# them on every four-hour segment can itself leave all trainer GPUs idle long
# enough for the occupied-idle reaper to cancel the job before rollout starts.
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-true}"

if [[ "${MOLT_PREFLIGHT:-true}" == "true" ]]; then
  shopt -s nullglob
  model_shards=("${NANO_OMNI_MODEL_NAME}"/model-*.safetensors)
  shopt -u nullglob
  [[ ${#model_shards[@]} -eq 17 ]] || {
    echo "ABORT: ${NANO_OMNI_MODEL_NAME} has ${#model_shards[@]}/17 model shards" >&2
    exit 1
  }
  for required_path in \
    "${NANO_OMNI_MODEL_NAME}/model.safetensors.index.json" \
    "${NANO_OMNI_MODEL_NAME}/config.json" \
    "${OSWORLD_GRPO_TRAIN_DATA}"; do
    [[ -e "${required_path}" ]] || {
      echo "ABORT: required parity path is missing: ${required_path}" >&2
      exit 1
    }
  done
  model_arch="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["architectures"][0])' \
      "${NANO_OMNI_MODEL_NAME}/config.json"
  )"
  [[ "${model_arch}" == "NemotronH_Nano_Omni_Reasoning_V3" ]] || {
    echo "ABORT: unexpected model architecture: ${model_arch}" >&2
    exit 1
  }
  echo "Parity preflight OK: 17 shards, architecture=${model_arch}"
fi

if [[ -n "${MOLT_ALIGNMENT_PROFILE:-}" ]]; then
  python3 "${ROOT}/examples/nemo_gym/verify_osworld_molt_aligned_env.py"
fi

exec "${ROOT}/examples/nemo_gym/submit_osworld_molt_async.sh"
