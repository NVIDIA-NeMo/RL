#!/usr/bin/env bash
set -Eeuo pipefail

physical_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if [[ "${physical_root}" == /lustre/fs?/* ]]; then
  ROOT="/lustre/fsw/${physical_root#/*/*/}"
else
  ROOT="${physical_root}"
fi
RUNTIME_ROOT="${OSWORLD_RUNTIME_ROOT:-$(dirname "${ROOT}")/osworld-cc-runtime}"
PRIVATE_ENV="${OSWORLD_PRIVATE_ENV:-}"

if [[ -n "${PRIVATE_ENV}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${PRIVATE_ENV}"
  set +a
fi

if [[ -z "${OPENSANDBOX_DOMAIN:-}" && -n "${OPENSANDBOX_BASE_URL:-}" ]]; then
  opensandbox_host="${OPENSANDBOX_BASE_URL#*://}"
  export OPENSANDBOX_DOMAIN="${opensandbox_host%%/*}"
fi

: "${OPENSANDBOX_DOMAIN:?Set OPENSANDBOX_DOMAIN or OSWORLD_PRIVATE_ENV}"
: "${OPENSANDBOX_API_KEY:?Set OPENSANDBOX_API_KEY or OSWORLD_PRIVATE_ENV}"
: "${OSWORLD_GRPO_TRAIN_DATA:?Set OSWORLD_GRPO_TRAIN_DATA}"
: "${OSWORLD_GRPO_VAL_DATA:?Set OSWORLD_GRPO_VAL_DATA}"

export NUM_NODES="${NUM_NODES:-3}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export MOLT_INFERENCE_NODES="${MOLT_INFERENCE_NODES:-1}"
export MOLT_EXPERT_MODEL_PARALLEL_SIZE="${MOLT_EXPERT_MODEL_PARALLEL_SIZE:-16}"
export MOLT_CONTEXT_PARALLEL_SIZE="${MOLT_CONTEXT_PARALLEL_SIZE:-2}"
export MOLT_PIPELINE_PARALLEL_SIZE="${MOLT_PIPELINE_PARALLEL_SIZE:-1}"
export MOLT_RUN_NAME="${MOLT_RUN_NAME:-osworld-cc-molt-async-smoke}"
export MOLT_MAX_STEPS="${MOLT_MAX_STEPS:-1}"
export MOLT_MAX_STALENESS="${MOLT_MAX_STALENESS:-1}"
export MOLT_ASYNC_QUEUE_SIZE="${MOLT_ASYNC_QUEUE_SIZE:-1}"
export MOLT_IS_RATIO_MIN="${MOLT_IS_RATIO_MIN:-0.95}"
export MOLT_IS_RATIO_MAX="${MOLT_IS_RATIO_MAX:-1.05}"
export OSWORLD_RESULTS_DIR="${OSWORLD_RESULTS_DIR:-${RUNTIME_ROOT}/results/${MOLT_RUN_NAME}}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OSWORLD_RESULTS_DIR}/checkpoints}"
export BASE_LOG_DIR="${BASE_LOG_DIR:-${RUNTIME_ROOT}/slurm-logs}"

mkdir -p "${OSWORLD_RESULTS_DIR}" "${CHECKPOINT_DIR}" "${BASE_LOG_DIR}"

JOB_NAME="molt-${MOLT_RUN_NAME}"
active="$(squeue -h -u "${USER}" -n "${JOB_NAME}" -o '%i %j %T')"
if [[ -n "${active}" ]]; then
  echo "Refusing duplicate submission; active job:" >&2
  echo "${active}" >&2
  exit 3
fi

export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/nemo_rl_ci/sqsh_files/rl.63363213.sqsh}"
export MOUNTS="${MOUNTS:-/lustre:/lustre}"
export NANO_OMNI_MODEL_NAME="${NANO_OMNI_MODEL_NAME:-/lustre/fsw/portfolios/coreai/users/aroshanghias/checkpoints/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
GYM_ROOT="${ROOT}/3rdparty/Gym-workspace/Gym"
# Gym config paths are repo-relative, while component server directories are
# relative to the Gym checkout. Keep both roots, with the component root first.
export NEMO_GYM_EXTRA_ROOTS="${GYM_ROOT}:${ROOT}"
export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf-home}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${RUNTIME_ROOT}/hf-modules-cache}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl}"
# Put the local Gym checkout before the container copy. This is required for
# component actors to use local setup fixes (notably the Python 3.13-compatible
# headless OpenCV normalization in nemo_gym.cli.setup_command).
export PYTHONPATH="${HF_MODULES_CACHE}:${GYM_ROOT}:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OSWORLD_POOL_REF="${OSWORLD_POOL_REF:-${OPENSANDBOX_POOL_REF:-osworld-kvm}}"
# Never reuse a login-node .venv inside the training container.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${RUNTIME_ROOT}/venvs/${MOLT_RUN_NAME}/nemo-rl}"
mkdir -p "${HF_HOME}" "${HF_MODULES_CACHE}" "${NRL_MEGATRON_CHECKPOINT_DIR}"

export GRPO_MAX_NUM_STEPS="${MOLT_MAX_STEPS}"
export GRPO_MAX_NUM_EPOCHS=1
export OSWORLD_NUM_PROMPTS_PER_STEP="${OSWORLD_NUM_PROMPTS_PER_STEP:-1}"
export OSWORLD_NUM_GENERATIONS="${OSWORLD_NUM_GENERATIONS:-2}"
export OSWORLD_MAX_STEPS="${OSWORLD_MAX_STEPS:-11}"
# Grow from three live screenshots to ten, then compact back to two
# historical screenshots plus the current one.
export OSWORLD_MAX_MODEL_LEN="${OSWORLD_MAX_MODEL_LEN:-49152}"
export OSWORLD_MAX_NEW_TOKENS="${OSWORLD_MAX_NEW_TOKENS:-4096}"
export OSWORLD_MAX_ACTIVE_IMAGES="${OSWORLD_MAX_ACTIVE_IMAGES:-10}"
# Bound the frozen RADIO forward peak independently of the number of images
# retained by context compaction. Outputs are concatenated in original order.
export NEMOTRON_OMNI_VISION_CHUNK_SIZE="${NEMOTRON_OMNI_VISION_CHUNK_SIZE:-1}"
# Cache one optimizer window's projected frozen image features in host BF16.
# The first logprob pass runs with optimizer/gradient memory offloaded; later
# reference-logprob and train passes reuse these entries and skip RADIO.
export NEMOTRON_OMNI_VISION_CACHE_MAX_ENTRIES="${NEMOTRON_OMNI_VISION_CACHE_MAX_ENTRIES:-64}"
export MOLT_OFFLOAD_OPTIMIZER_FOR_LOGPROB="${MOLT_OFFLOAD_OPTIMIZER_FOR_LOGPROB:-true}"
# The guard must measure the 11th candidate before compacting; vLLM therefore
# needs one extra preflight slot beyond the 10-image live-context watermark.
export OSWORLD_VLLM_MAX_IMAGES="${OSWORLD_VLLM_MAX_IMAGES:-11}"
export OSWORLD_CC_KEEP_LAST_IMAGE_GROUPS="${OSWORLD_CC_KEEP_LAST_IMAGE_GROUPS:-2}"
# Keep scheduled compaction above the image watermark so the 10-image guard
# selects the boundary, reproducing the 3 -> 10 -> 3 sawtooth.
export OSWORLD_CC_ACTIONS_PER_CHUNK="${OSWORLD_CC_ACTIONS_PER_CHUNK:-100}"
export OSWORLD_CC_MAX_TOTAL_TOKENS="${OSWORLD_CC_MAX_TOTAL_TOKENS:-49152}"
export OSWORLD_CC_RESERVED_GENERATION_TOKENS="${OSWORLD_CC_RESERVED_GENERATION_TOKENS:-4096}"
export OSWORLD_NEMO_GYM_NUM_WORKERS="${OSWORLD_NEMO_GYM_NUM_WORKERS:-2}"
export OSWORLD_MAX_PARALLEL_ROLLOUTS="${OSWORLD_MAX_PARALLEL_ROLLOUTS:-2}"
export OSWORLD_USE_DYNAMIC_SAMPLING=false
export OSWORLD_LEARNING_RATE="${OSWORLD_LEARNING_RATE:-5e-6}"
export OSWORLD_TEMPERATURE="${OSWORLD_TEMPERATURE:-1.0}"
export OSWORLD_TOP_P=1.0

export CHECKPOINTING_ENABLED="${CHECKPOINTING_ENABLED:-true}"
export CHECKPOINT_SAVE_PERIOD="${CHECKPOINT_SAVE_PERIOD:-8}"
export OSWORLD_VAL_PERIOD="${OSWORLD_VAL_PERIOD:-100000}"
export OSWORLD_VAL_AT_START="${OSWORLD_VAL_AT_START:-false}"
export OSWORLD_VAL_AT_END="${OSWORLD_VAL_AT_END:-false}"
export OSWORLD_VAL_BATCH_SIZE="${OSWORLD_VAL_BATCH_SIZE:-71}"

export WANDB_ENABLED="${WANDB_ENABLED:-false}"
export WANDB_PROJECT="${WANDB_PROJECT:-osworld-context-compaction}"
export WANDB_GROUP="${WANDB_GROUP:-osworld-cc-molt-async}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${MOLT_RUN_NAME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export UV_LOCK_TIMEOUT="${UV_LOCK_TIMEOUT:-900}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-true}"
export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-1}"
export SETUP_COMMAND="${SETUP_COMMAND:-/opt/nemo_rl_venv/bin/pip install --quiet --no-input tensordict pyvers wandb==0.21.0}"

CONFIG="${ROOT}/examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_cc_molt_async.yaml"
PARSER="${ROOT}/nemo_rl/models/generation/vllm/reasoning_parsers/nano_v3_reasoning_parser.py"
export COMMAND="cd ${ROOT} && uv run --locked examples/nemo_gym/run_grpo_nemo_gym.py --config ${CONFIG} policy.generation.vllm_cfg.reasoning_parser_plugin=${PARSER}"

sbatch --parsable \
  --chdir="${RUNTIME_ROOT}" \
  --nodes="${NUM_NODES}" \
  --gres="gpu:${GPUS_PER_NODE}" \
  --account="${SBATCH_ACCOUNT:-coreai_dlalgo_nemorl}" \
  --partition="${SBATCH_PARTITION:-batch}" \
  --job-name="${JOB_NAME}" \
  --time="${SBATCH_TIME:-04:00:00}" \
  --output="${BASE_LOG_DIR}/slurm-%j.out" \
  --export=ALL \
  "${ROOT}/ray.sub"
