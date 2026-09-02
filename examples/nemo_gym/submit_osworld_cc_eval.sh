#!/usr/bin/env bash
set -Eeuo pipefail

physical_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if [[ "${physical_root}" == /lustre/fs?/* ]]; then
  ROOT="/lustre/fsw/${physical_root#/*/*/}"
else
  ROOT="${physical_root}"
fi
RUNTIME_ROOT="${OSWORLD_RUNTIME_ROOT:-$(dirname "${ROOT}")/osworld-cc-runtime}"
if [[ -z "${OPENSANDBOX_DOMAIN:-}" && -n "${OPENSANDBOX_BASE_URL:-}" ]]; then
  opensandbox_host="${OPENSANDBOX_BASE_URL#*://}"
  export OPENSANDBOX_DOMAIN="${opensandbox_host%%/*}"
fi
: "${OPENSANDBOX_DOMAIN:?Set OPENSANDBOX_DOMAIN or OPENSANDBOX_BASE_URL}"
: "${OPENSANDBOX_API_KEY:?Set OPENSANDBOX_API_KEY}"
: "${OSWORLD_GRPO_VAL_DATA:?Set OSWORLD_GRPO_VAL_DATA}"
: "${EVAL_NAME:?Set EVAL_NAME}"

EVAL_CHECKPOINT_PATH="${EVAL_CHECKPOINT_PATH:-}"
NANO_OMNI_MODEL_NAME="${NANO_OMNI_MODEL_NAME:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/jianh/projects/OpenRLHF-main/ckpts/rfc0037-sft-step400}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/nemo_rl_ci/sqsh_files/rl-gym.65293387.sqsh}"
# Jianh's standalone 361-task scoring protocol uses 100 turns even though the
# training rollout budget is 150.
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-100}"
NUM_NODES="${NUM_NODES:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-32}"
EVAL_VAL_BATCH_SIZE="${EVAL_VAL_BATCH_SIZE:-361}"
EVAL_NUM_GENERATIONS="${EVAL_NUM_GENERATIONS:-4}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"
EVAL_TOP_P="${EVAL_TOP_P:-1.0}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-49152}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-16384}"
EVAL_MAX_ACTIVE_IMAGES="${EVAL_MAX_ACTIVE_IMAGES:-10}"
EVAL_VLLM_MAX_IMAGES="${EVAL_VLLM_MAX_IMAGES:-20}"
EVAL_SLEEP_AFTER_EXECUTION="${EVAL_SLEEP_AFTER_EXECUTION:-3}"
RESULTS_DIR="${RESULTS_DIR:-${RUNTIME_ROOT}/results/osworld-cc-eval/${EVAL_NAME}}"
BASE_LOG_DIR="${BASE_LOG_DIR:-${RESULTS_DIR}/slurm}"
GYM_ROOT="${ROOT}/3rdparty/Gym-workspace/Gym"
HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf-home}"
HF_MODULES_CACHE="${HF_MODULES_CACHE:-${RUNTIME_ROOT}/hf-modules-cache}"
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${RUNTIME_ROOT}/venvs/eval-${EVAL_NAME}/nemo-rl}"
PYTHONPATH="${HF_MODULES_CACHE}:${GYM_ROOT}:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
SETUP_COMMAND="${SETUP_COMMAND:-/opt/nemo_rl_venv/bin/pip install --quiet --no-input tensordict pyvers gprof2dot wandb==0.28.1}"
mkdir -p "${BASE_LOG_DIR}" "${HF_HOME}" "${HF_MODULES_CACHE}" "${NRL_MEGATRON_CHECKPOINT_DIR}"

PRETRAINED_OVERRIDES=""
if [[ -n "${EVAL_CHECKPOINT_PATH}" ]]; then
  [[ -d "${EVAL_CHECKPOINT_PATH}" ]] || {
    echo "Checkpoint not found: ${EVAL_CHECKPOINT_PATH}" >&2
    exit 2
  }
  PRETRAINED_OVERRIDES="++checkpointing.pretrained_checkpoint.path=${EVAL_CHECKPOINT_PATH} ++checkpointing.pretrained_checkpoint.format=megatron_bridge"
fi

PARSER="${ROOT}/nemo_rl/models/generation/vllm/reasoning_parsers/nano_v3_reasoning_parser.py"
EVAL_PYTHON_RUNNER="${EVAL_PYTHON_RUNNER:-uv run --locked}"
COMMAND="cd ${ROOT} && ${EVAL_PYTHON_RUNNER} examples/nemo_gym/run_grpo_nemo_gym.py --config examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_cc.yaml policy.generation.vllm_cfg.reasoning_parser_plugin=${PARSER} ++policy.train_global_batch_size=2 ++policy.megatron_cfg.validation.eval_global_batch_size=2 ++policy.megatron_cfg.scheduler.lr_decay_iters=1 ++policy.megatron_cfg.tensor_model_parallel_size=1 ++policy.megatron_cfg.expert_model_parallel_size=8 ++policy.megatron_cfg.pipeline_model_parallel_size=1 ++policy.megatron_cfg.context_parallel_size=8 ++policy.sequence_packing.enabled=true ${PRETRAINED_OVERRIDES}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-true}"

sbatch \
  --nodes="${NUM_NODES}" \
  --gres=gpu:8 \
  --account=coreai_dlalgo_nemorl \
  --partition="${SBATCH_PARTITION}" \
  --job-name="cc-eval-${EVAL_NAME}" \
  --time="${SBATCH_TIME}" \
  --output="${BASE_LOG_DIR}/slurm-%j.out" \
  --export=ALL,OPENSANDBOX_DOMAIN,OPENSANDBOX_API_KEY,OSWORLD_POOL_REF=osworld-kvm,CONTAINER="${CONTAINER}",MOUNTS=/lustre:/lustre,GPUS_PER_NODE=8,NUM_NODES="${NUM_NODES}",BASE_LOG_DIR="${BASE_LOG_DIR}",NEMO_GYM_EXTRA_ROOTS="${GYM_ROOT}:${ROOT}",NANO_OMNI_MODEL_NAME="${NANO_OMNI_MODEL_NAME}",OSWORLD_GRPO_TRAIN_DATA="${OSWORLD_GRPO_VAL_DATA}",OSWORLD_GRPO_VAL_DATA="${OSWORLD_GRPO_VAL_DATA}",OSWORLD_RESULTS_DIR="${RESULTS_DIR}",CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints",CHECKPOINTING_ENABLED=false,GRPO_MAX_NUM_STEPS=0,OSWORLD_NUM_PROMPTS_PER_STEP=2,OSWORLD_NUM_GENERATIONS=1,OSWORLD_NUM_VAL_GENERATIONS="${EVAL_NUM_GENERATIONS}",OSWORLD_MAX_STEPS="${EVAL_MAX_STEPS}",OSWORLD_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN}",OSWORLD_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS}",OSWORLD_AGENT_MAX_TOKENS="${EVAL_MAX_NEW_TOKENS}",OSWORLD_MAX_IMAGE_HISTORY_LENGTH=3,OSWORLD_MAX_ACTIVE_IMAGES="${EVAL_MAX_ACTIVE_IMAGES}",OSWORLD_VLLM_MAX_IMAGES="${EVAL_VLLM_MAX_IMAGES}",OSWORLD_CC_KEEP_LAST_IMAGE_GROUPS=2,OSWORLD_CC_ACTIONS_PER_CHUNK=100,OSWORLD_CC_MAX_TOTAL_TOKENS="${EVAL_MAX_MODEL_LEN}",OSWORLD_CC_RESERVED_GENERATION_TOKENS=11152,OSWORLD_SLEEP_AFTER_EXECUTION="${EVAL_SLEEP_AFTER_EXECUTION}",OSWORLD_NEMO_GYM_NUM_WORKERS="${EVAL_NUM_WORKERS}",OSWORLD_MAX_PARALLEL_ROLLOUTS="${EVAL_NUM_WORKERS}",OSWORLD_USE_DYNAMIC_SAMPLING=false,OSWORLD_VAL_PERIOD=100000,OSWORLD_VAL_AT_START=true,OSWORLD_VAL_AT_END=false,OSWORLD_VAL_BATCH_SIZE="${EVAL_VAL_BATCH_SIZE}",OSWORLD_TEMPERATURE="${EVAL_TEMPERATURE}",OSWORLD_TOP_P="${EVAL_TOP_P}",WANDB_ENABLED=false,NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS}",NRL_IGNORE_VERSION_MISMATCH=1,HF_HOME="${HF_HOME}",HF_MODULES_CACHE="${HF_MODULES_CACHE}",NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR}",UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT}",PYTHONPATH="${PYTHONPATH}",SETUP_COMMAND="${SETUP_COMMAND}",COMMAND="${COMMAND}" \
  "${ROOT}/ray.sub"
