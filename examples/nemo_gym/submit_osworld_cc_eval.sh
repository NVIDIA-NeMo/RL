#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
: "${OPENSANDBOX_DOMAIN:?Set OPENSANDBOX_DOMAIN}"
: "${OPENSANDBOX_API_KEY:?Set OPENSANDBOX_API_KEY}"
: "${OSWORLD_GRPO_VAL_DATA:?Set OSWORLD_GRPO_VAL_DATA}"
: "${EVAL_NAME:?Set EVAL_NAME}"

EVAL_CHECKPOINT_PATH="${EVAL_CHECKPOINT_PATH:-}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-15}"
NUM_NODES="${NUM_NODES:-2}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-32}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/results/osworld-cc-eval/${EVAL_NAME}}"
BASE_LOG_DIR="${BASE_LOG_DIR:-${RESULTS_DIR}/slurm}"
mkdir -p "${BASE_LOG_DIR}"

PRETRAINED_OVERRIDES=""
if [[ -n "${EVAL_CHECKPOINT_PATH}" ]]; then
  [[ -d "${EVAL_CHECKPOINT_PATH}" ]] || {
    echo "Checkpoint not found: ${EVAL_CHECKPOINT_PATH}" >&2
    exit 2
  }
  PRETRAINED_OVERRIDES="++checkpointing.pretrained_checkpoint.path=${EVAL_CHECKPOINT_PATH} ++checkpointing.pretrained_checkpoint.format=megatron_bridge"
fi

COMMAND="cd ${ROOT} && uv run --locked examples/nemo_gym/run_grpo_nemo_gym.py --config examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_cc.yaml ++policy.train_global_batch_size=2 ++policy.megatron_cfg.validation.eval_global_batch_size=2 ++policy.megatron_cfg.scheduler.lr_decay_iters=1 ${PRETRAINED_OVERRIDES}"

sbatch \
  --nodes="${NUM_NODES}" \
  --gres=gpu:8 \
  --account=coreai_dlalgo_nemorl \
  --partition=batch \
  --job-name="cc-eval-${EVAL_NAME}" \
  --time=04:00:00 \
  --output="${BASE_LOG_DIR}/slurm-%j.out" \
  --export=ALL,OPENSANDBOX_DOMAIN,OPENSANDBOX_API_KEY,OSWORLD_POOL_REF=osworld-kvm,CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_ci/nemo_rl_ci/sqsh_files/rl.61531646.sqsh,MOUNTS=/lustre:/lustre,GPUS_PER_NODE=8,NUM_NODES="${NUM_NODES}",BASE_LOG_DIR="${BASE_LOG_DIR}",NEMO_GYM_EXTRA_ROOTS="${ROOT}/3rdparty/Gym-workspace/Gym",NANO_OMNI_MODEL_NAME=/lustre/fs1/portfolios/coreai/users/aroshanghias/checkpoints/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16,OSWORLD_GRPO_TRAIN_DATA="${OSWORLD_GRPO_VAL_DATA}",OSWORLD_GRPO_VAL_DATA="${OSWORLD_GRPO_VAL_DATA}",OSWORLD_RESULTS_DIR="${RESULTS_DIR}",CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints",CHECKPOINTING_ENABLED=false,GRPO_MAX_NUM_STEPS=0,OSWORLD_NUM_PROMPTS_PER_STEP=2,OSWORLD_NUM_GENERATIONS=1,OSWORLD_MAX_STEPS="${EVAL_MAX_STEPS}",OSWORLD_MAX_MODEL_LEN=16384,OSWORLD_MAX_ACTIVE_IMAGES=8,OSWORLD_CC_KEEP_LAST_IMAGE_GROUPS=2,OSWORLD_CC_ACTIONS_PER_CHUNK=2,OSWORLD_CC_MAX_TOTAL_TOKENS=16384,OSWORLD_CC_RESERVED_GENERATION_TOKENS=1,OSWORLD_NEMO_GYM_NUM_WORKERS="${EVAL_NUM_WORKERS}",OSWORLD_MAX_PARALLEL_ROLLOUTS="${EVAL_NUM_WORKERS}",OSWORLD_USE_DYNAMIC_SAMPLING=false,OSWORLD_VAL_PERIOD=100,OSWORLD_VAL_AT_START=true,OSWORLD_VAL_AT_END=false,OSWORLD_VAL_BATCH_SIZE=32,OSWORLD_TEMPERATURE=0.0,OSWORLD_TOP_P=1.0,WANDB_ENABLED=false,HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/Nemo-RL-Library/osworld-cc-runtime/hf-home,COMMAND="${COMMAND}" \
  "${ROOT}/ray.sub"
