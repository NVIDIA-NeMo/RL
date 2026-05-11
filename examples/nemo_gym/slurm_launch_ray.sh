#!/usr/bin/env bash
# Bring up an interactive Ray cluster on Slurm via ray.sub.
#
# Slurm directives (account, partition, nodes, time, gres) live in ray.sub's
# `#SBATCH` block — that's the file Slurm actually parses.
#
# After Slurm prints the JOBID, attach with:
#   bash <JOBID>-attach.sh
# then inside the container run:
#   bash examples/nemo_gym/slurm_train_nano3_grpo.sh
set -euo pipefail

# read vars from env file
ENV_FILE="/lustre/fsw/general_sa/xiyu/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Error: env file not found at ${ENV_FILE}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

if [[ -z "${HF_TOKEN}" ]]; then
  echo "Error: HF_TOKEN not set in ${ENV_FILE}" >&2
  exit 1
fi
if [[ -z "${WANDB_API_KEY}" ]]; then
  echo "Error: WANDB_API_KEY not set in ${ENV_FILE}" >&2
  exit 1
fi

export EOS_ROOT=/lustre/fsw/general_sa/xiyu
export PROJECT_ROOT=${EOS_ROOT}/260505_prime_verifiers
export CODE_DIR=${PROJECT_ROOT}/RL
export HF_HOME_DIR=${EOS_ROOT}/hf_home
export RESULT_DIR=${PROJECT_ROOT}/results/nemotron_3_nano_grpo
export ENROOT_CACHE_PATH=${EOS_ROOT}/.enroot

CONTAINER=/lustre/fsw/general_sa/xiyu/containers/nemo-rl-v0.6.0_prime_verifiers.sqsh

mkdir -p "${RESULT_DIR}/checkpoints" "${RESULT_DIR}/logs"

# Mounts:
#   - CODE_DIR -> /opt/nemo-rl: overlays the image-baked NeMo-RL tree (which
#     includes 3rdparty/Gym-workspace/Gym at the OLD submodule pin) with our
#     Lustre checkout. Required so Ray actor venvs — whose editable finders
#     hard-code absolute paths under /opt/nemo-rl/... — see our bumped Gym
#     (e.g. wiki-search.yaml) instead of the image-baked one.
#   - CODE_DIR -> CODE_DIR: kept so any code that hard-codes the absolute
#     lustre path of the repo (recipes, scripts) keeps working.
#   - EOS_ROOT -> EOS_ROOT: data, results, checkpoints at their stable paths.
#   - HF_HOME_DIR -> /opt/hf_home: HF cache at the image-expected path.
MOUNTS="${CODE_DIR}:/opt/nemo-rl,${CODE_DIR}:${CODE_DIR},${EOS_ROOT}:${EOS_ROOT},${HF_HOME_DIR}:/opt/hf_home"

cd "${CODE_DIR}"

# COMMAND is intentionally unset — ray.sub will bring up the Ray cluster
# and leave the head node idle so we can attach interactively.
CONTAINER="${CONTAINER}" \
MOUNTS="${MOUNTS}" \
BASE_LOG_DIR="${RESULT_DIR}/logs" \
RAY_LOG_SYNC_FREQUENCY=60 \
HF_TOKEN="${HF_TOKEN}" \
WANDB_API_KEY="${WANDB_API_KEY}" \
sbatch ray.sub

cat <<EOF

Submitted Ray cluster bring-up. Once the job is RUNNING:
  squeue -u \$USER                             # find <JOBID>
  bash <JOBID>-attach.sh                       # drop into a shell on the head node
  bash examples/nemo_gym/slurm_train_nano3_grpo.sh             # start training inside that shell
EOF
