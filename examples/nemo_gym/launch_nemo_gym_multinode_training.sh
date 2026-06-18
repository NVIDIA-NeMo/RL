# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
# ----- PARAMETERS -----
# Optional: WANDB_API_KEY, HF_TOKEN
# Required: EXP_NAME, GPUS_PER_NODE, HF_CKPT_PATH, NEMO_GYM_SWE_TRAIN_DATA_PATH,
# NEMO_GYM_SWE_VALIDATION_DATA_PATH, NEMO_GYM_SWE_SIF_DIR,
# NRL_MEGATRON_CHECKPOINT_DIR, REPO_LOCATION, CONTAINER_IMAGE_PATH,
# SLURM_ACCOUNT, SLURM_PARTITION

require_env() {
    local name="$1"
    local description="$2"

    if [[ -z "${!name:-}" ]]; then
        echo "Error: ${name} must be set." >&2
        echo "  ${description}" >&2
        exit 1
    fi
}

require_path() {
    local name="$1"
    local kind="$2"
    local description="$3"
    local path="${!name}"

    case "${kind}" in
        file)
            if [[ ! -f "${path}" ]]; then
                echo "Error: ${name} must point to an existing file." >&2
                echo "  Current value: ${path}" >&2
                echo "  ${description}" >&2
                exit 1
            fi
            ;;
        dir)
            if [[ ! -d "${path}" ]]; then
                echo "Error: ${name} must point to an existing directory." >&2
                echo "  Current value: ${path}" >&2
                echo "  ${description}" >&2
                exit 1
            fi
            ;;
        path)
            if [[ ! -e "${path}" ]]; then
                echo "Error: ${name} must point to an existing path." >&2
                echo "  Current value: ${path}" >&2
                echo "  ${description}" >&2
                exit 1
            fi
            ;;
        *)
            echo "Internal error: unsupported path kind '${kind}' for ${name}." >&2
            exit 1
            ;;
    esac
}

require_env "EXP_NAME" "Experiment name used for Slurm job name and result directory."
require_env "REPO_LOCATION" "Host checkout path where ray.sub will be submitted."
require_env "CONTAINER_IMAGE_PATH" "Container image path passed to sbatch/ray.sub."
require_env "SLURM_ACCOUNT" "Slurm account for the allocation."
require_env "SLURM_PARTITION" "Slurm partition for the allocation."
require_env "GPUS_PER_NODE" "Number of GPUs to request and advertise per Slurm node."
require_env "HF_CKPT_PATH" "Host path to the HF checkpoint directory mounted as policy.model_name."
require_env "NRL_MEGATRON_CHECKPOINT_DIR" "Host path to the preconverted Megatron checkpoint cache. A conversion is performed the first time, so this can be an empty dir."
require_env "NEMO_GYM_SWE_TRAIN_DATA_PATH" "Host path to the SWE training JSONL."
require_env "NEMO_GYM_SWE_VALIDATION_DATA_PATH" "Host path to the SWE validation JSONL."
require_env "NEMO_GYM_SWE_SIF_DIR" "Host directory containing the SWE task SIF images."
require_path "REPO_LOCATION" "dir" "ray.sub is submitted from this checkout."
require_path "HF_CKPT_PATH" "dir" "This directory is mounted as policy.model_name inside the container."
require_path "NRL_MEGATRON_CHECKPOINT_DIR" "dir" "This directory is mounted as the Megatron checkpoint cache inside the container."
require_path "NEMO_GYM_SWE_TRAIN_DATA_PATH" "file" "This JSONL is mounted as the training dataset inside the container."
require_path "NEMO_GYM_SWE_VALIDATION_DATA_PATH" "file" "This JSONL is mounted as the validation dataset inside the container."
require_path "NEMO_GYM_SWE_SIF_DIR" "dir" "This directory is mounted as the SWE task SIF directory inside the container."
if ! [[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: GPUS_PER_NODE must be a positive integer." >&2
    echo "  Current value: ${GPUS_PER_NODE}" >&2
    exit 1
fi

TRAIN_NODES="${TRAIN_NODES:-16}"
GEN_NODES="${GEN_NODES:-24}"
NODES="${NODES:-$((TRAIN_NODES + GEN_NODES))}"
CONTAINER_REPO_LOCATION="${CONTAINER_REPO_LOCATION:-/opt/nemo-rl}"
RECIPE="${RECIPE:-examples/nemo_gym/grpo_qwen3_235b_swe_openhands_async.yaml}"
CONTAINER_INPUT_ROOT="${CONTAINER_INPUT_ROOT:-/inputs/nemo_gym}"
CONTAINER_HF_CKPT_PATH="${CONTAINER_HF_CKPT_PATH:-${HF_CKPT_PATH}}"
CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR="${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR:-${CONTAINER_INPUT_ROOT}/mcore_ckpt}"
CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH:-${CONTAINER_INPUT_ROOT}/data/train.jsonl}"
CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH:-${CONTAINER_INPUT_ROOT}/data/validation.jsonl}"
CONTAINER_NEMO_GYM_SWE_SIF_DIR="${CONTAINER_NEMO_GYM_SWE_SIF_DIR:-${CONTAINER_INPUT_ROOT}/sif}"

# ray.sub is submitted from the host checkout, but training runs from the
# baked checkout inside the container.
cd "${REPO_LOCATION}"
OUT_DIR="$(pwd)/results/${EXP_NAME}"
HOST_HF_HOME="${HF_HOME:-$(pwd)/.cache}"
export BASE_LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/checkpoint" "${HOST_HF_HOME}"

# Construct the command
COMMAND=$(cat <<EOF
cd ${CONTAINER_REPO_LOCATION}


HF_HOME=${CONTAINER_REPO_LOCATION}/.cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_TOKEN="${HF_TOKEN:-}" \
WANDB_API_KEY="${WANDB_API_KEY:-}" \
HF_CKPT_PATH="${CONTAINER_HF_CKPT_PATH}" \
CONTAINER_HF_CKPT_PATH="${CONTAINER_HF_CKPT_PATH}" \
NRL_MEGATRON_CHECKPOINT_DIR="${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR}" \
NEMO_GYM_SWE_WORKSPACE_ROOT=/logs/nemo_gym/workspace \
NEMO_GYM_SWE_TRAIN_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH}" \
NEMO_GYM_SWE_VALIDATION_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH}" \
NEMO_GYM_SWE_SIF_DIR="${CONTAINER_NEMO_GYM_SWE_SIF_DIR}" \
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config ${RECIPE} \
    ++logger.mlperf_enabled=True \
    ++logger.mlperf.log_file=${OUT_DIR}/logs/mllogger.log \
    ++logger.mlperf.benchmark=grpo_nemo_gym \
    ++logger.mlperf.target_accuracy=${MLPERF_TARGET_ACCURACY:-1.0} \
    ++logger.mlperf.force_success_status=False \
    ++cluster.num_nodes=$NODES \
    ++cluster.gpus_per_node=$GPUS_PER_NODE \
    ++policy.generation.colocated.resources.num_nodes=$GEN_NODES \
    ++policy.generation.colocated.resources.gpus_per_node=$GPUS_PER_NODE \
    ++logger.wandb.name=$EXP_NAME \
    ++logger.log_dir=/logs \
    ++checkpointing.checkpoint_dir=/checkpoint \
    $@
EOF
)

echo -e "Running command:\n$COMMAND"

# OccupiedIdleGPUsJobReaper exemption: async (non-colocated) legitimately idles its
# training-node GPU pool while the replay buffer fills from slow SWE reward computation,
# which otherwise trips the idle-GPU reaper. Override via SLURM_COMMENT env if needed.
SLURM_IDLE_EXEMPT_MINS="${SLURM_IDLE_EXEMPT_MINS:-120}"
SLURM_COMMENT="${SLURM_COMMENT:-{\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"${SLURM_IDLE_EXEMPT_MINS}\",\"reason\":\"rl-rollout-warmup\",\"description\":\"NeMo-RL GRPO: training GPUs idle during rollout/SWE-reward buffer-fill\"}}}"


# Host paths above are mounted to stable container paths consumed by the YAML.
# The logs dir is ALSO identity-mounted (host path -> same path in container)
# because ray.sub uses the host-side BASE_LOG_DIR=${OUT_DIR}/logs as $LOG_DIR
# both on the host (-o redirects, mkdir) and inside the head container
# (`touch $LOG_DIR/STARTED_RAY_HEAD`). Without the identity mount that touch
# fails and the launcher waits forever for the cluster to come up.
MOUNTS="${OUT_DIR}/logs:${OUT_DIR}/logs,${HOST_HF_HOME}:${CONTAINER_REPO_LOCATION}/.cache,${OUT_DIR}/checkpoint:/checkpoint,${OUT_DIR}/logs:/logs"
MOUNTS="${MOUNTS},${HF_CKPT_PATH}:${HF_CKPT_PATH}"
MOUNTS="${MOUNTS},${NRL_MEGATRON_CHECKPOINT_DIR}:${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR}"
MOUNTS="${MOUNTS},${NEMO_GYM_SWE_TRAIN_DATA_PATH}:${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH}"
MOUNTS="${MOUNTS},${NEMO_GYM_SWE_VALIDATION_DATA_PATH}:${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH}"
MOUNTS="${MOUNTS},${NEMO_GYM_SWE_SIF_DIR}:${CONTAINER_NEMO_GYM_SWE_SIF_DIR}"
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
    MOUNTS="${MOUNTS},${EXTRA_MOUNTS}"
fi

MOUNTS=$MOUNTS \
COMMAND=$COMMAND \
CONTAINER=$CONTAINER_IMAGE_PATH \
CONTAINER_WORKDIR=$CONTAINER_REPO_LOCATION \
GPUS_PER_NODE=$GPUS_PER_NODE \
sbatch \
    --nodes=$NODES \
    --account=$SLURM_ACCOUNT \
    --partition=$SLURM_PARTITION \
    --time=${SLURM_TIME:-1:0:0} \
    --job-name=$EXP_NAME \
    --comment="$SLURM_COMMENT" \
    ${SLURM_EXCLUDE:+--exclude=$SLURM_EXCLUDE} \
    ray.sub
