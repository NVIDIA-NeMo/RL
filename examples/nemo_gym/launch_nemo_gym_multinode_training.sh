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
if [[ -n "${NEMO_GYM_SWE_FALLBACK_SIF_DIR:-}" ]]; then
    require_path "NEMO_GYM_SWE_FALLBACK_SIF_DIR" "dir" "Optional fallback directory for R2E SIF images not present under NEMO_GYM_SWE_SIF_DIR."
fi
NEMO_GYM_SWE_TRAIN_DATA_DIR="$(dirname "${NEMO_GYM_SWE_TRAIN_DATA_PATH}")"
NEMO_GYM_SWE_VALIDATION_DATA_DIR="$(dirname "${NEMO_GYM_SWE_VALIDATION_DATA_PATH}")"
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
if [[ "${HF_CKPT_PATH}" == */snapshots/* ]]; then
    HF_MODEL_CACHE_DIR="${HF_MODEL_CACHE_DIR:-$(dirname "$(dirname "${HF_CKPT_PATH}")")}"
    CONTAINER_HF_MODEL_CACHE_DIR="${CONTAINER_HF_MODEL_CACHE_DIR:-$(dirname "$(dirname "${CONTAINER_HF_CKPT_PATH}")")}"
fi
CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR="${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR:-${CONTAINER_INPUT_ROOT}/mcore_ckpt}"
CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH:-${CONTAINER_INPUT_ROOT}/data/train.jsonl}"
CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH:-${CONTAINER_INPUT_ROOT}/data/validation.jsonl}"
CONTAINER_NEMO_GYM_SWE_SIF_DIR="${CONTAINER_NEMO_GYM_SWE_SIF_DIR:-${CONTAINER_INPUT_ROOT}/sif}"
MLPERF_SUBMISSION_ORG="${MLPERF_SUBMISSION_ORG:-reference}"
MLPERF_SUBMISSION_PLATFORM="${MLPERF_SUBMISSION_PLATFORM:-reference}"

# Pick a fresh random seed every launch unless one is provided explicitly via GRPO_SEED.
# Combine two $RANDOM draws (each 0-32767) into a wider 0..~2^30 range.
GRPO_SEED="${GRPO_SEED:-$(( (RANDOM << 15) | RANDOM ))}"
echo "Using grpo.seed=${GRPO_SEED}"

# ray.sub is submitted from the host checkout, but training runs from the
# baked checkout inside the container.
cd "${REPO_LOCATION}"
OUT_DIR="$(pwd)/results/${EXP_NAME}"
HOST_HF_HOME="${HF_HOME:-$(pwd)/.cache}"
export BASE_LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/checkpoint" "${HOST_HF_HOME}"


# Qwen 3.5 support lives in qwen_35/ so non-Qwen recipes keep the baked
# container code. Selecting a recipe under qwen_35/ automatically mounts the
# Qwen-specific config and runtime overlay into the container.
_qwen35_append_extra_mount() {
    local mount="$1"
    case ",${EXTRA_MOUNTS:-}," in
        *",${mount},"*) ;;
        *) export EXTRA_MOUNTS="${EXTRA_MOUNTS:+${EXTRA_MOUNTS},}${mount}" ;;
    esac
}

_qwen35_stage_tree() {
    local host_root="$1"
    local stage_root="$2"
    local description="$3"

    if [[ ! -d "${host_root}" ]]; then
        echo "Error: ${description} directory does not exist: ${host_root}" >&2
        exit 1
    fi

    mkdir -p "${stage_root}"
    cp -a "${host_root}/." "${stage_root}/"
    echo "Staged ${description}: ${host_root} -> ${stage_root}"
}

_qwen35_mount_tree() {
    local host_root="$1"
    local container_root="$2"
    local description="$3"

    if [[ ! -d "${host_root}" ]]; then
        echo "Error: ${description} directory does not exist: ${host_root}" >&2
        exit 1
    fi

    local src rel
    while IFS= read -r -d '' src; do
        rel="${src#${host_root}/}"
        _qwen35_append_extra_mount "${src}:${container_root}/${rel}"
    done < <(find "${host_root}" -type f -print0)
}

_qwen35_stage_file() {
    local host_file="$1"
    local stage_root="$2"
    local rel="$3"
    local description="$4"

    if [[ ! -f "${host_file}" ]]; then
        echo "Error: ${description} file does not exist: ${host_file}" >&2
        exit 1
    fi

    mkdir -p "$(dirname "${stage_root}/${rel}")"
    cp -a "${host_file}" "${stage_root}/${rel}"
    echo "Staged ${description}: ${host_file} -> ${stage_root}/${rel}"
}

_qwen35_mount_file() {
    local host_file="$1"
    local container_file="$2"
    local description="$3"

    if [[ ! -f "${host_file}" ]]; then
        echo "Error: ${description} staged file does not exist: ${host_file}" >&2
        exit 1
    fi

    _qwen35_append_extra_mount "${host_file}:${container_file}"
}

_qwen35_overlay_mode="${QWEN35_OVERLAY:-auto}"
_qwen35_recipe="${RECIPE#./}"
_qwen35_should_mount=0
case "${_qwen35_overlay_mode}" in
    0|false|False|no|NO) _qwen35_should_mount=0 ;;
    1|true|True|yes|YES) _qwen35_should_mount=1 ;;
    auto)
        if [[ "${_qwen35_recipe}" == qwen_35/* ]]; then
            _qwen35_should_mount=1
        fi
        ;;
    *)
        echo "Error: QWEN35_OVERLAY must be auto, 0, or 1; got '${_qwen35_overlay_mode}'." >&2
        exit 1
        ;;
esac

if [[ "${_qwen35_should_mount}" == "1" ]]; then
    QWEN35_CONFIG_DIR="${QWEN35_CONFIG_DIR:-${REPO_LOCATION}/qwen_35/configs}"
    QWEN35_OVERLAY_DIR="${QWEN35_OVERLAY_DIR:-${REPO_LOCATION}/qwen_35/overrides}"
    QWEN35_MOUNT_STAGE_DIR="${QWEN35_MOUNT_STAGE_DIR:-${OUT_DIR}/qwen_35_mounts}"
    QWEN35_STAGED_CONFIG_DIR="${QWEN35_MOUNT_STAGE_DIR}/configs"
    QWEN35_STAGED_OVERLAY_DIR="${QWEN35_MOUNT_STAGE_DIR}/overrides"

    _qwen35_stage_tree "${QWEN35_CONFIG_DIR}" "${QWEN35_STAGED_CONFIG_DIR}" "Qwen 3.5 config"
    _qwen35_mount_tree "${QWEN35_STAGED_CONFIG_DIR}" "${CONTAINER_REPO_LOCATION}/qwen_35/configs" "Qwen 3.5 staged config"

    _qwen35_overlay_dsts=(
        "nemo_rl/environments/nemo_gym.py"
        "nemo_rl/models/generation/vllm/vllm_worker_async.py"
        "nemo_rl/models/megatron/setup.py"
        "nemo_rl/models/policy/workers/megatron_policy_worker.py"
        "3rdparty/Gym-workspace/Gym/responses_api_models/vllm_model/app.py"
    )
    for _qwen35_dst_rel in "${_qwen35_overlay_dsts[@]}"; do
        _qwen35_stage_file "${QWEN35_OVERLAY_DIR}/${_qwen35_dst_rel}" "${QWEN35_STAGED_OVERLAY_DIR}" "${_qwen35_dst_rel}" "Qwen 3.5 overlay"
        _qwen35_mount_file "${QWEN35_STAGED_OVERLAY_DIR}/${_qwen35_dst_rel}" "${CONTAINER_REPO_LOCATION}/${_qwen35_dst_rel}" "Qwen 3.5 overlay"
    done
    unset _qwen35_dst_rel _qwen35_overlay_dsts

    # Defaults consumed by the Qwen-only files. They are harmless for
    # non-Qwen jobs because these files are not mounted for those recipes.
    export NEMO_RL_QWEN35_TRUNCATE_PROMPT_TOKENS="${NEMO_RL_QWEN35_TRUNCATE_PROMPT_TOKENS:-${QWEN35_TRUNCATE_PROMPT_TOKENS:-65535}}"
fi
unset _qwen35_overlay_mode _qwen35_recipe _qwen35_should_mount
unset -f _qwen35_append_extra_mount _qwen35_stage_tree _qwen35_mount_tree _qwen35_stage_file _qwen35_mount_file

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
MLPERF_SUBMISSION_ORG="${MLPERF_SUBMISSION_ORG}" \
MLPERF_SUBMISSION_PLATFORM="${MLPERF_SUBMISSION_PLATFORM}" \
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config ${RECIPE} \
    ++logger.mlperf_enabled=True \
    ++logger.mlperf.log_file=${OUT_DIR}/logs/mllogger.log \
    ++logger.mlperf.benchmark=qwen3_235b_grpo_swe \
    ++logger.mlperf.target_accuracy=${MLPERF_TARGET_ACCURACY:-1.0} \
    ++logger.mlperf.force_success_status=False \
    ++cluster.num_nodes=$NODES \
    ++cluster.gpus_per_node=$GPUS_PER_NODE \
    ++policy.generation.colocated.resources.num_nodes=$GEN_NODES \
    ++policy.generation.colocated.resources.gpus_per_node=$GPUS_PER_NODE \
    ++logger.wandb.name=$EXP_NAME \
    ++logger.log_dir=/logs \
    ++checkpointing.checkpoint_dir=/checkpoint \
    ++grpo.seed=${GRPO_SEED} \
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
if [[ -n "${HF_MODEL_CACHE_DIR:-}" ]]; then
    MOUNTS="${MOUNTS},${HF_MODEL_CACHE_DIR}:${CONTAINER_HF_MODEL_CACHE_DIR}"
fi
MOUNTS="${MOUNTS},${HF_CKPT_PATH}:${HF_CKPT_PATH}"
MOUNTS="${MOUNTS},${NRL_MEGATRON_CHECKPOINT_DIR}:${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR}"
MOUNTS="${MOUNTS},${NEMO_GYM_SWE_TRAIN_DATA_PATH}:${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH}"
MOUNTS="${MOUNTS},${NEMO_GYM_SWE_VALIDATION_DATA_PATH}:${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH}"
# Compatibility mount: older launch recipes pass host-side data.train.data_path
# overrides. Keep the narrow /inputs mounts above as the canonical container
# paths, but also make the JSONL parent directory visible at the host path so
# those overrides do not fail inside the container.
MOUNTS="${MOUNTS},${NEMO_GYM_SWE_TRAIN_DATA_DIR}:${NEMO_GYM_SWE_TRAIN_DATA_DIR}"
if [[ "${NEMO_GYM_SWE_VALIDATION_DATA_DIR}" != "${NEMO_GYM_SWE_TRAIN_DATA_DIR}" ]]; then
    MOUNTS="${MOUNTS},${NEMO_GYM_SWE_VALIDATION_DATA_DIR}:${NEMO_GYM_SWE_VALIDATION_DATA_DIR}"
fi
MOUNTS="${MOUNTS},${NEMO_GYM_SWE_SIF_DIR}:${CONTAINER_NEMO_GYM_SWE_SIF_DIR}"
# Compatibility mount for configs that pass host-side sif_dir or include
# absolute host-side container_formatter entries.
MOUNTS="${MOUNTS},${NEMO_GYM_SWE_SIF_DIR}:${NEMO_GYM_SWE_SIF_DIR}"
if [[ -n "${NEMO_GYM_SWE_FALLBACK_SIF_DIR:-}" ]]; then
    MOUNTS="${MOUNTS},${NEMO_GYM_SWE_FALLBACK_SIF_DIR}:${NEMO_GYM_SWE_FALLBACK_SIF_DIR}"
fi
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
    MOUNTS="${MOUNTS},${EXTRA_MOUNTS}"
fi

SBATCH_EXTRA_ARGS=()
SBATCH_QOS_VALUE="${SLURM_QOS:-${SBATCH_QOS:-}}"
SBATCH_GRES_VALUE="${SLURM_GRES:-${SBATCH_GRES:-}}"
SBATCH_SEGMENT_VALUE="${SLURM_SEGMENT:-${SBATCH_SEGMENT:-}}"
if [[ -n "${SBATCH_QOS_VALUE}" ]]; then
    SBATCH_EXTRA_ARGS+=("--qos=${SBATCH_QOS_VALUE}")
fi
if [[ -n "${SBATCH_GRES_VALUE}" ]]; then
    SBATCH_EXTRA_ARGS+=("--gres=${SBATCH_GRES_VALUE}")
fi
if [[ -n "${SBATCH_SEGMENT_VALUE}" ]]; then
    SBATCH_EXTRA_ARGS+=("--segment=${SBATCH_SEGMENT_VALUE}")
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
    "${SBATCH_EXTRA_ARGS[@]}" \
    ${SLURM_EXCLUDE:+--exclude=$SLURM_EXCLUDE} \
    ray.sub
