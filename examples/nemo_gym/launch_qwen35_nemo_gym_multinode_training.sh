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
# Qwen 3.5-specific NeMo-Gym launcher. The container is expected to be built
# from this branch. By default the host checkout's source dirs (nemo_rl/,
# examples/, qwen_35/) are overlay-mounted over the baked container checkout so
# local fixes take effect without an image rebuild; dependency layers
# (pyproject.toml, uv.lock, venvs, patched 3rdparty Gym) stay baked. Set
# NRL_SOURCE_OVERLAY=0 to run the baked sources only.
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
NEMO_GYM_SWE_TRAIN_DATA_DIR="$(dirname "${NEMO_GYM_SWE_TRAIN_DATA_PATH}")"
NEMO_GYM_SWE_VALIDATION_DATA_DIR="$(dirname "${NEMO_GYM_SWE_VALIDATION_DATA_PATH}")"
require_positive_integer() {
    local name="$1"
    local value="${!name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: ${name} must be a positive integer; got '${value}'." >&2
        exit 1
    fi
}

require_positive_integer GPUS_PER_NODE

TRAIN_NODES="${TRAIN_NODES:-32}"
GEN_NODES="${GEN_NODES:-32}"
NODES="${NODES:-$((TRAIN_NODES + GEN_NODES))}"
require_positive_integer TRAIN_NODES
require_positive_integer GEN_NODES
require_positive_integer NODES
if (( NODES <= GEN_NODES )); then
    echo "Error: NODES (${NODES}) must exceed GEN_NODES (${GEN_NODES})." >&2
    exit 1
fi
CONTAINER_REPO_LOCATION="${CONTAINER_REPO_LOCATION:-/opt/nemo-rl}"
RECIPE="${RECIPE:-qwen_35/configs/grpo_qwen35_397b_swe_openhands_async.yaml}"
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

# Shell-quote extra Hydra overrides so values with spaces or special
# characters survive the trip through the COMMAND heredoc.
EXTRA_OVERRIDES=""
if (( $# )); then
    printf -v EXTRA_OVERRIDES ' %q' "$@"
fi

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
RAY_CGRAPH_get_timeout=1800 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
HF_TOKEN="\${HF_TOKEN:-}" \
WANDB_API_KEY="\${WANDB_API_KEY:-}" \
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
    ++logger.mlperf.benchmark=qwen35_397b_grpo_swe \
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
    ${EXTRA_OVERRIDES:-}
EOF
)

echo -e "Running command:\n$COMMAND"

# Sync node-local /tmp/ray session logs to the shared log dir every 2 min so
# worker crash tracebacks survive job teardown (consumed by ray.sub).
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-120}"

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
# Source overlay: mount the host checkout's source dirs over the baked
# container checkout so local fixes take effect without an image rebuild.
# 3rdparty/ is intentionally NOT overlaid (the image applies Gym patches at
# build time that the host submodule checkout does not have).
NRL_SOURCE_OVERLAY="${NRL_SOURCE_OVERLAY:-1}"
if [[ "${NRL_SOURCE_OVERLAY}" == "1" ]]; then
    for overlay_dir in nemo_rl examples qwen_35; do
        MOUNTS="${MOUNTS},${REPO_LOCATION}/${overlay_dir}:${CONTAINER_REPO_LOCATION}/${overlay_dir}"
    done
fi
# R2E-Gym eval-harness repair: the image's baked copies of log.py/utils.py are
# syntax-broken (the sandbox-script optional-import repair is non-idempotent
# and re-wrapped the imports at build time), which crashes
# run_local_evaluation on import and zeroes EVERY train/val reward. Mount
# corrected copies whose import lines cannot re-match the baked repair
# patterns. See docker/mlperf/mlperf-gym.patch (PY_R2E_RUNTIME heredoc).
R2E_FIXES_DIR="${R2E_FIXES_DIR:-${REPO_LOCATION}/docker/mlperf/runtime_fixes}"
if [[ -d "${R2E_FIXES_DIR}" ]]; then
    R2E_UTILS_DIR="${CONTAINER_REPO_LOCATION}/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/swe_r2e_gym_setup/R2E-Gym/src/r2egym/agenthub/utils"
    MOUNTS="${MOUNTS},${R2E_FIXES_DIR}/log.py:${R2E_UTILS_DIR}/log.py"
    MOUNTS="${MOUNTS},${R2E_FIXES_DIR}/utils.py:${R2E_UTILS_DIR}/utils.py"
    # Gym HTTP client fix: transient connection errors were retried forever,
    # so a dead vLLM engine turned every rollout touching it into a permanent
    # hang and the training buffer never filled (job 2273194). The patched
    # copy caps the retry window (NEMO_GYM_TRANSIENT_RETRY_SECONDS, 120s
    # default) and then raises so the rollout is back-filled.
    if [[ -f "${R2E_FIXES_DIR}/server_utils.py" ]]; then
        MOUNTS="${MOUNTS},${R2E_FIXES_DIR}/server_utils.py:${CONTAINER_REPO_LOCATION}/3rdparty/Gym-workspace/Gym/nemo_gym/server_utils.py"
    fi
    # TE attention-backend gate fix: TE 2.15's FlashAttention-2 head_dim>192
    # allowlist enumerates sm 8.0/9.0/10.0/12.0 but not GB300's sm10.3, so the
    # Qwen3.5 full-attention layers (head_dim 256) fell back to unfused
    # attention ([B,H,S,S] materialization -> the training-step OOMs).
    # Verified empirically (preflight3b, job 2274314): with (10,3) allowed,
    # flash-attn 2.8.1 runs hdim-256 fwd+bwd on sm103 in thd and sbhd.
    # NOTE: the uv-cache hash in the target path is specific to this
    # container image; with a different image, locate TE via
    #   find /root/.cache/uv -path '*dot_product_attention/utils.py'
    # inside the container and override the mount accordingly.
    if [[ -f "${R2E_FIXES_DIR}/te_dpa_utils.py" ]]; then
        MOUNTS="${MOUNTS},${R2E_FIXES_DIR}/te_dpa_utils.py:/root/.cache/uv/archive-v0/oVtZpwakETGWXnnj/transformer_engine/pytorch/attention/dot_product_attention/utils.py"
    fi
    # fla GDN kernel fix (bug #16): backport of upstream fla PR #1000 onto
    # the pinned fla-core 0.4.2 - Blackwell selects unstable Triton autotune
    # configs for prepare_wy_repr_bwd_kernel (fla #999/#913: silent GDN
    # backward hang on B200-class GPUs; matches the every-second-train-step
    # deadlock signature, jobs 2289258..2295735). Target path verified via
    # fla.__file__ inside this image (job 2300937).
    if [[ -f "${R2E_FIXES_DIR}/fla_wy_fast.py" ]]; then
        MOUNTS="${MOUNTS},${R2E_FIXES_DIR}/fla_wy_fast.py:/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/lib/python3.13/site-packages/fla/ops/gated_delta_rule/wy_fast.py"
    fi
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
    ${SBATCH_EXTRA_ARGS[@]+"${SBATCH_EXTRA_ARGS[@]}"} \
    ${SLURM_EXCLUDE:+--exclude=$SLURM_EXCLUDE} \
    ray.sub
