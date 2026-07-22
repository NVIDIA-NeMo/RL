# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

: "${LOGDIR:=./results}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${EXTRA_MOUNTS:=""}"

# Host input paths, set by an external site data config.
: "${HF_CKPT_PATH:?HF_CKPT_PATH not set (source the external site data config)}"
: "${NRL_MEGATRON_CHECKPOINT_DIR:?NRL_MEGATRON_CHECKPOINT_DIR not set}"
: "${NEMO_GYM_SWE_TRAIN_DATA_PATH:?NEMO_GYM_SWE_TRAIN_DATA_PATH not set}"
: "${NEMO_GYM_SWE_VALIDATION_DATA_PATH:?NEMO_GYM_SWE_VALIDATION_DATA_PATH not set}"
: "${NEMO_GYM_SWE_SIF_DIR:?NEMO_GYM_SWE_SIF_DIR not set}"

# Container-side input locations; run_and_time.sh re-exports the original
# names to these values inside the container.
CONTAINER_INPUT_ROOT=${CONTAINER_INPUT_ROOT:-/inputs/nemo_gym}
export CONTAINER_HF_CKPT_PATH=${CONTAINER_HF_CKPT_PATH:-${HF_CKPT_PATH}}
export CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR=${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR:-${CONTAINER_INPUT_ROOT}/mcore_ckpt}
export CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH=${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH:-${CONTAINER_INPUT_ROOT}/data/train.jsonl}
export CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH=${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH:-${CONTAINER_INPUT_ROOT}/data/validation.jsonl}
export CONTAINER_NEMO_GYM_SWE_SIF_DIR=${CONTAINER_NEMO_GYM_SWE_SIF_DIR:-${CONTAINER_INPUT_ROOT}/sif}

_nemo_log_dir="${LOGDIR}/nemo_logs_${SLURM_JOB_ID:-${DATESTAMP}}"
_checkpoint_dir="${CHECKPOINTS_DIR:-${LOGDIR}/checkpoint}"
_hf_cache_dir="${HF_CACHE_DIR:-${LOGDIR}/hf_cache}"
( umask 0002; mkdir -p "${LOGDIR}" "${_nemo_log_dir}" "${_checkpoint_dir}" "${_hf_cache_dir}" )

_cont_mounts="${LOGDIR}:/results"
_cont_mounts+=",${_nemo_log_dir}:/logs"
_cont_mounts+=",${_checkpoint_dir}:/checkpoint"
_cont_mounts+=",${_hf_cache_dir}:/opt/nemo-rl/.cache"

# Dev-only NeMo-RL patch: mount the patch read-only and apply it inside each
# node's writable container before Ray starts. This avoids both an image
# rebuild and mutating a host source checkout.
if [[ -n "${NRL_RUNTIME_PATCH:-}" && "${NRL_SOURCE_OVERLAY:-0}" == "1" ]]; then
    echo "NRL_RUNTIME_PATCH cannot be combined with NRL_SOURCE_OVERLAY=1" >&2
    exit 1
fi
if [[ -n "${NRL_RUNTIME_PATCH:-}" ]]; then
    if [[ ! -f "${NRL_RUNTIME_PATCH}" ]]; then
        echo "NRL_RUNTIME_PATCH does not name a file: ${NRL_RUNTIME_PATCH}" >&2
        exit 1
    fi
    _runtime_patch_host="$(realpath "${NRL_RUNTIME_PATCH}")"
    export NRL_RUNTIME_PATCH_CONTAINER=/tmp/nemo-rl-runtime.patch
    _cont_mounts+=",${_runtime_patch_host}:${NRL_RUNTIME_PATCH_CONTAINER}:ro"
fi

# Identity mount: run.sub polls STARTED_RAY_HEAD on the host while the head
# container touches the same path inside.
if [[ -n "${RAY_CLUSTER_LOG_DIR:-}" ]]; then
    _cont_mounts+=",${RAY_CLUSTER_LOG_DIR}:${RAY_CLUSTER_LOG_DIR}"
fi

# HF checkpoint: identity mount. For HF-hub cache layouts mount the model
# cache root two levels up so snapshot symlinks into blobs/ resolve.
if [[ "${HF_CKPT_PATH}" == */snapshots/* ]]; then
    _hf_model_cache_dir="$(dirname "$(dirname "${HF_CKPT_PATH}")")"
    _cont_mounts+=",${_hf_model_cache_dir}:${_hf_model_cache_dir}"
else
    _cont_mounts+=",${HF_CKPT_PATH}:${HF_CKPT_PATH}"
fi

# Writable: the first run converts HF -> mcore into this cache
_cont_mounts+=",${NRL_MEGATRON_CHECKPOINT_DIR}:${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR}"

# jsonl + SIF dir: canonical container paths + host-path compat mounts (for
# recipes carrying absolute host-side data_path/container_formatter entries)
_cont_mounts+=",${NEMO_GYM_SWE_TRAIN_DATA_PATH}:${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH}:ro"
_cont_mounts+=",${NEMO_GYM_SWE_VALIDATION_DATA_PATH}:${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH}:ro"
_train_data_dir="$(dirname "${NEMO_GYM_SWE_TRAIN_DATA_PATH}")"
_val_data_dir="$(dirname "${NEMO_GYM_SWE_VALIDATION_DATA_PATH}")"
_cont_mounts+=",${_train_data_dir}:${_train_data_dir}"
if [[ "${_val_data_dir}" != "${_train_data_dir}" ]]; then
    _cont_mounts+=",${_val_data_dir}:${_val_data_dir}"
fi
_cont_mounts+=",${NEMO_GYM_SWE_SIF_DIR}:${CONTAINER_NEMO_GYM_SWE_SIF_DIR}:ro"
_cont_mounts+=",${NEMO_GYM_SWE_SIF_DIR}:${NEMO_GYM_SWE_SIF_DIR}:ro"

# apptainer needs /dev/fuse for the SIF sandboxes
_cont_mounts+=",/dev/fuse:/dev/fuse"

# Dev-only source overlay (CI runs baked sources): mount a host nemo-rl
# checkout's source dirs over the baked ones. 3rdparty/ stays baked (Gym
# patches are applied at image build).
if [[ "${NRL_SOURCE_OVERLAY:-0}" == "1" ]]; then
    : "${REPO_LOCATION:?NRL_SOURCE_OVERLAY=1 requires REPO_LOCATION (host nemo-rl checkout)}"
    for overlay_dir in nemo_rl examples qwen_35; do
        _cont_mounts+=",${REPO_LOCATION}/${overlay_dir}:/opt/nemo-rl/${overlay_dir}"
    done
fi

if [[ -n "${EXTRA_MOUNTS}" ]]; then
    _cont_mounts+=",${EXTRA_MOUNTS}"
fi

if [[ "${JET:-0}" -eq 1 ]]; then
    _cont_mounts="${_cont_mounts},${JET_DIR}:/root/.jet"
fi

mounts_to_verify="HF_CKPT:${CONTAINER_HF_CKPT_PATH} SIF_DIR:${CONTAINER_NEMO_GYM_SWE_SIF_DIR} TRAIN_DATA:${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH} VAL_DATA:${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH}"
