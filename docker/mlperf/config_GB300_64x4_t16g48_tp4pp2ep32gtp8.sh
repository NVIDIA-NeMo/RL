# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Authoritative GB300 system profile: 16 policy nodes and 48 generation nodes.
# The selected _gbs*.yaml child owns the batch-dependent hyperparameters.
# Runtime behavior is baked into the image; this profile does not mount source
# or configuration from the host.

_QWEN35_MLPERF_TARGET_ACCURACY=${MLPERF_TARGET_ACCURACY:-0.7}
source "$(dirname "${BASH_SOURCE[0]}")/config_common.sh"

# Production uses the image's pinned NeMo-RL tree and baked benchmark config.
unset NRL_RUNTIME_PATCH NRL_RUNTIME_PATCH_CONTAINER REPO_LOCATION EXTRA_MOUNTS
export NRL_SOURCE_OVERLAY=0

# Select the qualified curriculum-v2 training set. The artifact root is
# derived from the external site's SIF path so this profile does
# not change the reusable data defaults used by other Qwen configurations.
: "${NEMO_GYM_SWE_SIF_DIR:?source the external site data config before this profile}"
_QWEN35_CURRICULUM_FILENAME=benchmark_r2e_gym_easy_train.filtered.curriculum-v2-classic-cycles2-seed20260710.jsonl
_QWEN35_R2E_GYM_ROOT=${NEMO_GYM_SWE_SIF_DIR%/easy-sif}
_QWEN35_CURRICULUM_DEFAULT=${_QWEN35_R2E_GYM_ROOT}/easy-curriculum-subset/${_QWEN35_CURRICULUM_FILENAME}
export NEMO_GYM_SWE_TRAIN_DATA_PATH="${QWEN35_CURRICULUM_DATA_PATH:-${_QWEN35_CURRICULUM_DEFAULT}}"
unset _QWEN35_CURRICULUM_FILENAME _QWEN35_R2E_GYM_ROOT _QWEN35_CURRICULUM_DEFAULT

export RECIPE=${RECIPE:-qwen_35/configs/grpo_qwen35_397b_swe_openhands_async_gbs256.yaml}
export TRAIN_NODES=16
export GEN_NODES=48
export COLOCATED_GENERATION=0

export MLPERF_TARGET_ACCURACY=${_QWEN35_MLPERF_TARGET_ACCURACY}
unset _QWEN35_MLPERF_TARGET_ACCURACY

# EP32 spans eight four-GPU nodes; keep Slurm placement and NeMo-RL's logical
# topology aligned to the same NVLink domain.
export SEGMENT=8
export SBATCH_SEGMENT=${SBATCH_SEGMENT:-${SEGMENT}}

export RAY_CGRAPH_get_timeout=${RAY_CGRAPH_get_timeout:-1810}
export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-120}
export RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-10}

export DGXNNODES=$((TRAIN_NODES + GEN_NODES))
export DGXSYSTEM="$(basename "$(readlink -f "${BASH_SOURCE[0]}")" | sed 's/^config_//' | sed 's/\.sh$//')"

export WALLTIME_RUNANDTIME=${WALLTIME_RUNANDTIME:-240}
export WALLTIME=${WALLTIME:-240}
