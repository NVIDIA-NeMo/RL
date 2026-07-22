# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Qwen3.5-397B-A17B synchronous GRPO proxy: 16 colocated GB300 nodes.

source "$(dirname "${BASH_SOURCE[0]}")/config_common.sh"

# Keep the smoke recipe and driver in sync with this checkout. A single
# directory mount avoids Slurm's comma-separated environment export handling
# that can silently drop a multi-file EXTRA_MOUNTS override.
_QWEN35_PYTORCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXTRA_MOUNTS="${_QWEN35_PYTORCH_DIR}:/workspace/llm"

: "${HF_CKPT_PATH_397B:?source the external site data config before this config}"
export HF_CKPT_PATH="${HF_CKPT_PATH_397B}"

# config_common.sh provides the async benchmark recipe by default. Select the
# synchronous wrapper unless the caller supplied a different recipe.
if [[ "${RECIPE:-}" == "qwen_35/configs/grpo_qwen35_397b_swe_openhands_async_gbs256.yaml" ]]; then
    export RECIPE="/workspace/llm/conf/grpo_qwen35_397b_swe_openhands_sync.yaml"
fi

# Parallelism shape. The sync 16-node shape remains TP4/PP2/EP32.
export TP=${TP:-4}
export ETP=${ETP:-1}
export EP=${EP:-32}
export PP=${PP:-2}
export GTP=${GTP:-8}
export GPP=${GPP:-1}
export GEP=${GEP:-8}

# In colocated mode these are logical training/generation resource counts;
# both refer to the same physical 16-node Ray allocation.
export TRAIN_NODES=16
export GEN_NODES=16
export COLOCATED_GENERATION=1
export COLOCATED_GENERATION_NODES=16

# EP32 spans 8 nodes; keep the allocation segment-aligned for NVL72.
export SEGMENT=8
export SBATCH_SEGMENT=${SBATCH_SEGMENT:-${SEGMENT}}

export MAX_STEPS=${MAX_STEPS:-30}

# Colocated generation does not add a second physical pool.
export DGXNNODES=${TRAIN_NODES}
export DGXSYSTEM=$(basename "$(readlink -f "${BASH_SOURCE[0]}")" | sed 's/^config_//' | sed 's/\.sh$//')

export WALLTIME_RUNANDTIME=120
export WALLTIME=$((20 + ${NEXP:-1} * (WALLTIME_RUNANDTIME + 10)))
