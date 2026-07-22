# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Qwen3.5-122B-A10B async GRPO proxy: 16 GB300 nodes split evenly between
# Megatron training and vLLM generation.

source "$(dirname "${BASH_SOURCE[0]}")/config_common.sh"

# Keep the benchmark recipe and NeMo-Gym driver in sync with this checkout.
_QWEN35_PYTORCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXTRA_MOUNTS="${_QWEN35_PYTORCH_DIR}:/workspace/llm"
if [[ "${RECIPE:-}" == "qwen_35/configs/grpo_qwen35_397b_swe_openhands_async_gbs256.yaml" ]]; then
    export RECIPE="/workspace/llm/conf/grpo_qwen35_397b_swe_openhands_async_gbs256.yaml"
fi

: "${HF_CKPT_PATH_122B:?source the external site data config before this config}"
export HF_CKPT_PATH="${HF_CKPT_PATH_122B}"

export TRAIN_NODES=8
export GEN_NODES=8
export COLOCATED_GENERATION=0

# Parallelism shape. The YAML has matching defaults, while these exports make
# the launcher shape explicit and easy to override (for example PP=2).
export TP=${TP:-4}
export ETP=${ETP:-1}
export EP=${EP:-32}
export PP=${PP:-1}
export GTP=${GTP:-8}
export GPP=${GPP:-1}
export GEP=${GEP:-8}

# EP32 spans 8 nodes; keep the allocation segment-aligned for NVL72.
export SEGMENT=8
export SBATCH_SEGMENT=${SBATCH_SEGMENT:-${SEGMENT}}

# The launcher may override this for smoke runs; the committed default is the
# full benchmark step count.
export MAX_STEPS=${MAX_STEPS:-30}

export DGXNNODES=$((TRAIN_NODES + GEN_NODES))
export DGXSYSTEM=$(basename "$(readlink -f "${BASH_SOURCE[0]}")" | sed 's/^config_//' | sed 's/\.sh$//')

export WALLTIME_RUNANDTIME=120
export WALLTIME=$((20 + ${NEXP:-1} * (WALLTIME_RUNANDTIME + 10)))
