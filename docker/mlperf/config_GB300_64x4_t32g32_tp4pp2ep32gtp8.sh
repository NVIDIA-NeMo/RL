# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Reference config: 64 GB300 nodes = 32 training + 32 generation.
# Name encodes policy parallelism (tp/pp/ep) and vLLM generation TP (gtp),
# both configured in the recipe yaml.

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# Keep the benchmark recipe and NeMo-Gym driver in sync with this checkout.
_QWEN35_PYTORCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXTRA_MOUNTS="${_QWEN35_PYTORCH_DIR}:/workspace/llm"
if [[ "${RECIPE:-}" == "qwen_35/configs/grpo_qwen35_397b_swe_openhands_async_gbs256.yaml" ]]; then
    export RECIPE="/workspace/llm/conf/grpo_qwen35_397b_swe_openhands_async_gbs256.yaml"
fi

export TRAIN_NODES=32
export GEN_NODES=32
export COLOCATED_GENERATION=0

# The 64-node reference shape remains TP4/PP2/EP32. These variables are read
# by the async recipe through oc.env, so the launcher filename and runtime
# parallelism stay aligned while still allowing an explicit override.
export TP=${TP:-4}
export ETP=${ETP:-1}
export EP=${EP:-32}
export PP=${PP:-2}
export GTP=${GTP:-8}
export GPP=${GPP:-1}
export GEP=${GEP:-8}

# EP32 spans 8 nodes; segment-aligned placement keeps each EP alltoall group
# inside one NVL72 rack — required with overlap_grad_reduce on (bug #16).
# The mlperf_utils launcher turns SEGMENT into sbatch --segment; SBATCH_SEGMENT
# covers direct `sbatch run.sub`; run_and_time.sh pairs ++cluster.segment_size.
export SEGMENT=8
export SBATCH_SEGMENT=${SBATCH_SEGMENT:-${SEGMENT}}

# The launcher may override this for smoke runs; the committed default is the
# full benchmark step count.
export MAX_STEPS=${MAX_STEPS:-30}

export DGXNNODES=$((TRAIN_NODES + GEN_NODES))
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//')

export WALLTIME_RUNANDTIME=120
export WALLTIME=$((20 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 10)))
