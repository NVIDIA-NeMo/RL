#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --segment=1
#SBATCH --time=00:20:00
#SBATCH --job-name=mcore-fp64-router-padding

set -euo pipefail

: "${REPO:?Set REPO to the NeMo-RL checkout under /home}"
: "${CONTAINER:?Set CONTAINER to an immutable NeMo-RL image under /lustre}"
: "${RESULT_ROOT:?Set RESULT_ROOT under /lustre}"
: "${MCORE_FIX_SHA:?Set MCORE_FIX_SHA}"

MCORE_DIR=${REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
LOG_DIR=${RESULT_ROOT}/mcore-fp64-router-padding-unit
mkdir -p "${LOG_DIR}"
test "$(git -C "${MCORE_DIR}" rev-parse HEAD)" = "${MCORE_FIX_SHA}"

srun --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts="/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch" \
  bash -lc "
    set -euo pipefail
    cd ${MCORE_DIR}
    export PYTHONPATH=${MCORE_DIR}
    /opt/nemo_rl_venv/bin/python -m pytest -q \
      tests/unit_tests/transformer/moe/test_grouped_mlp.py \
      -k 'router_probability_padding'
  " 2>&1 | tee "${LOG_DIR}/${SLURM_JOB_ID}.log"
