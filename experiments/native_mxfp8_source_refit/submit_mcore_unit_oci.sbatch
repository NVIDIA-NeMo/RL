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
: "${SRUN_BIN:?Set SRUN_BIN to the resolved srun executable}"

MCORE_DIR=${REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
LOG_DIR=${RESULT_ROOT}/mcore-fp64-router-padding-unit
mkdir -p "${LOG_DIR}"
test "$(git -C "${MCORE_DIR}" rev-parse HEAD)" = "${MCORE_FIX_SHA}"

"${SRUN_BIN}" --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-remap-root \
  --no-container-mount-home \
  --container-mounts="/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch" \
  bash -lc "
    set -euo pipefail
    cd ${MCORE_DIR}
    UNIT_DEPS=/raid/scratch/${USER}/mcore-unit-deps/${MCORE_FIX_SHA}
    mkdir -p \"\${UNIT_DEPS}\"
    uv pip install --target \"\${UNIT_DEPS}\" typing_extensions pytest
    export PYTHONPATH=\${UNIT_DEPS}:${MCORE_DIR}
    /opt/nemo_rl_venv/bin/python - <<'PY'
import torch

from megatron.core.extensions.transformer_engine import Fp8Padding

padding = Fp8Padding(2, align_size=4)
row_indices = torch.arange(1, 4, dtype=torch.int64, device='cuda').unsqueeze(-1)
padded, counts = padding(row_indices, [2, 1])
expected = torch.tensor([[1], [2], [0], [0], [3], [0], [0], [0]], device='cuda')
torch.testing.assert_close(padded, expected)
assert counts == [4, 4]
print('TE INT64 padding probe passed')
PY
    /opt/nemo_rl_venv/bin/python -m pytest -q \
      tests/unit_tests/transformer/moe/test_grouped_mlp.py \
      -k 'router_probability_padding'
  " 2>&1 | tee "${LOG_DIR}/${SLURM_JOB_ID}.log"
