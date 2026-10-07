#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --time=00:20:00
#SBATCH --mem=0
set -euo pipefail
: "${CONTAINER:?Set immutable nightly image}"
: "${SOURCE_ARCHIVE:?Set immutable source archive}"
export LOCAL_ROOT="/raid/scratch/${USER}/nr-option-a-smoke-${SLURM_JOB_ID}"
export PYTHONPYCACHEPREFIX="${LOCAL_ROOT}/pycache"
export NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=1
srun --gres=gpu:1 --no-container-mount-home \
  --container-image="$CONTAINER" --container-mounts=/lustre:/lustre,/raid/scratch:/raid/scratch \
  bash -c '
    set -euo pipefail
    mkdir -p "$LOCAL_ROOT/source"
    tar -xf "$SOURCE_ARCHIVE" -C "$LOCAL_ROOT/source"
    cd "$LOCAL_ROOT/source"
    /opt/nemo_rl_venv/bin/python tools/config_cli.py expand experiments/lightning_pr4353_20261007/async-mxfp8-train-option-a.yaml >/dev/null
    /usr/local/bin/python-MegatronPolicyWorker experiments/lightning_pr4353_20261007/option_a_smoke.py
  '
