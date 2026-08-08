#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --account=coreai_chef_posttrain
#SBATCH --partition=batch
#SBATCH --time=00:20:00
#SBATCH --job-name=coreai_chef_posttrain.hybridep-ep32-buffer-probe
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/pr2964-20step-20260807/debug-super-ep32/logs/probe-%j.out

set -euo pipefail

debug_root=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/pr2964-20step-20260807/debug-super-ep32
container=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/containers/nemo-rl-nightly-cw-fallback-20260808/nemo_rl_nightly_20260805_15171871.sqsh
wheel=${DEEPEP_WHEEL:?DEEPEP_WHEEL must identify the wheel under test}
wheel_sha256=${DEEPEP_WHEEL_SHA256:?DEEPEP_WHEEL_SHA256 must be set}
overlay=/tmp/hybridep-buffer-probe-${SLURM_JOB_ID}

test "$(sha256sum "${wheel}" | cut -d' ' -f1)" = "${wheel_sha256}"
export wheel overlay

srun \
  --nodes=4 \
  --ntasks-per-node=1 \
  --container-image="${container}" \
  --container-mounts=/lustre:/lustre \
  --no-container-mount-home \
  bash -lc '
    set -euo pipefail
    rm -rf -- "${overlay}"
    mkdir -p "${overlay}"
    unset UV_CONFIG_FILE
    UV_NO_CONFIG=1 uv pip install \
      --python /opt/nemo_rl_venv/bin/python \
      --target "${overlay}" \
      --reinstall --no-deps --no-index "${wheel}"
  '

master_addr=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
export MASTER_ADDR=${master_addr}
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export PYTHONPATH=${overlay}
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=${NUM_OF_TOKENS_PER_CHUNK_COMBINE_API:-128}
export NVLINK_DOMAIN_SIZE=8
export USE_MNNVL=0
export NCCL_NVLS_ENABLE=0
export PROBE_MAX_TOKENS=${PROBE_MAX_TOKENS:-2048}

srun \
  --nodes=4 \
  --ntasks-per-node=8 \
  --gpus-per-node=8 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  --kill-on-bad-exit=1 \
  --container-image="${container}" \
  --container-mounts=/lustre:/lustre \
  --no-container-mount-home \
  /opt/nemo_rl_venv/bin/python "${debug_root}/probe_hybridep_buffer.py"
