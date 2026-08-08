#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --account=coreai_chef_posttrain
#SBATCH --partition=batch
#SBATCH --time=00:08:00
#SBATCH --job-name=coreai_chef_posttrain.hybridep-b200-dmabuf-probe
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/pr2964-20step-20260807/debug-super-ep32/logs/dmabuf-probe-%j.out

set -euo pipefail

debug_root=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/pr2964-20step-20260807/debug-super-ep32
container=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/containers/nemo-rl-nightly-cw-fallback-20260808/nemo_rl_nightly_20260805_15171871.sqsh
binary=${debug_root}/dmabuf-probe-${SLURM_JOB_ID}

srun \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=1 \
  --container-image="${container}" \
  --container-mounts=/lustre:/lustre \
  --no-container-mount-home \
  bash -lc "
    set -euo pipefail
    gcc -O2 -I/usr/local/cuda/include \\
      -o '${binary}' '${debug_root}/probe_dmabuf_mr.c' \\
      -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 \\
      -lcuda -libverbs
    '${binary}'
  "
