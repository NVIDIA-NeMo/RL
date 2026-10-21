#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=flint-full49-resume
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=02:00:00
#SBATCH --export=ALL
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/Automodel/flint-grpo-49layer-20261019/evidence/resume-allocation-%j.log

set -euo pipefail
set +x
umask 077
flint_task_root=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/Automodel/flint-grpo-49layer-20261019
flint_model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/Automodel/cedargrove-model_vv1
flint_image=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/Automodel/flint-nemorl-20261019/images/flint-nemorl-terry-20261019-arm64-native-01.sqsh
flint_secrets=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/nemo-rl-internal/secrets.sh
test -f "$flint_task_root/tools/check_refit_checkpoint.py"
test -f "$flint_model/config.json"
test -f "$flint_image"
test ! -e "$flint_task_root/RL/tools/generate_fingerprint.py"
source "$flint_secrets" >/dev/null 2>&1
set +x

exec srun --ntasks=1 \
  --container-image="$flint_image" \
  --container-mounts="$flint_task_root:/flint-work,$flint_model:/flint-model:ro" \
  --container-workdir=/flint-work/RL \
  --no-container-mount-home --no-container-entrypoint --container-writable \
  /bin/bash /flint-work/tools/run_resume_validation.sh
