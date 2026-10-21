#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
flint_base=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/Automodel
flint_audit="$flint_base/flint-refit-step0-4layer-20261020"
flint_original="$flint_base/flint-grpo-4layer-20261019"
flint_image="$flint_base/flint-nemorl-20261019/images/flint-nemorl-terry-20261019-arm64-native-01.sqsh"
exec srun --partition=batch --qos=interactive --account=coreai_dlalgo_nemorl \
  --job-name=flint-step0-refit --nodes=1 --ntasks=1 --gpus-per-node=4 \
  --cpus-per-task=32 --mem=256G --time=02:00:00 --export=ALL \
  --container-image="$flint_image" \
  --container-mounts="$flint_original:/flint-work:ro,$flint_audit:/flint-audit" \
  --container-workdir=/flint-work/RL --no-container-mount-home \
  --no-container-entrypoint --container-writable --pty \
  /bin/bash /flint-audit/tools/run.sh
