#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
set -euo pipefail
for key in NANO35_IMAGE NANO35_MODEL NANO35_DATA NANO35_SIF_TEMPLATE NANO35_VALIDATION_ROOT; do
    [[ -n "${!key:-}" ]] || { echo "Set $key first" >&2; exit 2; }
    export "$key"
done
# Slurm executes a spooled copy, so BASH_SOURCE does not identify the checkout.
NANO35_CODE_ROOT=${NANO35_CODE_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the NeMo-RL checkout}}
NANO35_RECIPE=examples/configs/recipes/llm/grpo-nano3.5-swe-32n4g-tp4cp16-async-trtllm.v1.yaml
[[ -f "$NANO35_CODE_ROOT/$NANO35_RECIPE" && -f "$NANO35_CODE_ROOT/tools/nano35/check_training_data.py" ]]
export NANO35_IMAGE_SHA256=$(sha256sum "$NANO35_IMAGE" | awk '{print $1}')
export NANO35_DATA_PREFLIGHT_OUTPUT="$NANO35_VALIDATION_ROOT/$SLURM_JOB_ID/data-preflight.json"
mkdir -p "$(dirname "$NANO35_DATA_PREFLIGHT_OUTPUT")"
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=none \
    --container-image="$NANO35_IMAGE" \
    --container-mounts="/lustre:/lustre,$NANO35_CODE_ROOT:/nrl-data-check:ro,$NANO35_CODE_ROOT/$NANO35_RECIPE:/opt/nemo-rl/$NANO35_RECIPE:ro" \
    --no-container-mount-home --container-workdir=/opt/nemo-rl \
    bash /nrl-data-check/tools/nano35/data_preflight_driver.sh
