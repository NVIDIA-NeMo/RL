#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit only when no other Nano build/training job is queued or running.
# Site account, partition, output and time limit are supplied to sbatch.
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
set -euo pipefail
for key in NANO35_IMAGE NANO35_MODEL NANO35_DATA NANO35_SIF_TEMPLATE NANO35_VALIDATION_ROOT; do
    [[ -n "${!key:-}" ]] || { echo "Set $key before submission" >&2; exit 2; }
    export "$key"
done
# Slurm runs a spooled script; resolve the submitted side-branch checkout.
NANO35_CODE_ROOT=${NANO35_CODE_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the NeMo-RL checkout}}
NANO35_PREFLIGHT_SOURCE="$NANO35_CODE_ROOT/tools/nano35/gpu_preflight.py"
NANO35_RECIPE=examples/configs/recipes/llm/grpo-nano3.5-swe-32n4g-tp4cp16-async-trtllm.v1.yaml
[[ -f "$NANO35_PREFLIGHT_SOURCE" && -f "$NANO35_CODE_ROOT/$NANO35_RECIPE" ]]
[[ "$NANO35_CODE_ROOT" != *,* && "$NANO35_CODE_ROOT" != *$'\n'* ]]
export NANO35_IMAGE_SHA256=$(sha256sum "$NANO35_IMAGE" | awk '{print $1}')
export NANO35_VALIDATION_ROOT="$NANO35_VALIDATION_ROOT/$SLURM_JOB_ID"
mkdir -p "$NANO35_VALIDATION_ROOT"
printf '%s\n' "$NANO35_IMAGE_SHA256" > "$NANO35_VALIDATION_ROOT/image.sha256"
sha256sum "$NANO35_PREFLIGHT_SOURCE" "$NANO35_CODE_ROOT/$NANO35_RECIPE" > "$NANO35_VALIDATION_ROOT/controller-and-recipe.sha256"
# Lyris allocates whole GPU nodes without configuring Slurm GRES. Match
# ray.sub's site detection instead of unconditionally requesting GPU TRES.
GPU_ARGS=()
if sinfo -p "$SLURM_JOB_PARTITION" -h -o '%G' | grep -q 'gpu:'; then
    GPU_ARGS+=(--gres=gpu:4)
fi
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
[[ -c /dev/fuse ]] || { echo 'SWE SIF execution requires host /dev/fuse' >&2; exit 1; }
srun --nodes=1 --ntasks=1 --cpus-per-task=128 --exclusive --cpu-bind=none "${GPU_ARGS[@]}" \
    --container-image="$NANO35_IMAGE" --container-mounts="/lustre:/lustre,/dev/fuse:/dev/fuse,$NANO35_PREFLIGHT_SOURCE:/opt/nemo-rl/tools/nano35/gpu_preflight.py:ro,$NANO35_CODE_ROOT/$NANO35_RECIPE:/opt/nemo-rl/$NANO35_RECIPE:ro" \
    --no-container-mount-home --container-workdir=/opt/nemo-rl \
    bash /opt/nemo-rl/tools/nano35/preflight_driver.sh

# Optional maintainer step in the same allocation; it only writes a local
# archive and runs after the exact runtime image has passed GPU/SWE checks.
if [[ -n ${NANO35_OCI_OUTPUT:-} ]]; then
    bash "${NANO35_OCI_EXPORTER:?Set the frozen local exporter path}" \
        "$NANO35_IMAGE" "$NANO35_OCI_OUTPUT" \
        "${NANO35_OCI_TAG:?Set a local OCI tag}"
fi
