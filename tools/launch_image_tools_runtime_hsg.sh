#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Independent one-node import/kernel qualification; does not start training.
# Use the same check as a preflight in the eventual training allocation.
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the immutable HSG snapshot}"
: "${RUN_DIR:?Set RUN_DIR to a new HSG results directory}"
: "${OVERLAY_DIR:?Set OVERLAY_DIR to a new or completely built isolated overlay}"
: "${MODEL_CHECKPOINT:?Set MODEL_CHECKPOINT to the user-selected step_120/hf}"
HSG_ROOT=/lustre/fs1/portfolios/nemotron/projects/nemotron_omni_vision/users/aroshanghias
CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yifuw/images/striped/rl-gym.67009223-gym_ln_fix.sqsh
DRY_RUN=${DRY_RUN:-1}
IMAGE_TOOLS_RUNTIME_CHECK=${IMAGE_TOOLS_RUNTIME_CHECK:-full}
case "$IMAGE_TOOLS_RUNTIME_CHECK" in full|callbacks) ;; *) exit 2;; esac
case "$DRY_RUN" in 0|1) ;; *) echo 'DRY_RUN must be 0 or 1' >&2; exit 2;; esac
case "$PROJECT_ROOT" in "$HSG_ROOT"/*/snapshots/*) ;; *) echo 'Expected a user-owned HSG snapshot' >&2; exit 2;; esac
case "$RUN_DIR" in "$HSG_ROOT"/*/results/*) ;; *) echo 'Expected a user-owned HSG results directory' >&2; exit 2;; esac
case "$OVERLAY_DIR" in "$HSG_ROOT"/*/cache/*) ;; *) echo 'Expected a user-owned HSG cache directory' >&2; exit 2;; esac
if test -e "$OVERLAY_DIR" && ! test -f "$OVERLAY_DIR/READY.json"; then
    echo 'Overlay exists but is incomplete; inspect it before retrying' >&2; exit 2
fi
test -f "$PROJECT_ROOT/tools/check_image_tools_runtime.py"
test -r "$CONTAINER"
test -r "$MODEL_CHECKPOINT/config.json"
test -r "$MODEL_CHECKPOINT/model.safetensors.index.json"
if test -e "$RUN_DIR"; then
    echo "Refusing to reuse existing RUN_DIR: $RUN_DIR" >&2
    exit 2
fi

overlay_parent=$(dirname "$OVERLAY_DIR")
cached_model_mount=
if [[ -n "${IMAGE_TOOLS_CACHED_MODEL_CONFIG:-}" ]]; then
    case "$IMAGE_TOOLS_CACHED_MODEL_CONFIG" in "$HSG_ROOT"/*/cache/megatron-checkpoints/*/iter_0000000/run_config.yaml) ;; *) echo 'Expected a user-owned converted model config' >&2; exit 2;; esac
    test -r "$IMAGE_TOOLS_CACHED_MODEL_CONFIG"
    cached_model_dir=$(dirname "$IMAGE_TOOLS_CACHED_MODEL_CONFIG")
    cached_model_mount=",$cached_model_dir:$cached_model_dir:ro"
fi
MCORE_DATASETS_REL=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets
inside='set -euo pipefail
if ! test -f "$OVERLAY_DIR/READY.json"; then
  /opt/nemo_rl_venv/bin/python /opt/nemo-rl/tools/build_image_tools_overlay.py --lock /opt/nemo-rl/uv.lock --output "$OVERLAY_DIR"
fi
export IMAGE_TOOLS_BUILD_PYTHON=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
export PATH="/opt/nemo-rl/tools/image_tools_build_bin:$PATH"
export NEMO_RL_SOURCE_OVERRIDE_ROOT=/opt/nemo-rl
export NEMO_GYM_EXTRA_ROOTS=/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym
export PYTHONPATH="$OVERLAY_DIR/packages:/opt/nemo-rl/tools/runtime_source_override:/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
export NRL_IGNORE_VERSION_MISMATCH=1
bash /opt/nemo-rl/tools/image_tools_dataset_helpers.sh build "$IMAGE_TOOLS_DATASET_DIR"
if [[ "$IMAGE_TOOLS_RUNTIME_CHECK" == callbacks ]]; then
  exec bash /opt/nemo-rl/tools/run_callback_regressions.sh
fi
exec /opt/nemo_rl_venv/bin/python -m tools.check_image_tools_runtime --gpu --overlay "$OVERLAY_DIR"'
step=(srun --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=12
    --container-image="$CONTAINER" --no-container-mount-home
    --container-mounts="$PROJECT_ROOT:/opt/nemo-rl:ro,$RUN_DIR:$RUN_DIR,$overlay_parent:$overlay_parent,$MODEL_CHECKPOINT:$MODEL_CHECKPOINT:ro,$RUN_DIR/runtime/mcore-datasets:/opt/nemo-rl/$MCORE_DATASETS_REL$cached_model_mount"
    --container-workdir=/opt/nemo-rl
    env PROJECT_ROOT=/opt/nemo-rl PYTHONDONTWRITEBYTECODE=1 OVERLAY_DIR="$OVERLAY_DIR" MODEL_CHECKPOINT="$MODEL_CHECKPOINT"
    IMAGE_TOOLS_DATASET_DIR="/opt/nemo-rl/$MCORE_DATASETS_REL"
    IMAGE_TOOLS_CACHED_MODEL_CONFIG="${IMAGE_TOOLS_CACHED_MODEL_CONFIG:-}"
    IMAGE_TOOLS_RUNTIME_CHECK="$IMAGE_TOOLS_RUNTIME_CHECK"
    HF_HOME="$RUN_DIR/cache/huggingface" TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
    bash -c "$inside")
printf -v command '%q ' "${step[@]}"
# sbatch --wrap runs /bin/sh (dash on HSG), while %q can emit Bash $'...' syntax.
# POSIX-quote the entire command as one argument to an explicit Bash interpreter.
wrapped_command=$(python3 -c 'import shlex, sys; print("exec /bin/bash -c " + shlex.quote(sys.argv[1]))' "$command")
submit=(sbatch --parsable --account=nemotron_omni_vision --partition=batch --qos=normal
    --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=12 --exclusive
    --time=00:45:00 --job-name=image-tools-overlay --chdir="$PROJECT_ROOT"
    --output="$RUN_DIR/slurm-%j.out" --error="$RUN_DIR/slurm-%j.err" --wrap="$wrapped_command")
if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY_RUN: '; printf '%q ' "${submit[@]}"; printf '\n'
    exit 0
fi
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
mkdir "$RUN_DIR/runtime"
bash "$PROJECT_ROOT/tools/image_tools_dataset_helpers.sh" stage "$PROJECT_ROOT/$MCORE_DATASETS_REL" "$RUN_DIR/runtime/mcore-datasets"
mkdir -p "$overlay_parent"
job_id=$("${submit[@]}")
printf '%s\n' "$job_id" > "$RUN_DIR/gpu-job-id.txt"
printf 'Submitted runtime qualification job %s; logs: %s\n' "$job_id" "$RUN_DIR"
