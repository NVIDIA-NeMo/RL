#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${PROJECT_ROOT:?}" "${RUN_DIR:?}" "${OVERLAY_DIR:?}" "${GAMES_VENV_ROOT:?}"
: "${GAMES_TRAIN_MANIFEST:?}" "${GAMES_EVAL_MANIFEST:?}" "${IMAGE_TRAIN_MANIFEST:?}" "${VISGYM_ASSET_ARCHIVE:?}"
HSG_ROOT=/lustre/fs1/portfolios/nemotron/projects/nemotron_omni_vision/users/aroshanghias
case "$PROJECT_ROOT" in "$HSG_ROOT"/*/snapshots/*) ;; *) exit 2;; esac
case "$RUN_DIR" in "$HSG_ROOT"/*/results/*) ;; *) exit 2;; esac
case "$GAMES_VENV_ROOT" in "$HSG_ROOT"/*/cache/gym-venvs/*) ;; *) exit 2;; esac
case "${DRY_RUN:-1}" in 0|1) ;; *) exit 2;; esac
test ! -e "$RUN_DIR"
for path in "$PROJECT_ROOT/tools/check_visual_image_tools_components.py" "$OVERLAY_DIR/READY.json" \
  "$GAMES_VENV_ROOT/.visual-games-suite-prefetch-complete" "$GAMES_TRAIN_MANIFEST" \
  "$GAMES_EVAL_MANIFEST" "$IMAGE_TRAIN_MANIFEST" "$VISGYM_ASSET_ARCHIVE"; do test -r "$path"; done
export PROJECT_ROOT RUN_DIR OVERLAY_DIR GAMES_VENV_ROOT GAMES_TRAIN_MANIFEST GAMES_EVAL_MANIFEST IMAGE_TRAIN_MANIFEST VISGYM_ASSET_ARCHIVE
submit=(sbatch --parsable --account=nemotron_omni_vision --partition=batch --qos=normal
  --nodes=1 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=16 --exclusive
  --time=01:00:00 --job-name=image-tools-games-components --chdir="$PROJECT_ROOT"
  --output="$RUN_DIR/slurm-%j.out" --error="$RUN_DIR/slurm-%j.err"
  "$PROJECT_ROOT/tools/visual_image_tools_components_hsg.sh")
if [[ "${DRY_RUN:-1}" == 1 ]]; then printf '%q ' "${submit[@]}"; printf '\n'; exit 0; fi
mkdir "$RUN_DIR"
job_id=$("${submit[@]}")
printf '%s\n' "$job_id" > "$RUN_DIR/gpu-job-id.txt"
printf 'Submitted component qualification %s; results: %s\n' "$job_id" "$RUN_DIR"
