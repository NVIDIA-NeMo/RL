#!/usr/bin/env bash
set -euo pipefail

# Convenience launcher for the length-adjusted Ultra pipeclean recipe.
#
# This intentionally follows examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh:
# callers provide cluster, container, model, data, cache, and optional judge
# model paths through environment variables. This wrapper only selects the
# length-adjusted config by default.
#
# Required by ultra_launch.sh:
#   EXP_NAME
#   MODEL_PATH
#   TRAIN_PATH
#   VAL_PATH
#   CONTAINER
#   SANDBOX_CONTAINER
#   PERSISTENT_CACHE
#   SLURM_PARTITION
#   SLURM_ACCOUNT
#
# Optional:
#   CONFIG_PATH        Override the recipe config.
#   WANDB_PROJ         W&B project name.
#   GENRM_MODEL        GenRM model path or HF id.
#   NL2BASH_JUDGE_MODEL
#   SAFETY_JUDGE_MODEL
#   EXTRA_MOUNTS       Comma-separated host:container mount pairs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_PATH="${CONFIG_PATH:-examples/configs/grpo_ultra_64n4g_length_adj_pipeclean.yaml}"
export WANDB_PROJ="${WANDB_PROJ:-nemotron-3-ultra-length-adjusted}"

exec "${SCRIPT_DIR}/ultra_launch.sh" "$@"
