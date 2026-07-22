#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Pre-populate the mcore checkpoint cache: nemo-rl converts HF -> mcore on the
# first run against an empty NRL_MEGATRON_CHECKPOINT_DIR; this runs the
# reference config for one step so nightly runs start from a warm cache.
# Usage (from pytorch/, on the cluster):
#   export LOGDIR=<scratch> CONT=<image>; source <external-site-config.sh>
#   bash data_scripts/convert_ckpt.sh
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
cd "${SCRIPT_DIR}/.."

: "${NRL_MEGATRON_CHECKPOINT_DIR:?NRL_MEGATRON_CHECKPOINT_DIR not set}"
mkdir -p "${NRL_MEGATRON_CHECKPOINT_DIR}"

source config_GB300_64x4_t32g32_tp4pp2ep32gtp8.sh
export MAX_STEPS=1
export NEXP=1

sbatch -N "${DGXNNODES}" --time="${WALLTIME}" run.sub
echo "Submitted 1-step run; the mcore cache will be populated under ${NRL_MEGATRON_CHECKPOINT_DIR}."
