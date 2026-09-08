#!/usr/bin/env bash

set -euo pipefail

export ACTION=${ACTION:-submit}
export MODEL=${MODEL:-qwen235}
export PRECISION_MODE=mxfp8
export FP8_PARAM=true
export MAX_STEPS=${MAX_STEPS:-20}
export RUN_GROUP=${RUN_GROUP:-20260908-fp64-router-fix-lyris}
export REPO=${REPO:-/home/${USER}/RL-qwen235-fp64-router-padding-20260908}
export CONTAINER=${CONTAINER:-$(readlink -f /lustre/fsw/coreai_dlalgo_llm/users/${USER}/containers/nemo_rl_nightly.sqsh)}
export HF_HOME=${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/hf_home}
export WANDB_HOME=${WANDB_HOME:-/home/${USER}/.config/nemo-rl-wandb}
export RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/experiments/qwen235-fp64-router-padding-20260908}
export SLURM_ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
export PARTITION=${PARTITION:-gb200}
export WALLTIME=${WALLTIME:-05:00:00}
export USE_GRES=0

exec "${REPO}/experiments/native_mxfp8_source_refit/submit_oci_hsg.sh"
