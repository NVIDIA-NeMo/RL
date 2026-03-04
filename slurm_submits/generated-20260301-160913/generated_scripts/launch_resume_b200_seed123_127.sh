#!/usr/bin/env bash
set -euo pipefail

GEN_DIR="/home/scratch.shaunakj_other/Development/RL/slurm_submits/generated-20260301-160913"
cd "$GEN_DIR"

export RESUME_CKPT_DIR_123="${RESUME_CKPT_DIR_123:-/home/scratch.shaunakj_other/results/tar-drift-reasonable-eagle3-s4-seed123-train-steps135-2026-03-02-045851}"
export RESUME_CKPT_DIR_127="${RESUME_CKPT_DIR_127:-/home/scratch.shaunakj_other/results/tar-drift-reasonable-eagle3-s4-seed127-train-steps135-2026-03-02-060549}"
export RUN_STEPS="${RUN_STEPS:-270}"

export SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:---partition=b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb --gpus-per-node=8 --cpus-per-task=224}"
export SBATCH_EXPORT_VARS="${SBATCH_EXPORT_VARS:-ALL,NEMO_RL_PY_EXECUTABLES_SYSTEM=1,NRL_FORCE_REBUILD_VENVS=false,RAY_ENABLE_UV_RUN_RUNTIME_ENV=0}"

./submit_resume_train_123_127.sh
