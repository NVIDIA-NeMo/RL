#!/bin/bash

# ---- User paths ----
export DATA_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/data/data
export SANDBOX_CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/sqsh/nemo-skills-sandbox.sqsh
export SWE_CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/sqsh/nemo_rl.super_v3.sqsh
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/cache}"

# /lustre/fs1 and /lustre/fsw are separate Lustre filesystems; mount both
# explicitly because a single /lustre:/lustre bind does not propagate sub-mounts.
# Also bind the prefetched gym venvs into /opt/gym_venvs (where NEMO_GYM_VENV_DIR points).
export GYM_VENVS_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/gym_venvs
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-/lustre/fs1:/lustre/fs1,/lustre/fsw:/lustre/fsw,${GYM_VENVS_DIR}:/opt/gym_venvs}"

# ---- SLURM / W&B ----
export SLURM_PARTITION=batch
export SLURM_ACCOUNT=coreai_dlalgo_nemorl
# Avoid pool0-00522 — its srun died early in job 11725718 (pyxis extraction issue).
export EXCLUDE_NODES="${EXCLUDE_NODES:-pool0-00522}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be exported before running this script}"

# ---- SWE-bench Apptainer .sif images (required for swe2) ----
# CONTAINER_FORMATTER is now baked into the YAML config (env.nemo_gym.swe_agents_train...)
# to avoid shell-stripping ${sif_dir} when passed as a Hydra CLI override.
export SIF_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/sif

export MODEL_PATH=/lustre/fsw/portfolios/llmservice/users/igitman/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
export CONFIG_PATH=examples/configs/recipes/llm/performance/grpo-nemotron-nano-16n8g-async-1off-swe.yaml
export TRAIN_PATH=$DATA_DIR/swe2/train-split.jsonl
export VAL_PATH=$DATA_DIR/swe2/val-split.jsonl
export CONTAINER=$SWE_CONTAINER
export EXP_NAME=stage2.2-swe2-nano-16n

mkdir -p "${PERSISTENT_CACHE}"

./super_launch.sh