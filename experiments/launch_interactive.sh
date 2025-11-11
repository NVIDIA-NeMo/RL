#!/bin/sh

export ACCOUNT=coreai_dlalgo_nemorl
export PARTITION=interactive
export NUM_NODES=2

CONTAINER=/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:main-c249efc8.squashfs
HF_HOME=/lustre/fs1/portfolios/coreai/users/ffrujeri/hf_home

# SETTINGS
export NAME="grpo-acemath_rl-test-interactive"
export project_name="nemo-rl-acemath-rl"
export VAL_PERIOD=1
export SAVE_PERIOD=5
export TOP_K_CHECKPOINT=10
export RESULTS_DIR="results/${NAME}"

RAY_DEDUP_LOGS=0 \
CONTAINER="${CONTAINER}" \
MOUNTS="/lustre:/lustre:ro,$PWD:$PWD" \
sbatch \
    --nodes="${NUM_NODES}" \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --job-name="$(whoami)-${NAME}" \
    --time=4:00:00 \
    --gres=gpu:8 \
    ray.sub
