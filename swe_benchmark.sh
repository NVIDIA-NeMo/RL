#!/bin/bash

# required env vars
: "${DATA_DIR:?DATA_DIR is required}"
: "${SANDBOX_CONTAINER:?SANDBOX_CONTAINER is required}"
: "${SWE_CONTAINER:?SWE_CONTAINER is required}"
: "${PERSISTENT_CACHE:?PERSISTENT_CACHE is required}"
: "${SLURM_PARTITION:?SLURM_PARTITION is required}"
: "${SLURM_ACCOUNT:?SLURM_ACCOUNT is required}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
: "${CONTAINER_FORMATTER:?CONTAINER_FORMATTER is required}"

export MODEL_PATH=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
export CONFIG_PATH=examples/configs/recipes/llm/performance/grpo-nemotron-nano-16n8g-async-1off-swe.yaml
export TRAIN_PATH=$DATA_DIR/swe2/train-split.jsonl
export VAL_PATH=$DATA_DIR/swe2/val-split.jsonl
export CONTAINER=$SWE_CONTAINER
export EXP_NAME=stage2.2-swe2-nano-64n

./super_launch.sh