#!/bin/bash

# set all these env vars to the values you want to use
export DATA_DIR=
export MODEL_PATH=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
export SANDBOX_CONTAINER=
export SWE_CONTAINER=
export PERSISTENT_CACHE=
export SLURM_PARTITION=
export SLURM_ACCOUNT=
export WANDB_API_KEY=
export CONFIG_PATH=examples/configs/recipes/llm/performance/grpo-nemotron-nano-16n8g-async-1off-swe.yaml
export TRAIN_PATH=$DATA_DIR/swe2/train-split.jsonl
export VAL_PATH=$DATA_DIR/swe2/val-split.jsonl
export CONTAINER=$SWE_CONTAINER
export EXP_NAME=stage2.2-swe2-nano-64n
export CONTAINER_FORMATTER=

./super_launch.sh