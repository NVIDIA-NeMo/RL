#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Build the train/val jsonl subsets from the R2E-Gym-Subset parquet shards.
# Usage: DATASET_DIR=<R2E-Gym__R2E-Gym-Subset> OUTPUT_DIR=<dest> CACHE_DIR=<cache> bash create_subset.sh
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

: "${DATASET_DIR:?DATASET_DIR not set (R2E-Gym__R2E-Gym-Subset with data/train-*.parquet)}"
: "${OUTPUT_DIR:?OUTPUT_DIR not set}"
: "${CACHE_DIR:?CACHE_DIR not set}"
# Must match the NEMO_GYM_SWE_SIF_DIR container path (container_formatter prefix)
: "${CONTAINER_IMAGE_DIR:=/inputs/nemo_gym/sif}"

python3 "${SCRIPT_DIR}/create_r2e_gym_easy_subset_jsonl.py" \
    --dataset-dir "${DATASET_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --container-image-dir "${CONTAINER_IMAGE_DIR}" \
    --train-ids "${SCRIPT_DIR}/instances_train.txt" \
    --val-ids "${SCRIPT_DIR}/instances_val.txt"
