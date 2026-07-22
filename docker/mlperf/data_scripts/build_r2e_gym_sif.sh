#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Build the R2E-Gym SWE instance SIF images (NEMO_GYM_SWE_SIF_DIR): step 1
# builds+pushes per-instance arm64 docker images, step 2 converts them to SIF.
# Usage (docker-capable node, fast scratch):
#   export DOCKER_REGISTRY=... DOCKER_USER=... DOCKER_TOKEN=...
#   export SIF_DIR=<lustre target> WORK_DIR=<scratch> STATE_DIR=<scratch>
#   bash build_r2e_gym_sif.sh
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
CONTAINER_DIR="${SCRIPT_DIR}/dataset-processing-container"

: "${DOCKER_REGISTRY:?DOCKER_REGISTRY not set}"
: "${DOCKER_USER:?DOCKER_USER not set}"
: "${DOCKER_TOKEN:?DOCKER_TOKEN not set}"
: "${SIF_DIR:?SIF_DIR not set}"
: "${WORK_DIR:=/tmp/r2e-sif-work}"
: "${STATE_DIR:=/tmp/r2e-sif-state}"
: "${MAX_WORKERS:=4}"
: "${DATASET_CONTAINER_IMAGE:=qwen35-grpo-dataset-processing}"

mkdir -p "${SIF_DIR}" "${WORK_DIR}" "${STATE_DIR}"

docker build -t "${DATASET_CONTAINER_IMAGE}" "${CONTAINER_DIR}"

_common_docker_args=(
    --rm
    -e DOCKER_REGISTRY -e DOCKER_USER -e DOCKER_TOKEN
    -e MAX_WORKERS
    -e SIF_DIR=/opt/data -e WORK_DIR=/workspace/sif -e STATE_DIR=/workspace/state
    -v /var/run/docker.sock:/var/run/docker.sock
    -v "${SIF_DIR}:/opt/data"
    -v "${WORK_DIR}:/workspace/sif"
    -v "${STATE_DIR}:/workspace/state"
)

docker run "${_common_docker_args[@]}" "${DATASET_CONTAINER_IMAGE}" \
    bash /workspace/run-r2e-gym-build-images.sh

docker run "${_common_docker_args[@]}" "${DATASET_CONTAINER_IMAGE}" \
    bash /workspace/run-build-sif-images.sh

echo "SIF images written to ${SIF_DIR}"
