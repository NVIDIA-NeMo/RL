#!/usr/bin/env bash

SIF_DIR=${SIF_DIR:-/opt/data}
WORK_DIR=${WORK_DIR:-/workspace/sif}
STATE_DIR=${STATE_DIR:-/workspace/state}
MAX_WORKERS=${MAX_WORKERS:-1}

INSTANCE_FILE="${1:-/workspace/r2e-gym-instances-to-build.txt}"

python /workspace/build_swe_sif_images.py \
    --r2e-gym-ids-file "${INSTANCE_FILE}" \
    --max-workers $MAX_WORKERS \
    --sif-dir $SIF_DIR \
    --work-dir $WORK_DIR \
    --registry $DOCKER_REGISTRY \
    $@
