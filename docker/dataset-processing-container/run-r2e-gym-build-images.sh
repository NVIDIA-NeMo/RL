#!/usr/bin/env bash
set -euo pipefail

STATE_DIR=${STATE_DIR:-/workspace/state}
MAX_WORKERS=${MAX_WORKERS:-1}
INSTANCE_FILE="${1:-/workspace/r2e-gym-instances-to-build.txt}"

uv run --with datasets python /workspace/build_r2e_gym_images.py \
    --instance-file "${INSTANCE_FILE}" \
    --state-file "${STATE_DIR}/r2e_gym_build_push_state.json" \
    --max-workers "${MAX_WORKERS}" \
    --cleanup-local \
    --push \
    --retry-failed \
    --registry "${DOCKER_REGISTRY}/r2e-gym"
