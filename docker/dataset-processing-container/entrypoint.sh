#!/usr/bin/env bash
set -euo pipefail

docker login -u "$DOCKER_USER" -p "$DOCKER_TOKEN" "$DOCKER_REGISTRY"
exec /bin/bash -c "$@"
