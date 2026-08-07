#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/..")

set -eou pipefail

# Ensure Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH."
    exit 1
fi

# CONTAINER is expected to be set as an environment variable
if [[ -z "${CONTAINER:-}" ]]; then
    echo "Error: CONTAINER environment variable is not set."
    echo "Usage: CONTAINER=<docker-container> $0 [optional pytest-args...]"
    exit 1
fi

CONTAINER=${CONTAINER}

export HF_HOME=${HF_HOME:-"$PROJECT_ROOT/hf_home"}
mkdir -p "$HF_HOME"

# Check if running in GitLab CI
INTERACTIVE_FLAG=""
if [[ "${CI:-false}" != "true" ]]; then
    # Setting this interactively lets us issue a keyboard interrupt.
    INTERACTIVE_FLAG="-it"
fi

# Use the caller's identity so files written to the bind-mounted checkout keep
# their host ownership. A container-owned tmpfs keeps HOME writable without
# exposing another host path. It remains executable because torch compile loads
# artifacts from the cache. --no-sync reuses the dependency-complete CI image;
# a dependency fingerprint mismatch means the image must be rebuilt.
docker run --user "$(id -u):$(id -g)" $INTERACTIVE_FLAG \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add=SYS_PTRACE \
  --rm \
  --gpus '"device=0,1"' \
  -v "$PROJECT_ROOT:/workspace" \
  -v "$HF_HOME:/hf_home" \
  --tmpfs "/home/nemo-rl:rw,exec,nosuid,nodev,mode=0700,uid=$(id -u),gid=$(id -g)" \
  -e HF_TOKEN \
  -e HF_HOME=/hf_home \
  -e HOME=/home/nemo-rl \
  -e UV_CACHE_DIR=/home/nemo-rl/.cache/uv \
  -w /workspace/tests \
  "$CONTAINER" -- \
  bash -x -c 'uv run --no-sync --group test bash -x ./run_unit.sh "$@"' bash "$@"
