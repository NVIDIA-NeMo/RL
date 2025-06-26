#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath $SCRIPT_DIR/..)

set -eou pipefail

# Ensure Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH."
    exit 1
fi

# CONTAINER is expected to be set as an environment variable
if [[ -z "${CONTAINER:-}" ]]; then
    echo "Error: CONTAINER environment variable is not set."
    echo "Usage: CONTAINER=<docker-container> $0 <script to run, e.g., functional/grpo.sh>"
    exit 1
fi

if [[ $# -ne 1 ]]; then
    echo "Error: Did not provide functional test script to run."
    echo "Usage: CONTAINER=<docker-container> $0 <script to run, e.g., functional/grpo.sh>"
    exit 1
fi

TEST_SCRIPT=$(realpath $1)
CONTAINER=${CONTAINER}

export HF_HOME=${HF_HOME:-$(realpath $SCRIPT_DIR/../hf_home)}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$(realpath $SCRIPT_DIR/../hf_datasets_cache)}
mkdir -p $HF_HOME
mkdir -p $HF_DATASETS_CACHE

# Check if running in GitLab CI
INTERACTIVE_FLAG=""
if [[ "${CI:-false}" != "true" ]]; then
    # Setting this interactively lets us issue a keyboard interrupt.
    INTERACTIVE_FLAG="-it"
fi

# Note: we run as root because:
#  1. running as ray prevents us from writing into the current working directory
#  2. running as ourselves (-u $(id -u):$(id -g)) causes torch compile to fail
#
# The workaround is we launch the job but set umask 000 so all files created as root are rwxrwxrwx.
# We have found that 111 does not always work and can leave the filesystem permissions in a bad state.

# Run the script inside the Docker container with GPU support
docker run -u root $INTERACTIVE_FLAG --ulimit memlock=-1 --ulimit stack=67108864 --rm --gpus '"device=0,1"' \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -v $HF_HOME:/hf_home \
  -v $HF_DATASETS_CACHE:/hf_datasets_cache \
  -e WANDB_API_KEY \
  -e HF_TOKEN \
  -e HF_HOME=/hf_home \
  -e HF_DATASETS_CACHE=/hf_datasets_cache \
  -e HOME=/tmp/ \
  -w $SCRIPT_DIR \
  "$CONTAINER" -- \
  bash -x -c "umask 000 && uv run bash -x $TEST_SCRIPT"
