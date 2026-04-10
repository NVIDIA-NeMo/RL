#!/bin/bash
#SBATCH --job-name=nemo-rl-full-tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=nemo_rl_full_tests_%j.out

# NeMo-RL Full Test Suite
# Runs all 984 unit tests with 2 GPUs
#
# Based on: https://github.com/NVIDIA-NeMo/RL/blob/main/docs/testing.md

set -e

# Parse command line arguments
SQSH_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --sqsh)
            SQSH_FILE="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo ""
            echo "Usage: sbatch run_full_tests.slurm [--sqsh <path>]"
            echo ""
            echo "Options:"
            echo "  --sqsh <path>  Use specific .sqsh file instead of registry"
            echo ""
            echo "Examples:"
            echo "  sbatch run_full_tests.slurm --sqsh /fsx/edward/docker_images/nemo-rl.sqsh"
            exit 1
            ;;
    esac
done

# Set enroot environment variables
export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-$HOME/.enroot/runtime}"
export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-$HOME/.enroot/data}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-$HOME/.enroot/cache}"

mkdir -p "$ENROOT_RUNTIME_PATH" "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH"

echo "========================================="
echo "NeMo-RL Full Test Suite"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "Started at: $(date)"
echo "========================================="

# Configuration
IMAGE_NAME="nemo-rl"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Determine container image source
if [ -n "$SQSH_FILE" ]; then
    if [ ! -f "$SQSH_FILE" ]; then
        echo "ERROR: .sqsh file not found: $SQSH_FILE"
        exit 1
    fi
    CONTAINER_IMAGE="$SQSH_FILE"
    echo "Using .sqsh file: $CONTAINER_IMAGE"
else
    REGISTRY="${REGISTRY}"
    if [ -z "$REGISTRY" ]; then
        echo "ERROR: REGISTRY environment variable not set"
        echo "Please set it or use --sqsh option"
        exit 1
    fi
    REGISTRY_IMAGE="${REGISTRY}/library/${IMAGE_NAME}:${IMAGE_TAG}"
    CONTAINER_IMAGE="docker://${REGISTRY_IMAGE}"
    echo "Using registry: $CONTAINER_IMAGE"
fi

echo "========================================="

# Run full test suite
echo ""
echo "========================================="
echo "Running full NeMo-RL test suite..."
echo "This will run all 984 unit tests"
echo "Expected time: 30-60 minutes"
echo "========================================="
echo ""

srun --container-image="$CONTAINER_IMAGE" \
     --container-mounts="/fsx:/fsx" \
     --no-container-mount-home \
     bash -c 'cd /opt/nemo-rl && uv run --group test bash tests/run_unit.sh'

TEST_EXIT_CODE=$?

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "========================================="
    echo "✓ All tests passed!"
    echo "========================================="
else
    echo "========================================="
    echo "✗ Tests failed (exit code: $TEST_EXIT_CODE)"
    echo "========================================="
fi
echo ""

exit $TEST_EXIT_CODE
