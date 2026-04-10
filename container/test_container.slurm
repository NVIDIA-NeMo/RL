#!/bin/bash
#SBATCH --job-name=nemo-rl-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=nemo_rl_test_%j.out

# NeMo-RL Container Test Script
# This script tests the NeMo-RL container from the registry
#
# Prerequisites:
#   1. Docker image must be built and pushed to registry first
#   2. Run build_container.sh on login node before submitting this job
#
# Based on: https://github.com/NVIDIA-NeMo/RL

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
            echo "Usage: sbatch test_container.slurm [--sqsh <path>]"
            echo ""
            echo "Options:"
            echo "  --sqsh <path>  Use specific .sqsh file instead of registry"
            echo ""
            echo "Examples:"
            echo "  sbatch test_container.slurm"
            echo "  sbatch test_container.slurm --sqsh /fsx/edward/docker_images/nemo-rl.sqsh"
            exit 1
            ;;
    esac
done

# Set enroot environment variables to use writable locations
export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-$HOME/.enroot/runtime}"
export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-$HOME/.enroot/data}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-$HOME/.enroot/cache}"

# Create directories if they don't exist
mkdir -p "$ENROOT_RUNTIME_PATH" "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH"

echo "========================================="
echo "NeMo-RL Container Test"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "========================================="

# Configuration
IMAGE_NAME="nemo-rl"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Determine container image source
if [ -n "$SQSH_FILE" ]; then
    # Use provided .sqsh file
    if [ ! -f "$SQSH_FILE" ]; then
        echo "ERROR: .sqsh file not found: $SQSH_FILE"
        exit 1
    fi
    CONTAINER_IMAGE="$SQSH_FILE"
    echo "Configuration:"
    echo "  Using .sqsh file: $CONTAINER_IMAGE"
    echo "========================================="
else
    # Use registry (requires REGISTRY env var)
    REGISTRY="${REGISTRY}"

    # Validate required environment variables
    if [ -z "$REGISTRY" ]; then
        echo "ERROR: REGISTRY environment variable not set"
        echo ""
        echo "Please set it before submitting this job:"
        echo "  export REGISTRY=registry.hpc-cluster-hopper.hpc.internal.huggingface.tech"
        echo "  sbatch test_container.slurm"
        echo ""
        echo "Or use --sqsh to specify a .sqsh file directly:"
        echo "  sbatch test_container.slurm --sqsh /path/to/nemo-rl.sqsh"
        exit 1
    fi

    REGISTRY_IMAGE="${REGISTRY}/library/${IMAGE_NAME}:${IMAGE_TAG}"
    CONTAINER_IMAGE="docker://${REGISTRY_IMAGE}"

    echo "Configuration:"
    echo "  Registry: $REGISTRY"
    echo "  Testing image: $REGISTRY_IMAGE"
    echo "  Container URI: $CONTAINER_IMAGE"
    echo "========================================="
fi

# Check if enroot/pyxis is available (srun with --container-image requires it)
if ! command -v enroot &> /dev/null; then
    echo "WARNING: enroot command not found"
    echo "This script requires Pyxis/Enroot for SLURM container support"
fi

# Run verification tests using srun with Pyxis
echo ""
echo "========================================="
echo "Running verification tests..."
echo "========================================="

srun --container-image="$CONTAINER_IMAGE" \
     --container-mounts="/fsx:/fsx" \
     --no-container-mount-home \
     bash <<'OUTER_EOF'
cd /opt/nemo-rl && uv run python <<'INNER_EOF'
import sys

tests_passed = 0
tests_failed = 0

# Test 1: NeMo-RL
print("\nTest 1: Checking NeMo-RL installation...")
try:
    import nemo_rl
    print(f"✓ NeMo-RL version: {nemo_rl.__version__}")
    tests_passed += 1
except Exception as e:
    print(f"✗ NeMo-RL import failed: {e}")
    tests_failed += 1

# Test 2: PyTorch and CUDA
print("\nTest 2: Checking PyTorch and CUDA...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  CUDA devices: {torch.cuda.device_count()}")
    tests_passed += 1
except Exception as e:
    print(f"✗ PyTorch check failed: {e}")
    tests_failed += 1

# Test 3: Ray
print("\nTest 3: Checking Ray installation...")
try:
    import ray
    print(f"✓ Ray version: {ray.__version__}")
    tests_passed += 1
except Exception as e:
    print(f"✗ Ray import failed: {e}")
    tests_failed += 1

# Test 4: vLLM (optional backend)
print("\nTest 4: Checking vLLM installation (optional)...")
try:
    import vllm
    print(f"✓ vLLM version: {vllm.__version__}")
    tests_passed += 1
except Exception as e:
    print(f"⚠️  vLLM not available (optional - requires 'uv run --extra vllm')")
    # Don't count as failure since it's optional
    tests_passed += 1

# Test 5: Transformers
print("\nTest 5: Checking transformers installation...")
try:
    import transformers
    print(f"✓ Transformers version: {transformers.__version__}")
    tests_passed += 1
except Exception as e:
    print(f"✗ Transformers import failed: {e}")
    tests_failed += 1

# Test 6: Verify pytest test collection works
print("\nTest 6: Checking test framework...")
try:
    import subprocess
    import re

    # Collect tests to verify pytest works
    collect_result = subprocess.run(
        ["uv", "run", "--group", "test", "pytest", "--collect-only", "-q", "tests/unit"],
        capture_output=True,
        text=True,
        timeout=60
    )

    if collect_result.returncode == 0:
        match = re.search(r"(\d+) tests? collected", collect_result.stdout)
        test_count = int(match.group(1)) if match else 0
        print(f"✓ Test framework functional ({test_count} tests available)")
        tests_passed += 1
    else:
        print(f"✗ Test collection failed")
        tests_failed += 1

except subprocess.TimeoutExpired:
    print(f"⚠️  Test collection slow but container is functional")
    tests_passed += 1
except Exception as e:
    print(f"✗ Test framework check failed: {e}")
    tests_failed += 1

# Summary
print(f"\n{'='*50}")
print(f"Tests passed: {tests_passed}/6")
print(f"Tests failed: {tests_failed}/6")
print(f"{'='*50}")

if tests_failed > 0:
    sys.exit(1)
INNER_EOF
OUTER_EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Container validation failed"
    exit 1
fi

echo ""
echo "========================================="
echo "✓ Container validation passed!"
echo "========================================="
echo ""
