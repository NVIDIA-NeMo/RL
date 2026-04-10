#!/bin/bash
# NeMo-RL Container Build Script
# Run this on the node where Docker is available (typically the login node)
# This script builds the Docker image and pushes it to your registry

set -e

echo "========================================="
echo "NeMo-RL Container Build Script"
echo "========================================="
echo "This script must run on a node with Docker access (login node)"
echo "Started at: $(date)"
echo "========================================="

# Parse command line arguments
SQSH_OUTPUT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --sqsh-output)
            SQSH_OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo ""
            echo "Usage: $0 [--sqsh-output <path>]"
            echo ""
            echo "Options:"
            echo "  --sqsh-output <path>  Create .sqsh file at specified path (requires enroot)"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Build only"
            echo "  $0 --sqsh-output nemo-rl.sqsh        # Build and create .sqsh"
            exit 1
            ;;
    esac
done

# Configuration
FRAMEWORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$FRAMEWORK_DIR/../.." && pwd)"
IMAGE_NAME="nemo-rl"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_IMAGES_DIR="${CONTAINER_IMAGES_DIR:-/fsx/edward/docker_images}"

# Path to local NeMo-RL clone (default: the repo this container dir lives in)
NEMO_RL_SOURCE="${NEMO_RL_SOURCE:-$(cd "$FRAMEWORK_DIR/.." && pwd)}"

# Registry configuration
REGISTRY="${REGISTRY:-registry.hpc-cluster-hopper.hpc.internal.huggingface.tech}"

if [ -z "$REGISTRY" ]; then
    echo "ERROR: REGISTRY environment variable not set"
    echo "Please set it to your cluster's Docker registry:"
    echo "  export REGISTRY=registry.hpc-cluster-hopper.hpc.internal.huggingface.tech"
    exit 1
fi

# Validate local source exists
if [ ! -d "$NEMO_RL_SOURCE" ]; then
    echo "ERROR: NeMo-RL source directory not found: $NEMO_RL_SOURCE"
    echo ""
    echo "Expected to find local NeMo-RL clone at: $NEMO_RL_SOURCE"
    echo ""
    echo "Options:"
    echo "  1. Clone NeMo-RL first:"
    echo "     cd $REPO_ROOT"
    echo "     ./shared/scripts/clone_frameworks.sh nemo-rl"
    echo ""
    echo "  2. Set NEMO_RL_SOURCE to your local clone:"
    echo "     export NEMO_RL_SOURCE=/path/to/your/nemo-rl"
    exit 1
fi

# Check if it's a git repository
if [ ! -d "$NEMO_RL_SOURCE/.git" ]; then
    echo "ERROR: $NEMO_RL_SOURCE is not a git repository"
    echo ""
    echo "The local source must be a git clone with .git directory"
    echo "Please clone it properly using git"
    exit 1
fi

LOCAL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
REGISTRY_IMAGE="${REGISTRY}/library/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Configuration:"
echo "  Framework directory: $FRAMEWORK_DIR"
echo "  Local NeMo-RL source: $NEMO_RL_SOURCE"
echo "  Local image: $LOCAL_IMAGE"
echo "  Registry image: $REGISTRY_IMAGE"
echo ""
echo "Using local NeMo-RL clone (commit info):"
cd "$NEMO_RL_SOURCE"
git --no-pager log -1 --pretty=format:"  Commit: %h%n  Date: %ai%n  Message: %s%n" || echo "  (Unable to read git info)"
cd "$FRAMEWORK_DIR"
echo "========================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not available on this node"
    echo ""
    echo "Please run this script on a node with Docker access (typically the login node)"
    echo "If you are on a login node and Docker is still not available, contact your cluster admin"
    exit 1
fi

# Check Docker daemon connectivity
if ! docker info &> /dev/null; then
    echo "ERROR: Cannot connect to Docker daemon"
    echo ""
    echo "Docker is installed but the daemon is not accessible."
    echo "Please check:"
    echo "  1. Are you on the correct login node?"
    echo "  2. Is the Docker service running?"
    echo "  3. Do you have permissions to access Docker?"
    exit 1
fi

echo "✓ Docker is available and daemon is accessible"
echo ""

# Build Docker image using official Dockerfile with local source
echo "========================================="
echo "Building Docker image from local source..."
echo "This will take 30-60 minutes due to:"
echo "  - Installing PyTorch 2.8.0, Ray 2.49.2"
echo "  - Building vLLM and other dependencies"
echo "  - Installing NeMo-RL from local clone"
echo ""
echo "Using official NeMo-RL Dockerfile with multi-stage build"
echo "Target stage: release"
echo "Build context: Using local NeMo-RL source"
echo "========================================="

cd "$FRAMEWORK_DIR"

# The official Dockerfile uses multi-stage builds
# Stages: nemo-rl (source) -> base -> hermetic (deps) -> release (final)
# We override the nemo-rl stage with local source using --build-context

# Build with buildx using local source override
if docker buildx version &> /dev/null; then
    echo "Using docker buildx for build..."
    docker buildx build \
        --target release \
        --build-context nemo-rl="$NEMO_RL_SOURCE" \
        --build-arg SKIP_SGLANG_BUILD="${SKIP_SGLANG_BUILD:-}" \
        -t $LOCAL_IMAGE \
        --load \
        .
else
    echo "ERROR: docker buildx is required for local source override"
    echo ""
    echo "The --build-context flag requires docker buildx"
    echo "Please install docker buildx or use Docker Desktop"
    exit 1
fi

if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully: $LOCAL_IMAGE"
else
    echo "ERROR: Docker image build failed"
    exit 1
fi

# Tag for registry
echo ""
echo "========================================="
echo "Tagging image for registry..."
echo "========================================="

docker tag $LOCAL_IMAGE $REGISTRY_IMAGE

if [ $? -eq 0 ]; then
    echo "✓ Image tagged: $REGISTRY_IMAGE"
else
    echo "ERROR: Failed to tag image"
    exit 1
fi

# Check if image already exists in registry
echo ""
echo "========================================="
echo "Checking registry for existing image..."
echo "========================================="
if docker manifest inspect $REGISTRY_IMAGE &> /dev/null; then
    echo "⚠️  WARNING: Image already exists in registry: $REGISTRY_IMAGE"
    echo ""
    echo "The existing image will be overwritten."
    echo "If you want to preserve it, consider using a different tag:"
    echo "  export IMAGE_TAG=v1.0.0"
    echo "  export IMAGE_TAG=\$(date +%Y%m%d-%H%M%S)"
    echo ""
    echo "Press any key to continue or Ctrl+C to cancel..."
    read -n 1 -s -r
else
    echo "✓ No existing image found in registry (new image)"
fi

# Push to registry
echo ""
echo "========================================="
echo "Pushing to registry..."
echo "This may take several minutes..."
echo "========================================="

docker push $REGISTRY_IMAGE

if [ $? -eq 0 ]; then
    echo "✓ Image pushed successfully to registry"
else
    echo "ERROR: Failed to push image to registry"
    echo ""
    echo "Please check:"
    echo "  1. Is the registry URL correct? ($REGISTRY)"
    echo "  2. Do you have push permissions to the registry?"
    echo "  3. Is the registry accessible from this node?"
    exit 1
fi

# Verify the push
echo ""
echo "========================================="
echo "Verifying registry push..."
echo "========================================="

# Try to pull the image to verify it's available
docker pull $REGISTRY_IMAGE &> /dev/null

if [ $? -eq 0 ]; then
    echo "✓ Image verified in registry"
else
    echo "WARNING: Could not verify image in registry (may still be accessible)"
fi

# Optional: Create .sqsh file with enroot
if [ -n "$SQSH_OUTPUT" ]; then
    echo ""
    echo "========================================="
    echo "Creating .sqsh file with enroot..."
    echo "========================================="

    # Check if enroot is available
    if ! command -v enroot &> /dev/null; then
        echo "ERROR: enroot is not available on this node"
        echo ""
        echo "The --sqsh-output option requires enroot to be installed."
        echo "You can create the .sqsh file later on compute nodes:"
        echo "  enroot import \"docker://${REGISTRY_IMAGE}\""
        echo "  mv library+${IMAGE_NAME}+${IMAGE_TAG}.sqsh ${SQSH_OUTPUT}"
        exit 1
    fi

    echo "✓ Enroot is available"

    # Set enroot environment variables to use writable locations
    export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-$HOME/.enroot/runtime}"
    export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-$HOME/.enroot/data}"
    export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-$HOME/.enroot/cache}"

    # Create directories if they don't exist
    mkdir -p "$ENROOT_RUNTIME_PATH" "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH"

    echo "Using enroot directories:"
    echo "  Runtime: $ENROOT_RUNTIME_PATH"
    echo "  Data: $ENROOT_DATA_PATH"
    echo "  Cache: $ENROOT_CACHE_PATH"

    # Determine output path
    if [[ "$SQSH_OUTPUT" = /* ]]; then
        # Absolute path provided
        SQSH_FINAL_PATH="$SQSH_OUTPUT"
        SQSH_DIR="$(dirname "$SQSH_FINAL_PATH")"
    else
        # Relative filename provided, use CONTAINER_IMAGES_DIR
        SQSH_FINAL_PATH="${CONTAINER_IMAGES_DIR}/${SQSH_OUTPUT}"
        SQSH_DIR="$CONTAINER_IMAGES_DIR"
    fi

    # Create output directory if needed
    mkdir -p "$SQSH_DIR"

    # Import with enroot
    echo ""
    echo "Importing from registry to .sqsh format..."
    echo "This may take 5-10 minutes..."
    echo ""

    enroot import "docker://${REGISTRY_IMAGE}"

    if [ $? -eq 0 ]; then
        ENROOT_FILENAME="library+${IMAGE_NAME}+${IMAGE_TAG}.sqsh"

        if [ -f "$ENROOT_FILENAME" ]; then
            # Remove existing file if it exists at destination
            if [ -f "$SQSH_FINAL_PATH" ]; then
                echo "Removing existing .sqsh file: $SQSH_FINAL_PATH"
                rm -f "$SQSH_FINAL_PATH"
            fi

            # Move to final location
            mv "$ENROOT_FILENAME" "$SQSH_FINAL_PATH"

            if [ $? -eq 0 ]; then
                SQSH_SIZE=$(du -h "$SQSH_FINAL_PATH" | cut -f1)
                echo "✓ .sqsh file created successfully: $SQSH_FINAL_PATH"
                echo "  Size: $SQSH_SIZE"
            else
                echo "ERROR: Failed to move .sqsh file to $SQSH_FINAL_PATH"
                exit 1
            fi
        else
            echo "ERROR: .sqsh file not found after import: $ENROOT_FILENAME"
            echo "The import may have failed or created the file with a different name"
            exit 1
        fi
    else
        echo "ERROR: enroot import failed"
        echo ""
        echo "Please check:"
        echo "  1. Is the registry accessible from this node?"
        echo "  2. Is the image available in the registry?"
        echo "  3. Do you have sufficient disk space?"
        exit 1
    fi
fi

echo ""
echo "========================================="
echo "Build Completed Successfully!"
echo "Finished at: $(date)"
echo "========================================="
echo ""
echo "Image pushed to: $REGISTRY_IMAGE"

if [ -n "$SQSH_OUTPUT" ]; then
    echo ".sqsh file created: $SQSH_FINAL_PATH"
fi

echo ""
echo "Next steps:"
echo ""

if [ -n "$SQSH_OUTPUT" ]; then
    echo "1. Test the .sqsh file on compute nodes:"
    echo ""
    echo "   sbatch test_container.slurm --sqsh \"${SQSH_FINAL_PATH}\""
    echo ""
    echo "2. Use the .sqsh file in your training jobs:"
    echo ""
    echo "   export CONTAINER_IMAGE=\"${SQSH_FINAL_PATH}\""
    echo ""
    echo "   srun --gpus-per-node=8 \\"
    echo "     --container-image=\"\${CONTAINER_IMAGE}\" \\"
    echo "     --container-mounts=\"/fsx:/fsx,/scratch:/scratch\" \\"
    echo "     --no-container-mount-home \\"
    echo "     bash -c 'cd /opt/nemo-rl && uv run python examples/run_grpo_math.py'"
else
    echo "1. Test the container on compute nodes:"
    echo ""
    echo "   export REGISTRY=\"${REGISTRY}\""
    echo "   sbatch test_container.slurm"
    echo ""
    echo "2. Use the container in your jobs:"
    echo ""
    echo "   Option A - Direct from registry (simplest):"
    echo "   export CONTAINER_IMAGE=\"docker://${REGISTRY_IMAGE}\""
    echo ""
    echo "   Option B - Create .sqsh file for better performance:"
    echo "   ./build_container.sh --sqsh-output nemo-rl.sqsh"
    echo "   # Or manually:"
    echo "   enroot import \"docker://${REGISTRY_IMAGE}\""
    echo "   mv library+nemo-rl+${IMAGE_TAG}.sqsh /fsx/edward/docker_images/"
    echo ""
    echo "3. Run training:"
    echo ""
    echo "   srun --gpus-per-node=8 \\"
    echo "     --container-image=\"\${CONTAINER_IMAGE}\" \\"
    echo "     --container-mounts=\"/fsx:/fsx,/scratch:/scratch\" \\"
    echo "     --no-container-mount-home \\"
    echo "     bash -c 'cd /opt/nemo-rl && uv run python examples/run_grpo_math.py'"
fi

echo ""
