#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eoux pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
DEFAULT_GIT_URL="https://github.com/terrykong/vllm.git"
DEFAULT_BRANCH="terryk/demo-custom-vllm"
DEFAULT_VLLM_COMMIT=a3319f4f04fbea7defe883e516df727711e516cd # use full commit hash from the main branch

# Parse command line arguments
GIT_URL=${1:-$DEFAULT_GIT_URL}
BRANCH=${2:-$DEFAULT_BRANCH}
export VLLM_COMMIT=${3:-$DEFAULT_VLLM_COMMIT}
export VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/${DEFAULT_VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"

BUILD_DIR=$(realpath "$SCRIPT_DIR/../3rdparty/vllm")
if [[ -e "$BUILD_DIR" ]]; then
  echo "[ERROR] $BUILD_DIR already exists. Please remove or move it before running this script."
  exit 1 
fi

echo "Building vLLM from:"
echo "  Vllm Git URL: $GIT_URL"
echo "  Vllm Branch: $BRANCH"
echo "  VLLM Wheel Commit: $VLLM_COMMIT"
echo "  VLLM Precompiled Wheel Location: $VLLM_PRECOMPILED_WHEEL_LOCATION"

# Clone the repository
echo "Cloning repository..."
git clone "$GIT_URL" "$BUILD_DIR"
cd "$BUILD_DIR"
git checkout "$BRANCH"

# Create a new Python environment using uv
echo "Creating Python environment..."
uv venv

# Remove all comments from requirements files to prevent use_existing_torch.py from incorrectly removing xformers
echo "Removing comments from requirements files..."
find requirements/ -name "*.txt" -type f -exec sed -i 's/#.*$//' {} \; 2>/dev/null || true
find requirements/ -name "*.txt" -type f -exec sed -i '/^[[:space:]]*$/d' {} \; 2>/dev/null || true

uv run --no-project use_existing_torch.py

# Install dependencies
echo "Installing dependencies..."
uv pip install --upgrade pip
uv pip install numpy setuptools setuptools_scm
uv pip install torch==2.7.0 --torch-backend=cu128

# Install vLLM using precompiled wheel
echo "Installing vLLM with precompiled wheel..."
uv pip install --no-build-isolation -e .

echo "Build completed successfully!"
echo "The built vLLM is available in: $BUILD_DIR"
echo "You can now update your pyproject.toml to use this local version."
echo "Follow instructions on https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/use-custom-vllm.md for how to configure your local NeMo RL environment to use this custom vLLM."
