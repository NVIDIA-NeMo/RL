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
DEFAULT_GIT_URL="https://github.com/NVIDIA/TransformerEngine.git"
DEFAULT_BRANCH="v2.3"

# Parse command line arguments
GIT_URL=${1:-$DEFAULT_GIT_URL}
BRANCH=${2:-$DEFAULT_BRANCH}

BUILD_DIR=$(realpath "$SCRIPT_DIR/../3rdparty/TransformerEngine")
if [[ -e "$BUILD_DIR" ]]; then
  echo "[ERROR] $BUILD_DIR already exists. Please remove or move it before running this script."
  exit 1 
fi

echo "Building transformer-engine-torch wheel from:"
echo "  TransformerEngine Git URL: $GIT_URL"
echo "  TransformerEngine Branch: $BRANCH"

# Clone the repository
echo "Cloning repository..."
git clone "$GIT_URL" "$BUILD_DIR"
cd "$BUILD_DIR"
git checkout "$BRANCH"

# Create a new Python environment using uv
echo "Creating Python environment..."
uv venv

# Instructions below are based on
# https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/.github/workflows/publish.yml

# Install dependencies
echo "Installing dependencies..."
uv pip install --upgrade setuptools==75.8.0
uv pip install ninja packaging wheel
uv pip install torch==2.7.1 --torch-backend=cu128

# Build the wheel
echo "Building transformer-engine-torch wheel..."
cd transformer_engine/pytorch
NVTE_RELEASE_BUILD=1 MAX_JOBS=$(($(nproc)/2)) \
	uv run python setup.py bdist_wheel --dist-dir=dist

echo "Build completed successfully!"
echo "The built transformer-engine-torch is available at: $(realpath $PWD/dist/*.whl)"
