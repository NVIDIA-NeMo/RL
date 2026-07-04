#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR/..")"

# Parse command line arguments
GIT_URL=${1:-https://github.com/NVIDIA/TensorRT-LLM.git}
GIT_REF=${2:-v1.3.0rc20}

BUILD_DIR=$(realpath "$SCRIPT_DIR/../3rdparty")/TensorRT-LLM
if [[ -e "$BUILD_DIR" ]]; then
  echo "[ERROR] $BUILD_DIR already exists. Please remove or move it before running this script."
  exit 1
fi

# Directory where the built wheel is exported for uv to discover.
# Prefer UV_FIND_LINKS (set in Dockerfile ENV) so this script and uv always agree
# on the wheel location. Fall back to WHEEL_OUTPUT_DIR, then /opt/trtllm_wheels.
WHEEL_OUTPUT_DIR=${UV_FIND_LINKS:-/opt/trtllm_wheels}
mkdir -p "$WHEEL_OUTPUT_DIR"

echo "Building TensorRT-LLM from:"
echo "  TRT-LLM Git URL: $GIT_URL"
echo "  TRT-LLM Git ref: $GIT_REF"

# git-lfs is required because TRT-LLM ships its `internal_cutlass_kernels`
# static archives (~67MB on aarch64, ~66MB on x86_64) as LFS-tracked .tar.xz
# files. Without LFS, the working tree contains 130-byte pointer stubs and
# cpp/tensorrt_llm/CMakeLists.txt aborts with "internal_cutlass_kernels library
# is truncated or incomplete".
#
# We fetch git-lfs as a static binary from the upstream GitHub release rather
# than via apt, so the build is reproducible on hosts without root / apt
# (matches Stage 2 (CMake) and Stage 3 (uv) which are also installer-based).
if ! command -v git-lfs >/dev/null 2>&1; then
    echo "Installing git-lfs (static binary, apt-free)..."
    LFS_VER=3.7.0
    LFS_ARCH=$(uname -m | sed 's/aarch64/arm64/;s/x86_64/amd64/')
    LFS_TGZ="/tmp/git-lfs-${LFS_VER}.tar.gz"
    curl --retry 3 --retry-delay 2 -fsSL -o "$LFS_TGZ" \
        "https://github.com/git-lfs/git-lfs/releases/download/v${LFS_VER}/git-lfs-linux-${LFS_ARCH}-v${LFS_VER}.tar.gz"
    LFS_TMP=$(mktemp -d)
    tar -xzf "$LFS_TGZ" -C "$LFS_TMP" --strip-components=1
    install -m 755 "$LFS_TMP/git-lfs" /usr/local/bin/git-lfs
    rm -rf "$LFS_TMP" "$LFS_TGZ"
fi
git lfs install --skip-repo

# Clone TRT-LLM + LFS pull + submodules
echo "Cloning TensorRT-LLM..."
git clone --depth=1 --branch "$GIT_REF" "$GIT_URL" "$BUILD_DIR"
cd "$BUILD_DIR"
echo "Fetching LFS objects (internal_cutlass_kernels archives)..."
git lfs pull
git submodule update --init --recursive --depth=1

# requirements.txt patches:
#   - bump modelopt pin to >=0.44.0a0 to match the runtime venv version; the
#     venv already has 0.44.0a0 installed, so the TRT-LLM wheel build picks it
#     up directly without a separate modelopt from-source build step.
#   - remove `setuptools<80` ceiling. Modern setuptools (>=80) is required by
#     several of our other dependencies (e.g. transformer-engine build deps);
#     downgrading creates an unresolvable conflict in the venv.
sed -i 's|nvidia-modelopt\[torch\]~=0\.37\.0|nvidia-modelopt[torch]>=0.44.0a0|' requirements.txt
sed -i 's|^setuptools<80$|setuptools|' requirements.txt

# cutlass_kernels/CMakeLists.txt invokes `setup_library.py develop --user`,
# which (a) requires a setup.py shim and (b) the `--user` flag is invalid
# inside a venv. Rewrite the COMMAND to copy setup_library.py to setup.py
# (so `develop` finds a buildable target) and drop `--user`.
sed -i 's|COMMAND \${Python3_EXECUTABLE} setup_library.py develop --user|COMMAND bash -c "cp -f setup_library.py setup.py \&\& \${Python3_EXECUTABLE} setup_library.py develop"|' \
    cpp/tensorrt_llm/kernels/cutlass_kernels/CMakeLists.txt

# Build the wheel.
#   -a 100-real: Blackwell (sm_100) only — gb200 target. Bump to include
#                90 for Hopper or 100,90 for both.
#   --nvrtc_dynamic_linking: required so the wheel links against the venv's
#                            libnvrtc-builtins lazily instead of statically.
echo "Building TensorRT-LLM wheel (this takes ~30-60 minutes)..."
python3 scripts/build_wheel.py \
    -a "80-real;90-real;100-real" \
    -G Ninja \
    --clean \
    --nvrtc_dynamic_linking \
    -D "ENABLE_UCX=OFF"

# Copy the wheel to WHEEL_OUTPUT_DIR.
# When called from the PEP 517 backend (_backend.py), WHEEL_OUTPUT_DIR is set to
# the wheel_directory that uv passes to build_wheel — uv then picks up the wheel
# from there and installs it into the venv.
echo "Copying TensorRT-LLM wheel to ${WHEEL_OUTPUT_DIR}..."
cp "$BUILD_DIR"/build/tensorrt_llm-*.whl "$WHEEL_OUTPUT_DIR/"

# Remove source tree and build artifacts to reclaim disk space.
echo "Cleaning up TensorRT-LLM source and build artifacts..."
rm -rf "$BUILD_DIR"

echo "Build completed successfully!"
echo "TRT-LLM wheel copied to: ${WHEEL_OUTPUT_DIR}"
