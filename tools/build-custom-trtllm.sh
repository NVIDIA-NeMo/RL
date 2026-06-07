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
GIT_REF=${2:-v1.3.0rc15}
MODELOPT_GIT_URL=${3:-https://github.com/NVIDIA/TensorRT-Model-Optimizer.git}
MODELOPT_GIT_REF=${4:-0.37.0}

BUILD_DIR=$(realpath "$SCRIPT_DIR/../3rdparty")/TensorRT-LLM
if [[ -e "$BUILD_DIR" ]]; then
  echo "[ERROR] $BUILD_DIR already exists. Please remove or move it before running this script."
  exit 1
fi

echo "Building TensorRT-LLM from:"
echo "  TRT-LLM Git URL: $GIT_URL"
echo "  TRT-LLM Git ref: $GIT_REF"
echo "  ModelOpt Git URL: $MODELOPT_GIT_URL"
echo "  ModelOpt Git ref: $MODELOPT_GIT_REF"

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

# modelopt 0.37 from source. Two things force this:
#   (1) PyPI does not publish a cp313/aarch64 wheel for nvidia-modelopt 0.37.x
#   (2) upstream 0.37.0 pins requires-python = ">=3.10,<3.13" in both
#       pyproject.toml and setup.py — we patch the 3.13 ceiling.
# TRT-LLM v1.3.0rc15's requirements.txt asks for `nvidia-modelopt[torch]~=0.37.0`,
# which would conflict with NeMo-RL main's unpinned `nvidia-modelopt[torch]`
# (uv resolves to 0.43.x at lock time). We install the 0.37 wheel after the
# venv is synced so the trtllm wheel build picks up the matching version.
echo "Building modelopt $MODELOPT_GIT_REF from source..."
MODELOPT_SRC=$(mktemp -d)/modelopt
git clone --depth=1 --branch "$MODELOPT_GIT_REF" "$MODELOPT_GIT_URL" "$MODELOPT_SRC"
sed -i 's|requires-python = ">=3.10,<3.13"|requires-python = ">=3.10,<3.14"|' "$MODELOPT_SRC/pyproject.toml"
sed -i 's|python_requires=">=3.10,<3.13"|python_requires=">=3.10,<3.14"|' "$MODELOPT_SRC/setup.py"
MODELOPT_WHEEL_DIR=$(mktemp -d)
python3 -m pip wheel --no-deps -w "$MODELOPT_WHEEL_DIR" "$MODELOPT_SRC"
python3 -m pip install --no-deps --force-reinstall "$MODELOPT_WHEEL_DIR"/*.whl
rm -rf "$MODELOPT_SRC" "$MODELOPT_WHEEL_DIR"

# Clone TRT-LLM + LFS pull + submodules
echo "Cloning TensorRT-LLM..."
git clone --depth=1 --branch "$GIT_REF" "$GIT_URL" "$BUILD_DIR"
cd "$BUILD_DIR"
echo "Fetching LFS objects (internal_cutlass_kernels archives)..."
git lfs pull
git submodule update --init --recursive --depth=1

# requirements.txt patches:
#   - relax modelopt pin to be compatible with our from-source 0.37 install
#     (upstream is `~=0.37.0` which only allows 0.37.x; we want 0.37+ to also
#     work if a newer wheel becomes available without a rebuild)
#   - remove `setuptools<80` ceiling. Modern setuptools (>=80) is required by
#     several of our other dependencies (e.g. transformer-engine build deps);
#     downgrading creates an unresolvable conflict in the venv.
sed -i 's|nvidia-modelopt\[torch\]~=0\.37\.0|nvidia-modelopt[torch]>=0.37.0|' requirements.txt
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
    --use_ccache \
    --nvrtc_dynamic_linking \
    -D "ENABLE_UCX=OFF"

# Install the freshly built wheel. --no-deps because the venv already has
# everything; --force-reinstall in case a previous run left a partial install.
echo "Installing TensorRT-LLM wheel..."
python3 -m pip install --no-deps --force-reinstall "$BUILD_DIR"/build/tensorrt_llm-*.whl

echo "Build completed successfully!"
echo "TRT-LLM source is available at: $BUILD_DIR"
echo "TRT-LLM wheel installed in: $(python3 -c 'import tensorrt_llm, os; print(os.path.dirname(tensorrt_llm.__file__))')"
