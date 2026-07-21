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

# Assert a `sed -i` patch target exists before patching. `sed` exits 0 even when
# the pattern doesn't match, so if TRT-LLM upstream changes these files our
# patches would silently no-op and the build would fail later in an obscure way.
# Fail loudly here instead. Args: <file> <fixed-string-pattern>.
assert_patch_target() {
    grep -qF -- "$2" "$1" || {
        echo "[ERROR] Expected patch target not found in $1 (did TRT-LLM upstream change?):" >&2
        echo "          $2" >&2
        exit 1
    }
}

# Required positional arguments: the fork URL and commit ref to build.
# This is purely a build script — it does not resolve defaults. The caller owns
# the source of truth (the PEP 517 backend _backend.py reads them from the
# [tool.trtllm] table of the workspace pyproject.toml and passes them here).
if [[ $# -lt 2 || -z "${1:-}" || -z "${2:-}" ]]; then
    echo "[ERROR] Usage: $0 <GIT_URL> <GIT_REF>" >&2
    echo "        Both the TensorRT-LLM fork URL and commit ref are required." >&2
    exit 1
fi
GIT_URL=$1
GIT_REF=$2

BUILD_DIR=$(realpath "$SCRIPT_DIR/../3rdparty")/TensorRT-LLM
if [[ -e "$BUILD_DIR" ]]; then
  echo "[ERROR] $BUILD_DIR already exists. Please remove or move it before running this script."
  exit 1
fi

# Directory where the built wheel is written. Set by _backend.py when called
# via uv; defaults to /opt/trtllm_wheels when run standalone.
WHEEL_OUTPUT_DIR=${WHEEL_OUTPUT_DIR:-/opt/trtllm_wheels}
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
# `--branch` only accepts branch/tag names, not commit hashes.
# Use init + fetch --depth=1 <hash> to get a shallow clone at a specific commit.
echo "Cloning TensorRT-LLM..."
git init "$BUILD_DIR"
git -C "$BUILD_DIR" remote add origin "$GIT_URL"
git -C "$BUILD_DIR" fetch --depth=1 origin "$GIT_REF"
git -C "$BUILD_DIR" checkout FETCH_HEAD
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
assert_patch_target requirements.txt 'nvidia-modelopt[torch]~=0.37.0'
sed -i 's|nvidia-modelopt\[torch\]~=0\.37\.0|nvidia-modelopt[torch]>=0.44.0a0|' requirements.txt
assert_patch_target requirements.txt 'setuptools<80'
sed -i 's|^setuptools<80$|setuptools|' requirements.txt

# cutlass_kernels/CMakeLists.txt invokes `setup_library.py develop --user`,
# which (a) requires a setup.py shim and (b) the `--user` flag is invalid
# inside a venv. Rewrite the COMMAND to copy setup_library.py to setup.py
# (so `develop` finds a buildable target) and drop `--user`.
assert_patch_target cpp/tensorrt_llm/kernels/cutlass_kernels/CMakeLists.txt \
    'COMMAND ${Python3_EXECUTABLE} setup_library.py develop --user'
sed -i 's|COMMAND \${Python3_EXECUTABLE} setup_library.py develop --user|COMMAND bash -c "cp -f setup_library.py setup.py \&\& \${Python3_EXECUTABLE} setup_library.py develop"|' \
    cpp/tensorrt_llm/kernels/cutlass_kernels/CMakeLists.txt

# SM arch list. Sourced from BUILD_CUSTOM_TRTLLM_ARCH so it stays in sync with
# _backend.py, which folds the same value into the wheel cache key (a change to
# the arch list must invalidate the cached wheel). The default below MUST match
# _backend.py's _DEFAULT_ARCH.
#   90-real;100-real: build Hopper (sm_90) and Blackwell (sm_100) kernels,
#                     i.e. H100/GB200/B200 only (B200 is also sm_100) — other
#                     SKUs (e.g. A100 sm_80, L40 sm_89, consumer Blackwell
#                     RTX 50-series sm_120) need this list extended.
ARCH="${BUILD_CUSTOM_TRTLLM_ARCH:-90-real;100-real}"
JOBS="${TRTLLM_BUILD_JOBS:-24}"
NPROC=$(nproc 2>/dev/null || echo "$JOBS")
if (( JOBS > NPROC )); then
    JOBS=$NPROC
fi

# Build the wheel.
#   --job_count: parallel compile jobs (see JOBS above).
#   --nvrtc_dynamic_linking: required so the wheel links against the venv's
#                            libnvrtc-builtins lazily instead of statically.
echo "Building TensorRT-LLM wheel (arch=${ARCH}, jobs=${JOBS})..."
# Bracket the build with ccache stats so CI logs surface the cache hit rate.
# Builds run cold unless vars.TRTLLM_BUILD_CACHE is set, so this makes a
# missing/misconfigured cache obvious instead of a silent full rebuild.
# `|| true` keeps it non-fatal when ccache isn't on PATH.
ccache --zero-stats >/dev/null 2>&1 || true
python3 scripts/build_wheel.py \
    -a "$ARCH" \
    -G Ninja \
    --clean \
    --use_ccache \
    --nvrtc_dynamic_linking \
    --job_count "$JOBS" \
    -D "ENABLE_UCX=OFF"
ccache --show-stats || true

echo "Copying TensorRT-LLM wheel to ${WHEEL_OUTPUT_DIR}..."
cp "$BUILD_DIR"/build/tensorrt_llm-*.whl "$WHEEL_OUTPUT_DIR/"

# Remove source tree and build artifacts to reclaim disk space.
echo "Cleaning up TensorRT-LLM source and build artifacts..."
rm -rf "$BUILD_DIR"

echo "Build completed successfully!"
echo "TRT-LLM wheel copied to: ${WHEEL_OUTPUT_DIR}"
