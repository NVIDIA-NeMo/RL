#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "Usage: $0 <git-url> <commit>" >&2
    exit 2
fi
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(realpath "$SCRIPT_DIR/..")
GIT_URL=$1
GIT_REF=$2
WHEEL_OUTPUT_DIR=${WHEEL_OUTPUT_DIR:-/opt/trtllm_wheels}
BUILD_PARENT=${TRTLLM_BUILD_PARENT:-/opt/nano35-build-cache}
ARCH=${BUILD_CUSTOM_TRTLLM_ARCH:-100-real}
JOBS=${TRTLLM_BUILD_JOBS:-24}
python3 - <<'PYTARGET'
import platform
import sys

if sys.version_info[:2] != (3, 13) or platform.machine() != 'aarch64':
    raise SystemExit('This Nano SWE wheel targets CPython 3.13 on ARM64.')
PYTARGET
BUILD_KEY=$(python3 - "$GIT_URL" "$GIT_REF" "$ARCH" "$SCRIPT_DIR/trtllm-nano35.patch" <<'PYKEY'
import hashlib
import sys
from pathlib import Path
import torch

identity = '|'.join([*sys.argv[1:4], torch.__version__, str(torch.version.cuda), sys.implementation.cache_tag])
print(hashlib.sha256(identity.encode() + Path(sys.argv[4]).read_bytes()).hexdigest()[:20])
PYKEY
)
mkdir -p "$BUILD_PARENT"
BUILD_DIR="$BUILD_PARENT/$BUILD_KEY"
exec 8>"$BUILD_DIR.lock"
flock -x 8
mkdir -p "$BUILD_DIR"
git lfs version
mkdir -p "$WHEEL_OUTPUT_DIR"
exec > >(tee -a "$WHEEL_OUTPUT_DIR/build.log") 2>&1
printf 'TRT build: ref=%s, arch=%s, build_dir=%s\n' "$GIT_REF" "$ARCH" "$BUILD_DIR"

# The container rootfs is node-local. /opt survives an incomplete SQSH export,
# unlike /tmp, so a later author job can resume source and compiler state.
if [[ ! -d "$BUILD_DIR/source" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout --filter=blob:none "$GIT_URL" "$BUILD_DIR/source"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$BUILD_DIR/source" checkout --detach "$GIT_REF"
    git -C "$BUILD_DIR/source" lfs install --local
    git -C "$BUILD_DIR/source" lfs pull
    git -C "$BUILD_DIR/source" submodule update --init --recursive --depth=1
fi
[[ $(git -C "$BUILD_DIR/source" rev-parse HEAD) == "$GIT_REF" ]]
if ! git -C "$BUILD_DIR/source" apply --reverse --check "$SCRIPT_DIR/trtllm-nano35.patch" 2>/dev/null; then
    git -C "$BUILD_DIR/source" apply --check "$SCRIPT_DIR/trtllm-nano35.patch"
    git -C "$BUILD_DIR/source" apply "$SCRIPT_DIR/trtllm-nano35.patch"
fi
# Reset only the build-generated files in this script's private checkout.
git -C "$BUILD_DIR/source" restore --source="$GIT_REF" -- \
    requirements.txt constraints.txt requirements-dev.txt scripts/build_wheel.py setup.py \
    tensorrt_llm/version.py cpp/tensorrt_llm/kernels/cutlass_kernels/CMakeLists.txt

export NRL_TRT_BUILD_TOOLS="$BUILD_DIR/build-tools"
# The outer uv sync holds the actor venv lock while invoking this backend.
# Conan 2.33 supports the project's current urllib3 and distro constraints.
# An independent target leaves the locked runtime environment unchanged.
uv pip install --python "$(command -v python3)" --target "$NRL_TRT_BUILD_TOOLS" \
    build==1.6.1 conan==2.33.0 nanobind==3.1.0 \
    pybind11-stubgen==3.0.0 patchelf==0.19.1.0 meson==1.12.1
export PYTHONPATH="$NRL_TRT_BUILD_TOOLS${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$NRL_TRT_BUILD_TOOLS/bin:$PATH"
uv pip freeze --path "$NRL_TRT_BUILD_TOOLS" > "$WHEEL_OUTPUT_DIR/build-tools.txt"
"$NRL_TRT_BUILD_TOOLS/bin/conan" profile detect --force

python3 - "$REPO_ROOT" "$BUILD_DIR/source" <<'PYBUILD'
import ast
import sys
import tomllib
from pathlib import Path

repo, source = map(Path, sys.argv[1:])
metadata = tomllib.loads((repo / '3rdparty/TensorRT-LLM-workspace/pyproject.toml').read_text())
(source / 'requirements.txt').write_text('\n'.join(metadata['project']['dependencies']) + '\n')
(source / 'constraints.txt').write_text('# Constraints are included in requirements.txt.\n')
(source / 'requirements-dev.txt').write_text('-r requirements.txt\n')
# The upstream helper installs a separate dependency set; use uv's resolved environment.
p = source / 'scripts/build_wheel.py'
s = p.read_text()
tree = ast.parse(s)
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'setup_venv')
lines = s.splitlines(keepends=True)
replacement = '''def setup_venv(project_dir, requirements_file, no_venv, yes=False, build_root=None):
    python = Path(sys.executable)
    conan = Path(os.environ["NRL_TRT_BUILD_TOOLS"]) / "bin" / "conan"
    if not conan.is_file():
        raise RuntimeError("The preinstalled build environment is missing conan")
    print(f"-- Using resolved NeMo-RL build environment: {python}")
    return python, conan
'''
s = ''.join(lines[:node.lineno - 1]) + replacement + ''.join(lines[node.end_lineno:])
s = s.replace('-m pip install nanobind', '-m pip show nanobind')
s = s.replace('-m pip install pybind11-stubgen', '-m pip show pybind11-stubgen')
p.write_text(s)
# CUTLASS is a build-time Python source dependency. Upstream's `develop --user`
# invokes pip inside a venv and fails because user site-packages are disabled.
p = source / 'cpp/tensorrt_llm/kernels/cutlass_kernels/CMakeLists.txt'
s = p.read_text()
start = s.index('execute_process(', s.index('set(cutlass_source_dir'))
end = s.index('set(GENERATE_KERNELS_SCRIPT_DIR', start)
s = s[:start] + '''if(NOT EXISTS "${cutlass_source_dir}/python/cutlass_library/__init__.py")
  message(FATAL_ERROR "The pinned CUTLASS Python library is missing")
endif()
set(ENV{PYTHONPATH} "${cutlass_source_dir}/python:$ENV{PYTHONPATH}")

''' + s[end:]
p.write_text(s)
# --skip-stubs intentionally omits the stub directory; require the compiled
# extension for this exact interpreter instead of using that directory as proof.
p = source / 'setup.py'
s = p.read_text()
old_check = 'if not (tensorrt_llm_path / "bindings").exists():'
assert old_check in s
s = s.replace('import os\n', 'from importlib.machinery import EXTENSION_SUFFIXES\n\nimport os\n', 1)
s = s.replace(old_check, '''if not any(
        (tensorrt_llm_path / f"bindings{suffix}").is_file()
        for suffix in EXTENSION_SUFFIXES
    ):''', 1)
p.write_text(s)
(source / 'NANO35_BUILD_VERSION').write_text(metadata['project']['version'])
PYBUILD

cd "$BUILD_DIR/source"
VERSION=$(cat NANO35_BUILD_VERSION)
export CCACHE_DIR=${CCACHE_DIR:-$BUILD_DIR/ccache}
export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-20G}
export MAX_JOBS=$JOBS
export CMAKE_BUILD_PARALLEL_LEVEL=$JOBS
export NINJA_STATUS='Ninja progress: %f/%t (%p) '
python3 scripts/build_wheel.py \
    -a "$ARCH" -G Ninja --use_ccache --nvrtc_dynamic_linking \
    --job_count "$JOBS" --build_root "$BUILD_DIR/native" \
    --dist_dir "$BUILD_DIR/wheels" --version-override "$VERSION" \
    --configure_cmake --skip-stubs -D ENABLE_UCX=OFF

python3 - "$BUILD_DIR/wheels" "$WHEEL_OUTPUT_DIR" "$VERSION" <<'PYCOPY'
import email
import shutil
import sys
import zipfile
from pathlib import Path

src, dst = map(Path, sys.argv[1:3])
version = sys.argv[3]
wheels = list(src.glob('tensorrt_llm-*.whl'))
if len(wheels) != 1:
    raise RuntimeError(f'Expected one wheel, found {wheels}')
wheel = wheels[0]
if '-cp313-' not in wheel.name or 'aarch64' not in wheel.name:
    raise RuntimeError(f'Unexpected target ABI: {wheel.name}')
with zipfile.ZipFile(wheel) as z:
    meta_name = next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))
    meta = email.message_from_bytes(z.read(meta_name))
    if meta['Version'] != version:
        raise RuntimeError(f'Wheel version mismatch: {meta["Version"]} != {version}')
tmp = dst / (wheel.name + '.partial')
shutil.copy2(wheel, tmp)
tmp.replace(dst / wheel.name)
print(f'Wheel ready: {dst / wheel.name}')
PYCOPY
# Only remove this job's new temporary tree after publishing a validated wheel.
rm -rf -- "$BUILD_DIR"
