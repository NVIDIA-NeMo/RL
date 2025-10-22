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

set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR/..")"


# Parse command line arguments
GIT_URL=${1:-https://github.com/terrykong/vllm.git}
GIT_REF=${2:-terryk/demo-custom-vllm}
# NOTE: VLLM_USE_PRECOMPILED=1 didn't always seem to work since the wheels were sometimes built against an incompatible torch/cuda combo.
# This commit was chosen as one close to the v0.10 release: git merge-base --fork-point origin/main tags/v0.10.0
VLLM_WHEEL_COMMIT=${3:-d8ee5a2ca4c73f2ce5fdc386ce5b4ef3b6e6ae70}  # use full commit hash from the main branch
export VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/${VLLM_WHEEL_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"

BUILD_DIR=$(realpath "$SCRIPT_DIR/../3rdparty/vllm")
if [[ -e "$BUILD_DIR" ]]; then
  echo "[ERROR] $BUILD_DIR already exists. Please remove or move it before running this script."
  exit 1 
fi

echo "Building vLLM from:"
echo "  Vllm Git URL: $GIT_URL"
echo "  Vllm Git ref: $GIT_REF"
echo "  Vllm Wheel commit: $VLLM_WHEEL_COMMIT"
echo "  Vllm Wheel location: $VLLM_PRECOMPILED_WHEEL_LOCATION"

# Clone the repository
echo "Cloning repository..."
git clone "$GIT_URL" "$BUILD_DIR"
cd "$BUILD_DIR"
git checkout "$GIT_REF"

# Create a new Python environment using uv
echo "Creating Python environment..."
# Pop the project environment set by user to not interfere with the one we create for the vllm repo
OLD_UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT
unset UV_PROJECT_ENVIRONMENT
uv venv

# Remove all comments from requirements files to prevent use_existing_torch.py from incorrectly removing xformers
echo "Removing comments from requirements files..."
find requirements/ -name "*.txt" -type f -exec sed -i 's/#.*$//' {} \; 2>/dev/null || true
find requirements/ -name "*.txt" -type f -exec sed -i '/^[[:space:]]*$/d' {} \; 2>/dev/null || true
# Replace xformers==.* (but preserve any platform markers at the end)
# NOTE: that xformers is bumped from 0.0.30 to 0.0.31 to work with torch==2.7.1. This version may need to change to change when we upgrade torch.
find requirements/ -name "*.txt" -type f -exec sed -i -E 's/^(xformers)==[^;[:space:]]*/\1==0.0.31/' {} \; 2>/dev/null || true

uv run --no-project use_existing_torch.py

# Install dependencies
echo "Installing dependencies..."
uv pip install --upgrade pip
uv pip install numpy setuptools setuptools_scm
uv pip install torch==2.7.1 --torch-backend=cu128

# Install vLLM using precompiled wheel
echo "Installing vLLM with precompiled wheel..."
uv pip install --no-build-isolation -e .

echo "Build completed successfully!"
echo "The built vLLM is available in: $BUILD_DIR"

echo "Updating repo pyproject.toml to point vLLM to local clone..."

PYPROJECT_TOML="$REPO_ROOT/pyproject.toml"
if [[ ! -f "$PYPROJECT_TOML" ]]; then
  echo "[ERROR] pyproject.toml not found at $PYPROJECT_TOML. This script must be run from the repo root and pyproject.toml must exist."
  exit 1
fi

cd "$REPO_ROOT"

export UV_PROJECT_ENVIRONMENT=$OLD_UV_PROJECT_ENVIRONMENT
if [[ -n "$UV_PROJECT_ENVIRONMENT" ]]; then
    # We optionally set this if the project environment is outside of the project directory.
    # If we do not set this then uv pip install commands will fail
    export VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT
fi
# Use tomlkit via uv to idempotently update pyproject.toml
uv run --no-project --with tomlkit python - <<'PY'
from pathlib import Path
from tomlkit import parse, dumps, inline_table

pyproject_path = Path("pyproject.toml")
text = pyproject_path.read_text()
doc = parse(text)

# 1) Ensure setuptools_scm in [project].dependencies
project = doc.get("project")
if project is None:
    raise SystemExit("[ERROR] Missing [project] in pyproject.toml")

deps = project.get("dependencies")

if not any(x.startswith("setuptools_scm") for x in deps):
    deps.append("setuptools_scm")

# 2) Update [project.optional-dependencies].vllm: unpin vllm==... -> vllm
opt = project.get("optional-dependencies")
vllm_list = opt["vllm"]
# Remove any pinned vllm==...
keep_items = []
has_unpinned_vllm = False
for item in vllm_list:
    s = str(item).strip()
    if s.startswith("vllm=="):
        continue
    if s == "vllm":
        has_unpinned_vllm = True
    keep_items.append(item)
if not has_unpinned_vllm:
    keep_items.append("vllm")
vllm_list.clear()
for it in keep_items:
    vllm_list.append(it)

# 3) Add [tool.uv.sources].vllm = { path = "3rdparty/vllm", editable = true }
tool = doc.setdefault("tool", {})
uv = tool.setdefault("uv", {})
sources = uv.setdefault("sources", {})
desired = inline_table()
desired.update({"path": "3rdparty/vllm", "editable": True})
sources["vllm"] = desired

# 4) Ensure [tool.uv].no-build-isolation-package includes "vllm"
nbip = uv.setdefault("no-build-isolation-package", [])
nbip_strs = [str(x) for x in nbip]
if "vllm" not in nbip_strs:
    nbip.append("vllm")

pyproject_path.write_text(dumps(doc))
print("[INFO] Updated pyproject.toml for local vLLM.")
PY

# Ensure build deps and re-lock
uv pip install setuptools_scm
uv lock

# Write to a file that a docker build will use to set the necessary env vars
cat <<EOF >$BUILD_DIR/nemo-rl.env
export VLLM_GIT_REF=$GIT_REF
export VLLM_PRECOMPILED_WHEEL_LOCATION=$VLLM_PRECOMPILED_WHEEL_LOCATION
EOF

cat <<EOF
[INFO] pyproject.toml updated. NeMo RL is now configured to use the local vLLM at 3rdparty/vllm.
[INFO] Verify this new vllm version by running:

VLLM_PRECOMPILED_WHEEL_LOCATION=$VLLM_PRECOMPILED_WHEEL_LOCATION \\
  uv run --extra vllm vllm serve Qwen/Qwen3-0.6B

[INFO] For more information on this custom install, visit https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/use-custom-vllm.md
[IMPORTANT] Remember to set the shell variable 'VLLM_PRECOMPILED_WHEEL_LOCATION' when running NeMo RL apps with this custom vLLM to avoid re-compiling.
EOF
