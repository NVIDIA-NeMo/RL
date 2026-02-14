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
GIT_URL=${1:-https://github.com/flashinfer-ai/flashinfer}
GIT_REF=${2:-main}

BUILD_DIR=$(realpath "$SCRIPT_DIR/../3rdparty/flashinfer")
if [[ -e "$BUILD_DIR" ]]; then
  echo "[ERROR] $BUILD_DIR already exists. Please remove or move it before running this script."
  exit 1 
fi

echo "Building FlashInfer from:"
echo "  FlashInfer Git URL: $GIT_URL"
echo "  FlashInfer Git ref: $GIT_REF"

# Clone the repository
echo "Cloning repository..."
# When running inside Docker with --mount=type=ssh, the known_hosts file is empty.
# Skip host key verification for internal builds (only applies to SSH URLs).
GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" git clone --recursive "$GIT_URL" "$BUILD_DIR"
cd "$BUILD_DIR"
git checkout "$GIT_REF"
git submodule update

PYPROJECT_TOML="$REPO_ROOT/pyproject.toml"
if [[ ! -f "$PYPROJECT_TOML" ]]; then
  echo "[ERROR] pyproject.toml not found at $PYPROJECT_TOML. This script must be run from the repo root and pyproject.toml must exist."
  exit 1
fi

cd "$REPO_ROOT"

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

# 1) Add [tool.uv.sources].flashinfer-python = { path = "3rdparty/flashinfer", editable = true }
tool = doc.setdefault("tool", {})
uv = tool.setdefault("uv", {})
sources = uv.setdefault("sources", {})
desired = inline_table()
desired.update({"path": "3rdparty/flashinfer", "editable": True})
sources["flashinfer-python"] = desired

# 2) Add flashinfer-python to [project.optional-dependencies].vllm
project = doc.get("project")
if project is None:
    raise SystemExit("[ERROR] Missing [project] in pyproject.toml")

opt = project.get("optional-dependencies")
vllm_list = opt["vllm"]
if not vllm_list:
    vllm_list = []
if "flashinfer-python" not in vllm_list:
    vllm_list.append("flashinfer-python")
opt["vllm"] = vllm_list

pyproject_path.write_text(dumps(doc))
print("[INFO] Updated pyproject.toml for local FlashInfer.")
PY

# Ensure build deps and re-lock
uv pip install setuptools_scm
uv lock

cat <<EOF
[INFO] pyproject.toml updated. NeMo RL is now configured to use the local FlashInfer at 3rdparty/flashinfer.
EOF
