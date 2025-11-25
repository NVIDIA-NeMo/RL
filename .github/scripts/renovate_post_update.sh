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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR/../..")"

echo "==================================="
echo "Renovate Post-Update Script"
echo "==================================="
echo ""

cd "$REPO_ROOT"

# Step 1: Sync submodule dependencies to setup.py files
echo "Step 1: Syncing submodule dependencies..."
python3 "$SCRIPT_DIR/sync_submodule_dependencies.py"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to sync submodule dependencies"
    exit 1
fi
echo ""

# Step 2: Run uv lock to regenerate lock file
echo "Step 2: Running uv lock..."

# Install uv if not available (needed when running inside Renovate's Docker container)
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed or not in PATH after installation attempt"
    exit 1
fi

# Run uv lock to regenerate the lock file
# Note: Workspace members are defined in pyproject.toml [tool.uv.workspace].members
# Some members point to submodule paths (e.g., Automodel-workspace/Automodel), not the parent dir
uv lock
if [ $? -ne 0 ]; then
    echo "ERROR: uv lock failed"
    exit 1
fi
echo ""

# Step 3: Stage all changes for commit
echo "Step 3: Staging changes..."
git add -A 3rdparty/*/setup.py uv.lock
if [ $? -ne 0 ]; then
    echo "WARNING: Failed to stage files with git add"
    # Don't exit, as Renovate might handle git operations differently
fi
echo ""

echo "==================================="
echo "Post-update completed successfully"
echo "==================================="
echo ""
echo "Changed files:"
git diff --cached --name-only || git status --short || echo "(git status unavailable)"

