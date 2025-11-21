#!/usr/bin/env python3
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

"""
Sync CACHED_DEPENDENCIES in 3rdparty/*/setup.py files from their corresponding submodule pyproject.toml files.

This script reads dependencies from submodule pyproject.toml files and updates the CACHED_DEPENDENCIES
list in the wrapper setup.py files to keep them in sync.
"""

import re
import sys
import tomllib
from pathlib import Path
from typing import List, Tuple


def get_repo_root() -> Path:
    """Get the repository root directory."""
    script_path = Path(__file__).resolve()
    # Script is in .github/scripts/, so go up 2 levels
    return script_path.parent.parent.parent


def read_dependencies_from_pyproject(pyproject_path: Path) -> List[str]:
    """Read dependencies from a pyproject.toml file."""
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")
    
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    
    if "project" not in data or "dependencies" not in data["project"]:
        raise ValueError(f"No [project].dependencies found in {pyproject_path}")
    
    return [str(dep).strip() for dep in data["project"]["dependencies"]]


def update_cached_dependencies(setup_py_path: Path, new_dependencies: List[str]) -> bool:
    """
    Update CACHED_DEPENDENCIES list in a setup.py file.
    
    Returns True if changes were made, False otherwise.
    """
    if not setup_py_path.exists():
        raise FileNotFoundError(f"setup.py not found at {setup_py_path}")
    
    content = setup_py_path.read_text()
    
    # Find CACHED_DEPENDENCIES list using regex
    # Pattern matches: CACHED_DEPENDENCIES = [ ... ]
    pattern = r'(CACHED_DEPENDENCIES\s*=\s*\[)(.*?)(\])'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        raise ValueError(f"CACHED_DEPENDENCIES not found in {setup_py_path}")
    
    # Build new dependencies list with proper formatting
    indent = "    "
    formatted_deps = []
    for dep in new_dependencies:
        formatted_deps.append(f'{indent}"{dep}",')
    
    new_deps_str = "\n" + "\n".join(formatted_deps) + "\n"
    
    # Replace the content between brackets
    new_content = content[:match.start(2)] + new_deps_str + content[match.end(2):]
    
    # Check if content changed
    if new_content == content:
        return False
    
    setup_py_path.write_text(new_content)
    return True


def sync_megatron_bridge() -> Tuple[bool, str]:
    """Sync Megatron-Bridge dependencies."""
    repo_root = get_repo_root()
    pyproject_path = repo_root / "3rdparty" / "Megatron-Bridge-workspace" / "Megatron-Bridge" / "pyproject.toml"
    setup_py_path = repo_root / "3rdparty" / "Megatron-Bridge-workspace" / "setup.py"
    
    try:
        dependencies = read_dependencies_from_pyproject(pyproject_path)
        changed = update_cached_dependencies(setup_py_path, dependencies)
        status = "updated" if changed else "unchanged"
        return True, f"Megatron-Bridge: {status}"
    except Exception as e:
        return False, f"Megatron-Bridge: ERROR - {e}"


def sync_penguin() -> Tuple[bool, str]:
    """Sync Penguin dependencies."""
    repo_root = get_repo_root()
    
    # Penguin submodule is at Penguin-workspace/Penguin/pyproject.toml
    # but the directory structure shows it doesn't exist yet, so we check for it
    penguin_pyproject = repo_root / "3rdparty" / "Penguin-workspace" / "Penguin" / "pyproject.toml"
    setup_py_path = repo_root / "3rdparty" / "Penguin-workspace" / "setup.py"
    
    # Check if Penguin submodule exists
    if not penguin_pyproject.exists():
        return True, "Penguin: skipped (submodule not initialized)"
    
    try:
        dependencies = read_dependencies_from_pyproject(penguin_pyproject)
        changed = update_cached_dependencies(setup_py_path, dependencies)
        status = "updated" if changed else "unchanged"
        return True, f"Penguin: {status}"
    except Exception as e:
        return False, f"Penguin: ERROR - {e}"


def sync_megatron_lm() -> Tuple[bool, str]:
    """
    Sync Megatron-LM dependencies.
    
    Note: Megatron-LM has hardcoded requirements in setup.py, but we should verify
    they match the submodule's requirements files.
    """
    repo_root = get_repo_root()
    
    # Megatron-LM doesn't have a simple pyproject.toml with dependencies in the same format
    # It has requirements files in various places. For now, we'll just report status.
    megatron_lm_dir = repo_root / "3rdparty" / "Megatron-LM-workspace" / "Megatron-LM"
    
    if not megatron_lm_dir.exists():
        return True, "Megatron-LM: skipped (submodule not initialized)"
    
    # The setup.py for Megatron-LM has hardcoded dependencies that are manually curated
    # from requirements files. We'll just report it as a manual check.
    return True, "Megatron-LM: manual check required (uses hardcoded requirements)"


def main():
    """Main function to sync all submodule dependencies."""
    print("Syncing submodule dependencies to 3rdparty setup.py files...")
    print()
    
    results = []
    all_success = True
    
    # Sync each submodule
    for sync_func in [sync_megatron_bridge, sync_penguin, sync_megatron_lm]:
        success, message = sync_func()
        results.append(message)
        if not success:
            all_success = False
    
    # Print results
    for result in results:
        print(f"  {result}")
    
    print()
    if all_success:
        print("✓ All submodule dependencies synced successfully")
        return 0
    else:
        print("✗ Some submodule dependencies failed to sync")
        return 1


if __name__ == "__main__":
    sys.exit(main())

