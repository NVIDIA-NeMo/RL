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

"""Generate a fingerprint for the NeMo RL codebase.

This script computes hashes for individual dependency components:
- pyproject.toml contents
- uv.lock contents
- Git submodule commit SHAs

The fingerprint is printed to stdout as JSON and can be used to detect container/code drift.
This script uses ONLY Python stdlib (no external packages) for maximum portability.

Usage:
    python tools/generate_fingerprint.py

Output:
    JSON object mapping component names to their hashes/commits
"""

import hashlib
import json
import subprocess
from pathlib import Path


def get_repo_root() -> Path:
    """Get the repository root directory relative to this script."""
    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent
    return repo_root


def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file's contents."""
    if not file_path.exists():
        return "missing"

    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def get_submodule_shas(repo_root: Path) -> dict[str, str]:
    """Get commit SHAs for all git submodules.

    Returns:
        Dictionary mapping submodule path to commit SHA
    """
    submodules = {}

    try:
        # Run git submodule status to get current commits
        result = subprocess.run(
            # Add --git-dir and --work-tree to ensure we can run git without the safe.directory check
            [
                "git",
                f"--git-dir={repo_root}/.git",
                f"--work-tree={repo_root}",
                "submodule",
                "status",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse output: " <commit> <path> (<branch>)" or "+<commit> <path> (<branch>)"
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.strip().split()
            if len(parts) >= 2:
                # Remove leading +/- indicators
                commit = parts[0].lstrip("+-")
                path = parts[1]
                submodules[path] = commit

    except subprocess.CalledProcessError:
        # If git command fails, return empty dict (e.g., not in a git repo)
        pass
    except FileNotFoundError:
        # Git not available
        pass

    return submodules


def generate_fingerprint() -> dict[str, str]:
    """Generate a fingerprint for the current codebase state.

    Returns:
        Dictionary mapping component names to their hashes/commits:
        - "pyproject.toml": MD5 hash of pyproject.toml
        - "uv.lock": MD5 hash of uv.lock
        - "submodules/<path>": Commit SHA for each submodule
    """
    repo_root = get_repo_root()

    fingerprint = {}

    # Hash pyproject.toml
    fingerprint["pyproject.toml"] = compute_file_hash(repo_root / "pyproject.toml")

    # Hash uv.lock
    fingerprint["uv.lock"] = compute_file_hash(repo_root / "uv.lock")

    # Get submodule SHAs (sorted by path for consistency)
    submodules = get_submodule_shas(repo_root)
    for path, sha in sorted(submodules.items()):
        fingerprint[f"submodules/{path}"] = sha

    return fingerprint


def main():
    """Main entry point: print fingerprint JSON to stdout."""
    fingerprint = generate_fingerprint()
    print(json.dumps(fingerprint, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
