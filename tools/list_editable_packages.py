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

"""List editable packages for all python executables in PATH.

This utility helps users identify which packages can be mounted for development.
It searches for all python* executables in PATH and lists their editable installs.
"""

import json
import os
import re
import subprocess
import sys


def find_python_executables():
    """Find all python* executables in PATH.

    Returns:
        List of (name, path) tuples for python executables
    """
    # Pattern to match:
    # - python (exact match, as representative of driver script's python)
    # - python-* wrapper scripts (like python-AsyncTrajectoryCollector, python-DTensorPolicyWorker, etc.)
    # Excludes python3, python3.12, etc. and argcomplete-related scripts
    python_pattern = re.compile(r"^python$|^python-(?!.*argcomplete).*$")

    executables = []
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)

    seen = set()
    for path_dir in path_dirs:
        if not path_dir or not os.path.isdir(path_dir):
            continue

        try:
            for entry in os.listdir(path_dir):
                # Filter by pattern (includes argcomplete exclusion via negative lookahead)
                if python_pattern.match(entry) and entry not in seen:
                    full_path = os.path.join(path_dir, entry)
                    if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                        # Verify it's actually a python executable
                        try:
                            result = subprocess.run(
                                [full_path, "--version"],
                                capture_output=True,
                                timeout=2,
                            )
                            if result.returncode == 0:
                                executables.append((entry, full_path))
                                seen.add(entry)
                        except (subprocess.TimeoutExpired, FileNotFoundError):
                            pass
        except (PermissionError, OSError):
            continue

    # Sort by name for consistent output
    executables.sort(key=lambda x: x[0])
    return executables


def get_editable_packages(python_exe):
    """Get list of editable packages for a python executable.

    Args:
        python_exe: Path to python executable

    Returns:
        List of (package_name, location) tuples for editable packages
    """
    try:
        result = subprocess.run(
            [python_exe, "-m", "pip", "list", "--format=json", "--editable"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return None

        packages = json.loads(result.stdout)
        editable_packages = []

        for pkg in packages:
            # Get more details about the package location
            show_result = subprocess.run(
                [python_exe, "-m", "pip", "show", pkg["name"]],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if show_result.returncode == 0:
                location = None
                editable_location = None

                for line in show_result.stdout.split("\n"):
                    if line.startswith("Location:"):
                        location = line.split(":", 1)[1].strip()
                    elif line.startswith("Editable project location:"):
                        editable_location = line.split(":", 1)[1].strip()

                # Prefer editable location if available
                final_location = editable_location or location
                if final_location:
                    editable_packages.append((pkg["name"], final_location))

        return editable_packages

    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ):
        return None


def main():
    """Main entry point: list editable packages for all python executables."""
    print("Searching for python executables in PATH...")
    print()

    executables = find_python_executables()

    if not executables:
        print("No python executables found in PATH.")
        return 1

    found_any_editable = False

    for name, path in executables:
        editable_packages = get_editable_packages(path)

        if editable_packages is None:
            continue  # Skip executables where pip list failed

        if not editable_packages:
            continue  # Skip executables with no editable packages

        found_any_editable = True
        print(f"{name}:")
        for pkg_name, pkg_location in sorted(editable_packages):
            print(f"  - {pkg_name}: {pkg_location}")
        print()

    if not found_any_editable:
        print("No editable packages found in any python executable.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
