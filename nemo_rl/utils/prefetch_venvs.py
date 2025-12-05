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
import os
import sys
from pathlib import Path

from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.utils.venvs import create_local_venv


def prefetch_venvs():
    """Prefetch all virtual environments that will be used by workers."""
    print("Prefetching virtual environments...")

    # Group venvs by py_executable to avoid duplicating work
    venv_configs = {}
    for actor_fqn, py_executable in ACTOR_ENVIRONMENT_REGISTRY.items():
        # Skip system python as it doesn't need a venv
        if py_executable == "python" or py_executable == sys.executable:
            print(f"Skipping {actor_fqn} (uses system Python)")
            continue

        # Only create venvs for uv-based executables
        if py_executable.startswith("uv"):
            if py_executable not in venv_configs:
                venv_configs[py_executable] = []
            venv_configs[py_executable].append(actor_fqn)

    # Create venvs
    for py_executable, actor_fqns in venv_configs.items():
        print(f"\nCreating venvs for py_executable: {py_executable}")
        for actor_fqn in actor_fqns:
            print(f"  Creating venv for: {actor_fqn}")
            try:
                python_path = create_local_venv(py_executable, actor_fqn)
                print(f"    Success: {python_path}")
            except Exception as e:
                print(f"    Error: {e}")
                # Continue with other venvs even if one fails
                continue

    print("\nVenv prefetching complete!")

    # Create convenience python wrapper scripts for frozen environment support (container-only)
    create_frozen_environment_symlinks(venv_configs)


def create_frozen_environment_symlinks(venv_configs):
    """Create python-{ClassName} wrapper scripts in /usr/local/bin for frozen environment support.

    Only runs in container (when NRL_CONTAINER=1 is set).

    Args:
        venv_configs: Dictionary mapping py_executable to list of actor FQNs
    """
    # Only create wrapper scripts in container
    if not os.environ.get("NRL_CONTAINER"):
        print(
            "\nSkipping frozen environment wrapper script creation (not in container)"
        )
        return

    print("\nCreating frozen environment wrapper scripts...")

    # Collect all wrapper mappings: class_name -> venv_path
    wrapper_mappings = {}

    for py_executable, actor_fqns in venv_configs.items():
        for actor_fqn in actor_fqns:
            # Extract class name from FQN (last part)
            # e.g., "nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker" -> "MegatronPolicyWorker"
            class_name = actor_fqn.split(".")[-1]

            # Get the venv path that was created
            try:
                python_path = create_local_venv(py_executable, actor_fqn)

                # Check for collisions
                if class_name in wrapper_mappings:
                    existing_path = wrapper_mappings[class_name]
                    if existing_path != python_path:
                        raise RuntimeError(
                            f"Collision detected: Multiple venvs want to use name '{class_name}'\n"
                            f"  Existing: {existing_path}\n"
                            f"  New: {python_path}\n"
                            f"This indicates two different worker classes have the same name."
                        )
                else:
                    wrapper_mappings[class_name] = python_path
            except Exception as e:
                print(f"  Warning: Could not get venv path for {actor_fqn}: {e}")
                continue

    # Create wrapper scripts
    wrapper_dir = Path("/usr/local/bin")
    created_wrappers = []

    for class_name, python_path in sorted(wrapper_mappings.items()):
        wrapper_name = f"python-{class_name}"
        wrapper_path = wrapper_dir / wrapper_name

        # Get the venv directory path (parent of bin/python)
        venv_path = Path(python_path).parent.parent

        # Create wrapper script content
        wrapper_content = f"""#!/bin/bash
VENV_PATH="{venv_path}"
export VIRTUAL_ENV="$VENV_PATH"
export PATH="$VENV_PATH/bin:$PATH"
exec "$VENV_PATH/bin/python" "$@"
"""

        try:
            # Remove existing wrapper if present
            if wrapper_path.exists() or wrapper_path.is_symlink():
                wrapper_path.unlink()

            # Write wrapper script
            wrapper_path.write_text(wrapper_content)

            # Make executable
            wrapper_path.chmod(0o755)

            created_wrappers.append(wrapper_name)
            print(f"  Created: {wrapper_name} -> {python_path}")
        except Exception as e:
            print(f"  Warning: Could not create wrapper script {wrapper_name}: {e}")
            continue

    if created_wrappers:
        print(f"\nCreated {len(created_wrappers)} frozen environment wrapper scripts")
        print("Users can now use these python executables directly:")
        for name in created_wrappers:
            print(f"  - {name}")
    else:
        print("\nNo frozen environment wrapper scripts were created")


if __name__ == "__main__":
    prefetch_venvs()
