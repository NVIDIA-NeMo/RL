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
import sys

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


if __name__ == "__main__":
    prefetch_venvs()
