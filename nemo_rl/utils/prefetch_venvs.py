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
import argparse
import sys

from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.utils.venvs import create_local_venv


def prefetch_venvs(filters=None):
    """Prefetch all virtual environments that will be used by workers.

    Args:
        filters: List of strings to match against actor FQNs. If provided, only
                actors whose FQN contains at least one of the filter strings will
                be prefetched. If None, all venvs are prefetched.
    """
    print("Prefetching virtual environments...")
    if filters:
        print(f"Filtering for: {filters}")

    # Group venvs by py_executable to avoid duplicating work
    venv_configs = {}
    for actor_fqn, py_executable in ACTOR_ENVIRONMENT_REGISTRY.items():
        # Apply filters if provided
        if filters and not any(f in actor_fqn for f in filters):
            continue
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
    parser = argparse.ArgumentParser(
        description="Prefetch virtual environments for Ray actors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prefetch all venvs
  python -m nemo_rl.utils.prefetch_venvs

  # Prefetch only vLLM-related venvs
  python -m nemo_rl.utils.prefetch_venvs vllm

  # Prefetch multiple specific venvs
  python -m nemo_rl.utils.prefetch_venvs vllm policy environment
        """,
    )
    parser.add_argument(
        "filters",
        nargs="*",
        help="Filter strings to match against actor FQNs. Only actors whose FQN "
        "contains at least one of these strings will be prefetched. "
        "If not provided, all venvs are prefetched.",
    )
    args = parser.parse_args()

    prefetch_venvs(filters=args.filters if args.filters else None)
