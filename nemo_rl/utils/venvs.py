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
import logging
import os
import shlex
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import ray
from ray.util import placement_group

dir_path = os.path.dirname(os.path.abspath(__file__))
git_root = os.path.abspath(os.path.join(dir_path, "../.."))
DEFAULT_VENV_DIR = os.path.join(git_root, "venvs")

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def create_local_venv(
    py_executable: str, venv_name: str, force_rebuild: bool = False
) -> str:
    """Create a virtual environment using uv and execute a command within it.

    The output can be used as a py_executable for a Ray worker assuming the worker
    nodes also have access to the same file system as the head node.

    This function is cached to avoid multiple calls to uv to create the same venv,
    which avoids duplicate logging.

    Args:
        py_executable (str): Command to run with the virtual environment (e.g., "uv.sh run --locked")
        venv_name (str): Name of the virtual environment (e.g., "foobar.Worker")
        force_rebuild (bool): If True, force rebuild the venv even if it already exists

    Returns:
        str: Path to the python executable in the created virtual environment
    """
    # This directory is where virtual environments will be installed
    # It is local to the driver process but should be visible to all worker nodes
    # If this directory is not accessible from worker nodes (e.g., on a distributed
    # cluster with non-shared filesystems), you may encounter errors when workers
    # try to access the virtual environments
    #
    # You can override this location by setting the NEMO_RL_VENV_DIR environment variable

    NEMO_RL_VENV_DIR = os.path.normpath(
        os.environ.get("NEMO_RL_VENV_DIR", DEFAULT_VENV_DIR)
    )
    logger.info(f"NEMO_RL_VENV_DIR is set to {NEMO_RL_VENV_DIR}.")

    # Create the venv directory if it doesn't exist
    os.makedirs(NEMO_RL_VENV_DIR, exist_ok=True)

    # Full path to the virtual environment
    venv_path = os.path.join(NEMO_RL_VENV_DIR, venv_name)

    # Force rebuild if requested
    if force_rebuild and os.path.exists(venv_path):
        logger.info(f"Force rebuilding venv at {venv_path}")
        import shutil

        shutil.rmtree(venv_path)

    logger.info(f"Creating new venv at {venv_path}")

    # Create the virtual environment
    uv_venv_cmd = ["uv", "venv", "--allow-existing", venv_path]
    subprocess.run(uv_venv_cmd, check=True)

    # Execute the command with the virtual environment
    env = os.environ.copy()
    # NOTE: UV_PROJECT_ENVIRONMENT is appropriate here only b/c there should only be
    #  one call to this in the driver. It is not safe to use this in a multi-process
    #  context.
    #  https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path
    env["UV_PROJECT_ENVIRONMENT"] = venv_path

    # Split the py_executable into command and arguments
    exec_cmd = shlex.split(py_executable)
    # Command doesn't matter, since `uv` syncs the environment no matter the command.
    exec_cmd.extend(["echo", f"Finished creating venv {venv_path}"])

    # Always run uv sync first to ensure the build requirements are set (for --no-build-isolation packages)
    subprocess.run(["uv", "sync"], env=env, check=True)
    subprocess.run(exec_cmd, env=env, check=True)

    # Return the path to the python executable in the virtual environment
    python_path = os.path.join(venv_path, "bin", "python")
    return python_path


# Ray-based helper to create a virtual environment on each Ray node
@ray.remote(num_cpus=1)
def _env_builder(
    py_executable: str, venv_name: str, node_idx: int, force_rebuild: bool = False
):
    # Check if another node is already building
    NEMO_RL_VENV_DIR = os.path.normpath(
        os.environ.get("NEMO_RL_VENV_DIR", DEFAULT_VENV_DIR)
    )
    venv_path = Path(NEMO_RL_VENV_DIR) / venv_name
    python_path = venv_path / "bin" / "python"
    started_file = venv_path / "STARTED_ENV_BUILDER"

    # Skip early return if force_rebuild is True
    if not force_rebuild and python_path.exists():
        logger.info(f"Using existing venv at {venv_path}")
        return str(python_path)

    # Sleep to stagger node startup
    time.sleep(1 * node_idx)

    if started_file.exists():
        # Another node is already building, wait for completion
        logger.info(
            f"Node {node_idx}: Another node is building {venv_name}, skipping..."
        )
        # Wait for the venv to be ready (check for python executable)
        python_path = venv_path / "bin" / "python"
        while not python_path.exists():
            time.sleep(1)
        return str(python_path)

    # Create the venv directory if needed
    venv_path.mkdir(parents=True, exist_ok=True)

    # Touch the started file to signal we're building
    started_file.touch()
    try:
        # Create the virtual environment on this node
        return create_local_venv(py_executable, venv_name, force_rebuild=force_rebuild)
    finally:
        # Clean up the started file
        if started_file.exists():
            started_file.unlink()


def create_local_venv_on_each_node(py_executable: str, venv_name: str):
    """Create a virtual environment on each Ray node.

    Args:
        py_executable (str): Command to run with the virtual environment
        venv_name (str): Name of the virtual environment

    Returns:
        str: Path to the python executable in the created virtual environment
    """
    # Determine the number of alive Ray nodes
    nodes = [n for n in ray.nodes() if n.get("Alive", False)]
    num_nodes = len(nodes)
    # Reserve one CPU on each node using a STRICT_SPREAD placement group
    bundles = [{"CPU": 1} for _ in range(num_nodes)]
    pg = placement_group(bundles=bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    force_rebuild = os.environ.get("NRL_FORCE_REBUILD_VENVS", "false").lower() == "true"
    # Launch one actor per node
    actors = [
        _env_builder.options(placement_group=pg).remote(
            py_executable, venv_name, i, force_rebuild
        )
        for i, _ in enumerate(nodes)
    ]
    # ensure setup runs on each node
    paths = ray.get([actor for actor in actors])
    # Normalize paths to handle double slashes and other path inconsistencies
    normalized_paths = [os.path.normpath(p) for p in paths]
    assert len(set(normalized_paths)) == 1, (
        f"All nodes should have the same venv, but got: {set(normalized_paths)}"
    )

    # Clean up the placement group
    ray.util.remove_placement_group(pg)
    # Return mapping from node IP to venv python path
    return paths[0]
