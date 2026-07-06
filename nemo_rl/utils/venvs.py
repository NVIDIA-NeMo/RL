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
import shutil
import subprocess
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import ray
from ray.util import placement_group

dir_path = os.path.dirname(os.path.abspath(__file__))
git_root = os.path.abspath(os.path.join(dir_path, "../.."))
DEFAULT_VENV_DIR = os.path.join(git_root, "venvs")

logger = logging.getLogger(__name__)


def _is_truthy(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


@contextmanager
def _serialized_uv_sync():
    if not _is_truthy(os.environ.get("NRL_SERIALIZE_UV_SYNC", "false")):
        yield
        return

    import fcntl

    lock_path_env = os.environ.get("NRL_UV_SYNC_LOCK_PATH")
    lock_path = (
        Path(lock_path_env)
        if lock_path_env
        else Path(git_root) / ".nemo_rl_uv_sync.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        logger.info("Waiting for uv sync lock at %s", lock_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            logger.info("Acquired uv sync lock at %s", lock_path)
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _uv_no_install_package_args() -> list[str]:
    packages = [
        package.strip()
        for package in os.environ.get("NRL_UV_NO_INSTALL_PACKAGES", "").split(",")
        if package.strip()
    ]
    args = []
    for package in packages:
        args.extend(["--no-install-package", package])
    return args


def _uv_sync_cmd_from_run_cmd(exec_cmd: list[str]) -> list[str]:
    """Build the matching ``uv sync`` command for a ``uv run`` executable."""
    sync_cmd = ["uv", "sync"]
    seen_directory = False
    if len(exec_cmd) >= 2 and exec_cmd[:2] == ["uv", "run"]:
        idx = 2
        while idx < len(exec_cmd):
            arg = exec_cmd[idx]
            if arg == "--":
                break
            if not arg.startswith("-"):
                break

            if arg in {
                "--locked",
                "--frozen",
                "--all-extras",
                "--no-dev",
                "--no-default-groups",
                "--all-groups",
            }:
                sync_cmd.append(arg)
                idx += 1
                continue

            if arg in {
                "--extra",
                "--no-extra",
                "--group",
                "--no-group",
                "--only-group",
                "--directory",
                "--project",
            }:
                if idx + 1 >= len(exec_cmd):
                    break
                sync_cmd.extend([arg, exec_cmd[idx + 1]])
                if arg == "--directory":
                    seen_directory = True
                idx += 2
                continue

            if arg.startswith("--directory="):
                seen_directory = True
                sync_cmd.append(arg)
                idx += 1
                continue

            if any(
                arg.startswith(prefix)
                for prefix in (
                    "--extra=",
                    "--no-extra=",
                    "--group=",
                    "--no-group=",
                    "--only-group=",
                    "--project=",
                )
            ):
                sync_cmd.append(arg)
                idx += 1
                continue

            idx += 1

    if not seen_directory:
        sync_cmd.extend(["--directory", git_root])
    return sync_cmd


def _uv_run_no_sync_cmd(exec_cmd: list[str]) -> list[str]:
    if len(exec_cmd) >= 2 and exec_cmd[:2] == ["uv", "run"] and "--no-sync" not in exec_cmd:
        return [*exec_cmd[:2], "--no-sync", *exec_cmd[2:]]
    return exec_cmd


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
    uv_no_install_args = _uv_no_install_package_args()
    if uv_no_install_args:
        logger.info(
            "Passing NRL_UV_NO_INSTALL_PACKAGES to uv: %s",
            os.environ["NRL_UV_NO_INSTALL_PACKAGES"],
        )
    sync_cmd = [*_uv_sync_cmd_from_run_cmd(exec_cmd), *uv_no_install_args]
    exec_cmd = _uv_run_no_sync_cmd(exec_cmd)

    # Command doesn't matter, since `uv` syncs the environment no matter the command.
    exec_cmd.extend(["echo", f"Finished creating venv {venv_path}"])

    # Always run uv sync first to ensure the build requirements are set (for --no-build-isolation packages)
    with _serialized_uv_sync():
        subprocess.run(sync_cmd, env=env, check=True)
        subprocess.run(exec_cmd, env=env, check=True)

    Path(venv_path, ".NEMO_RL_VENV_READY").touch()

    # Return the path to the python executable in the virtual environment
    python_path = os.path.join(venv_path, "bin", "python")
    return python_path


# Ray-based helper to create a virtual environment on each Ray node
@ray.remote(num_cpus=1)  # pragma: no cover
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
    ready_file = venv_path / ".NEMO_RL_VENV_READY"

    # Skip early return if force_rebuild is True
    if not force_rebuild and python_path.exists() and ready_file.exists():
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
        while started_file.exists() and not (
            python_path.exists() and ready_file.exists()
        ):
            time.sleep(1)
        if python_path.exists() and ready_file.exists():
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
    # Skip nodes with 0 CPUs (e.g. unschedulable head nodes) — including them
    # makes the STRICT_SPREAD placement group infeasible.
    nodes = [
        n
        for n in ray.nodes()
        if n.get("Alive", False) and n.get("Resources", {}).get("CPU", 0) > 0
    ]
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


def make_actor_runtime_env(actor_class_fqn: str) -> dict:
    """Build a Ray ``runtime_env`` for one of our registered actors.

    Resolves the actor's tier-specific py_executable via the registry,
    materializes a per-node venv when uv-managed, and packages it with
    ``VIRTUAL_ENV`` / ``UV_PROJECT_ENVIRONMENT`` env vars so workers see
    the same interpreter as the driver.

    Used by ReplayBuffer, AsyncTrajectoryCollector, and SyncRolloutActor
    — three actors that need the VLLM tier's venv on every node. Also
    used by the SGLang router and SGLang generation engines (SGLANG tier).
    """
    # Local import — venvs.py is dep-light; the registry imports
    # PY_EXECUTABLES which transitively pulls heavier deps.
    from nemo_rl.distributed.ray_actor_environment_registry import (
        get_actor_python_env,
    )

    py_exec = get_actor_python_env(actor_class_fqn)
    if py_exec.startswith("uv"):
        py_exec = create_local_venv_on_each_node(py_exec, actor_class_fqn)
    venv = os.path.dirname(os.path.dirname(py_exec))  # strip bin/python
    return {
        "py_executable": py_exec,
        "env_vars": {
            **os.environ,
            "VIRTUAL_ENV": venv,
            "UV_PROJECT_ENVIRONMENT": venv,
        },
    }
