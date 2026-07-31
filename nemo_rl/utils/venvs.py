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
from functools import lru_cache
from pathlib import Path

import ray
from ray.util import placement_group

dir_path = os.path.dirname(os.path.abspath(__file__))
git_root = os.path.abspath(os.path.join(dir_path, "../.."))
DEFAULT_VENV_DIR = os.path.join(git_root, "venvs")

logger = logging.getLogger(__name__)

# Env-var driven local editable installs.
#
#   NRL_<TIER>_EDITABLE — a *replacement*: syncs the tier's extra from the lockfile
#                         but skips building the pinned `skip_package` (e.g. no
#                         ~60-min tensorrt-llm wheel build), then installs the local
#                         path with `uv pip install --no-deps -e`. The extra provides
#                         the (locked) runtime deps; the pre-built editable provides
#                         the package itself. e.g. NRL_TRTLLM_EDITABLE.
# Value is a comma-separated list of local paths.
# Each entry: (env_var, venv-name substring, distribution to skip building).
_TIER_EDITABLE_ENV_VARS = (("NRL_TRTLLM_EDITABLE", "trtllm", "tensorrt-llm"),)


def _tier_editable_paths_for_venv(venv_name: str) -> list[str]:
    """Return replacement-editable paths (from NRL_<TIER>_EDITABLE) for this venv."""
    paths: list[str] = []
    for env_key, tier, _skip_package in _TIER_EDITABLE_ENV_VARS:
        if tier in venv_name.lower():
            paths += [p.strip() for p in os.environ.get(env_key, "").split(",") if p.strip()]
    return paths


def _tier_skip_package_for_venv(venv_name: str) -> str | None:
    """Return the pinned distribution to exclude from `uv sync` (the editable
    replaces it) for this venv's active tier editable, or None."""
    for env_key, tier, skip_package in _TIER_EDITABLE_ENV_VARS:
        if tier in venv_name.lower() and os.environ.get(env_key, "").strip():
            return skip_package
    return None


# Main-venv path whose cuDNN is treated as authoritative (all NeMo-RL containers
# ship it here). Overridable for non-standard images.
_MAIN_VENV = os.environ.get("NRL_MAIN_VENV", "/opt/nemo_rl_venv")


def _cudnn_link(target: str, linkname: str) -> None:
    """Idempotently (re)create a symlink; log, don't raise, on failure."""
    try:
        if os.path.islink(linkname) or os.path.exists(linkname):
            os.remove(linkname)
        os.symlink(target, linkname)
    except OSError as exc:  # pragma: no cover
        logger.warning(f"cuDNN symlink {linkname} -> {target} failed: {exc}")


def _cudnn_minor_suffix(cudnn_lib_dir: str) -> str:
    """Return the '.MINOR.PATCH' suffix the loader appends to libcudnn*.so.9
    (e.g. '.20.0' for cuDNN 9.20.0), or '' if it can't be determined.

    Derived from the installed nvidia-cudnn wheel's dist-info version: pure
    filesystem, so no GPU, no dlopen, and no system-wide cuDNN are needed (the
    /usr/lib soname the previous implementation relied on is absent in the
    baked containers). Falls back to a system versioned soname if present.
    """
    import glob
    import re

    # .../nvidia/cudnn/lib -> .../site-packages
    site_packages = os.path.dirname(os.path.dirname(os.path.dirname(cudnn_lib_dir)))
    for dist_info in glob.glob(
        os.path.join(site_packages, "nvidia_cudnn_cu*-*.dist-info")
    ):
        # nvidia_cudnn_cu13-9.20.0.48 -> minor=20 patch=0 -> ".20.0"
        m = re.search(
            r"nvidia_cudnn_cu[0-9]+-[0-9]+\.([0-9]+)\.([0-9]+)",
            os.path.basename(dist_info),
        )
        if m:
            return f".{m.group(1)}.{m.group(2)}"
    for soname in glob.glob("/usr/lib/*/libcudnn.so.9.*"):
        suffix = os.path.basename(soname)[len("libcudnn.so.9") :]
        if suffix:
            return suffix
    return ""


def _fix_cudnn_symlinks(venv_path: str) -> None:
    """Reconcile cuDNN after a freshly-built editable venv so it doesn't hit
    CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED / _VERSION_MISMATCH in conv ops.

    pip's nvidia-cudnn wheels ship only the major soname (libcudnn_*.so.9), but
    TE / the cuDNN frontend dlopen the full minor-versioned name
    (libcudnn_*.so.9.X.Y). (1) add those versioned symlinks for the SUBLIBRARIES
    in the main venv (not the main lib — the `cudnn` python pkg globs
    libcudnn.so.* and asserts exactly one match); (2) point this venv's cuDNN at
    the main venv's copies so the preloaded main lib and the dlopen'd sublibs
    share one build. Best-effort no-op if dirs are absent.

    Only called after an editable install below. Baked images get the same fix
    at build time via docker/mlperf/Dockerfile.cudnn-fix.
    """
    import glob

    main_dirs = glob.glob(
        os.path.join(_MAIN_VENV, "lib/python*/site-packages/nvidia/cudnn/lib")
    )
    trt_dirs = glob.glob(
        os.path.join(venv_path, "lib/python*/site-packages/nvidia/cudnn/lib")
    )
    if not (main_dirs and trt_dirs):
        return
    main_dir, trt_dir = main_dirs[0], trt_dirs[0]
    suffix = _cudnn_minor_suffix(main_dir)
    if not suffix:
        return
    # (1) minor-versioned names for the sublibraries in the authoritative dir.
    for f in glob.glob(os.path.join(main_dir, "libcudnn_*.so.9")):
        _cudnn_link(os.path.basename(f), f + suffix)  # relative, within main_dir
    # (2) point this venv's cuDNN (main lib + sublibs + versioned) at the main venv.
    for f in glob.glob(os.path.join(main_dir, "libcudnn*.so.9")) + glob.glob(
        os.path.join(main_dir, "libcudnn_*.so.9" + suffix)
    ):
        _cudnn_link(f, os.path.join(trt_dir, os.path.basename(f)))  # absolute -> main
    logger.info(f"Applied cuDNN symlink fix ({trt_dir} -> {main_dir}, minor {suffix})")


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
    # Command doesn't matter, since `uv` syncs the environment no matter the command.
    exec_cmd.extend(["echo", f"Finished creating venv {venv_path}"])

    # Always run uv sync first to ensure the build requirements are set (for --no-build-isolation packages)
    subprocess.run(["uv", "sync", "--directory", git_root], env=env, check=True)

    # A tier replacement editable (e.g. NRL_TRTLLM_EDITABLE) means: sync the tier's
    # extra from the lockfile but SKIP building the pinned package (e.g. avoid the
    # ~60-min tensorrt-llm wheel build), so all of the extra's locked runtime deps
    # land while the package itself is provided by the local editable below.
    tier_editable_paths = _tier_editable_paths_for_venv(venv_name)
    skip_package = _tier_skip_package_for_venv(venv_name)
    if skip_package:
        # Derive the sync command from py_executable (`uv run --locked --extra
        # <tier> --directory <root>`): swap `run` -> `sync`, exclude the pinned pkg.
        sync_cmd = shlex.split(py_executable)
        sync_cmd[1] = "sync"  # `uv run ...` -> `uv sync ...`
        sync_cmd += ["--no-install-package", skip_package]
        subprocess.run(sync_cmd, env=env, check=True)
    else:
        subprocess.run(exec_cmd, env=env, check=True)

    # Path to the python executable in the virtual environment
    python_path = os.path.join(venv_path, "bin", "python")

    # Install the local editable (NRL_<TIER>_EDITABLE) into this venv. Workers
    # launch via <venv>/bin/python directly (not `uv run`), so a
    # `uv run --with-editable` overlay would not reach them — the editable must
    # land inside UV_PROJECT_ENVIRONMENT. This runs after `uv sync` so it isn't
    # pruned. Installed with --no-deps: the extra sync above already provided the
    # (locked) runtime deps, and the pre-built editable's own metadata would
    # otherwise trigger an unsatisfiable PyPI resolution.
    for path in tier_editable_paths:
        logger.info(f"Installing editable '{path}' into {venv_path}")
        subprocess.run(
            ["uv", "pip", "install", "--no-deps", "--python", python_path, "-e", path],
            env=env,
            check=True,
        )

    # A tier editable (e.g. trtllm) pulls in cuDNN vision/conv ops; reconcile the
    # cuDNN sublibs so the freshly-built venv doesn't fail with
    # CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED / _VERSION_MISMATCH at runtime. Baked
    # images get this at build time via docker/mlperf/Dockerfile.cudnn-fix.
    # Set NRL_SKIP_CUDNN_FIX=1 to skip (e.g. containers whose cuDNN is already OK).
    _skip_cudnn = os.environ.get("NRL_SKIP_CUDNN_FIX", "").lower() in ("1", "true", "yes")
    if tier_editable_paths and not _skip_cudnn:
        _fix_cudnn_symlinks(venv_path)

    # Return the path to the python executable in the virtual environment
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
    # When a local editable is requested for this venv's tier (NRL_<TIER>_EDITABLE),
    # rebuild just this venv (not every venv) so it is rebuilt as base + editable
    # instead of syncing the pinned wheel.
    if not force_rebuild and _tier_editable_paths_for_venv(venv_name):
        logger.info(
            f"Editable install requested for {venv_name}; forcing rebuild of this venv."
        )
        force_rebuild = True
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
