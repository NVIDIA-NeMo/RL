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


def _reconcile_cutlass_cu13(venv_path: str, venv_name: str) -> None:
    # Workaround for the nvidia-cutlass-dsl packaging bug where the -libs-base and
    # -libs-cu13 component wheels write DIVERGENT content to the SAME paths under
    # nvidia_cutlass_dsl/ (the _cutlass_ir .so, MLIR bindings, and CuTe-DSL .py
    # sources). Last-writer-wins, and under uv the install order is racy, so the
    # venv can end up a non-deterministic mix of both variants.
    # cuDNN-frontend 1.25.0's cutedsl DeepSeek-Sparse-Attention path needs the
    # -libs-cu13 API (normalize_field_to_ir_name present, atom_tma_partition
    # target_tensors, nvvm.atomicrmw without a leading `res` arg); the -libs-base
    # side breaks it. This forces the CuTe-DSL tree to single-provenance -libs-cu13.
    # vLLM does the OPPOSITE (strips [cu13] -> base) for its own kernels, so this
    # only runs for venvs matching NRL_CUTLASS_CU13_VENV_SUBSTR (default "megatron")
    # and never touches the vLLM generation venv (which wants base).
    # Upstream (packaging bug, still OPEN):
    #   https://github.com/NVIDIA/cutlass/issues/3170
    #   https://github.com/NVIDIA/cutlass/issues/3259
    # Opt-in via NRL_FORCE_CUTLASS_CU13=1. Idempotent; best-effort (never fails the
    # build -- if it can't reconcile, the run fails loudly later at the cuDNN import).
    import glob

    if os.environ.get("NRL_FORCE_CUTLASS_CU13", "") not in ("1", "true", "True"):
        return
    substr = os.environ.get("NRL_CUTLASS_CU13_VENV_SUBSTR", "megatron")
    if substr and substr not in venv_name:
        return

    def _is_cu13(cutlass_dir: str) -> bool:
        common = os.path.join(cutlass_dir, "cute", "nvgpu", "common.py")
        opsgen = os.path.join(cutlass_dir, "_mlir", "dialects", "_nvvm_ops_gen.py")
        try:
            with open(common) as fh:
                has_norm = "def normalize_field_to_ir_name" in fh.read()
            with open(opsgen) as fh:
                no_res = "def atomicrmw(op, ptr, a" in fh.read()
            return has_norm and no_res
        except OSError:
            return False

    try:
        sps = glob.glob(os.path.join(venv_path, "lib", "python*", "site-packages"))
        if not sps:
            return
        dst = os.path.join(sps[0], "nvidia_cutlass_dsl", "python_packages", "cutlass")
        if not os.path.isdir(dst):
            return  # no CuTe-DSL in this venv; nothing to reconcile
        if _is_cu13(dst):
            logger.info(f"[cutlass-cu13] {venv_name}: already single-provenance cu13; skipping")
            return
        uv_cache = os.environ.get("UV_CACHE_DIR", os.path.expanduser("~/.cache/uv"))
        src = None
        pattern = os.path.join(uv_cache, "archive-v0", "*", "nvidia_cutlass_dsl_libs_cu13*.dist-info")
        for di in glob.glob(pattern):
            cand = os.path.join(os.path.dirname(di), "nvidia_cutlass_dsl", "python_packages", "cutlass")
            if os.path.isdir(cand) and _is_cu13(cand):
                src = cand
                break
        if src is None:
            logger.warning(
                f"[cutlass-cu13] {venv_name}: no pristine -libs-cu13 CuTe-DSL tree found under "
                f"{uv_cache}; leaving venv as-is (cuDNN DSA may fail). "
                "See https://github.com/NVIDIA/cutlass/issues/3170"
            )
            return
        try:
            subprocess.run(["rsync", "-a", src + "/", dst + "/"], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            subprocess.run(f"cp -a {shlex.quote(src)}/. {shlex.quote(dst)}/", shell=True, check=True)
        logger.info(f"[cutlass-cu13] {venv_name}: reconciled CuTe-DSL tree to -libs-cu13 from {src}")
    except Exception as e:  # best-effort: never break venv creation
        logger.warning(f"[cutlass-cu13] {venv_name}: reconcile skipped due to {e!r}")


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
    if force_rebuild:
        # Serialize installs to avoid CUTLASS DSL wheel overlap races during cache rebuilds.
        env.setdefault("UV_CONCURRENT_INSTALLS", "1")

    # Split the py_executable into command and arguments
    exec_cmd = shlex.split(py_executable)
    # Command doesn't matter, since `uv` syncs the environment no matter the command.
    exec_cmd.extend(["echo", f"Finished creating venv {venv_path}"])

    # Always run uv sync first to ensure the build requirements are set (for --no-build-isolation packages)
    subprocess.run(["uv", "sync", "--directory", git_root], env=env, check=True)
    subprocess.run(exec_cmd, env=env, check=True)

    # cutlass-dsl -libs-base/-libs-cu13 divergence workaround (NVIDIA/cutlass#3170, #3259):
    # force the CuTe-DSL install to single-provenance -libs-cu13 for cuDNN DSA
    # venvs (opt-in via NRL_FORCE_CUTLASS_CU13; no-op otherwise).
    _reconcile_cutlass_cu13(venv_path, venv_name)

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
    Path(NEMO_RL_VENV_DIR).mkdir(parents=True, exist_ok=True)
    started_file = Path(NEMO_RL_VENV_DIR) / f".{venv_name}.STARTED_ENV_BUILDER"

    # Skip early return if force_rebuild is True
    if not force_rebuild and python_path.exists():
        logger.info(f"Using existing venv at {venv_path}")
        return str(python_path)

    # Sleep to stagger node startup
    time.sleep(1 * node_idx)

    try:
        fd = os.open(started_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        owns_build = True
    except FileExistsError:
        owns_build = False

    if not owns_build:
        # Another node is already building, wait for completion
        logger.info(
            f"Node {node_idx}: Another node is building {venv_name}, skipping..."
        )
        # Wait for the venv to be ready (check for python executable)
        python_path = venv_path / "bin" / "python"
        while started_file.exists():
            if python_path.exists():
                return str(python_path)
            time.sleep(1)
        if not python_path.exists():
            raise RuntimeError(f"Venv build failed before {python_path} was created")
        return str(python_path)

    # Create the venv directory if needed
    venv_path.mkdir(parents=True, exist_ok=True)

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

    Used by ReplayBuffer, AsyncTrajectoryCollector, and
    SyncRolloutActor — three actors that need the VLLM tier's
    venv on every node.
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
