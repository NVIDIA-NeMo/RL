# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Persistent JIT compile caches for NeMo RL's vLLM and Megatron workers (opt-in).

Set ``NRL_COMPILE_CACHE_SHARED_DIR`` (kept between jobs, on a shared filesystem)
and ``NRL_COMPILE_CACHE_LOCAL_DIR`` (node-local). Each cache type (``vllm``,
``megatron``) lives in ``<shared>/<type>``, or in ``NRL_COMPILE_CACHE_<TYPE>_DIR``
if set; a node copies it once to ``<local>/<type>`` (``cache_sync.sh seed-type``)
and points that type's compilers there. Writing new entries back is the
launcher's job (``cache_sync.sh sidecar-types``). See
docs/guides/persistent-jit-caches.md.
"""

import logging
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

SHARED_VAR = "NRL_COMPILE_CACHE_SHARED_DIR"
LOCAL_VAR = "NRL_COMPILE_CACHE_LOCAL_DIR"
SCRIPT_VAR = "NRL_COMPILE_CACHE_SYNC_SCRIPT"
# Compiler cache variables and their folder under <local>/<type>.
COMPILER_DIRS = {
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "TRITON_CACHE_DIR": "triton",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "CUDA_CACHE_PATH": "cuda",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": "fa4_cute",
}


def local_dir() -> str | None:
    """The node-local cache root, or None if persistent caches are off."""
    shared, local = os.environ.get(SHARED_VAR), os.environ.get(LOCAL_VAR)
    if bool(shared) != bool(local):
        raise ValueError(
            f"Persistent compile caches need both {SHARED_VAR} and {LOCAL_VAR}"
        )
    return local or None


def script() -> str:
    """The cache_sync.sh to run: the one next to this module unless overridden."""
    path = os.environ.get(SCRIPT_VAR) or str(Path(__file__).with_name("cache_sync.sh"))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Persistent compile cache script not found: {path}")
    return path


def env_overrides(cache_type: str) -> dict[str, str]:
    """Point the compilers at ``<local>/<cache_type>``; {} if persistent caches are off."""
    local = local_dir()
    if local is None:
        return {}
    env = {var: f"{local}/{cache_type}/{sub}" for var, sub in COMPILER_DIRS.items()}
    env["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
    return env


def seed(cache_type: str) -> None:
    """Copy ``cache_type``'s persistent cache onto this node once (no-op if off).

    Returns when the copy is done, also one another process on this node started
    (e.g. the launcher, to overlap Ray start-up). A failed copy only warns (in
    cache_sync.sh); a missing script or variable raises.
    """
    if local_dir() is None:
        return
    subprocess.run(["bash", script(), "seed-type", cache_type], check=True)


def vllm_engine_dir(seed: int) -> str | None:
    """This engine's own VLLM_CACHE_ROOT under the local root, or None if off.

    Engines on one node must not share a folder (vLLM writes its cache index
    without locking), so each gets its own, filled from ``<local>/vllm/vllm``.
    """
    local = local_dir()
    return None if local is None else f"{local}/vllm_engines/{seed}"


def _vllm_patches() -> dict:
    # By path, as tools/external_gym_vllm/serve_vllm_on_ray.py does: importing the
    # package would pull in NeMo RL's generation stack on every node.
    return runpy.run_path(
        str(Path(__file__).parents[1] / "models/generation/vllm/patches.py")
    )


def _prepare_vllm_engine_on_this_node(engine_dir: str) -> None:
    # The engine's GPU workers on this node load artifacts another engine wrote.
    _vllm_patches()["_patch_vllm_compile_cache_relative_artifacts"](
        logging.getLogger(__name__)
    )
    seed("vllm")
    shared = Path(os.environ[LOCAL_VAR]) / "vllm" / "vllm"
    if shared.is_dir() and not os.path.exists(engine_dir):
        tmp = f"{engine_dir}.tmp{os.getpid()}"
        shutil.copytree(shared, tmp, symlinks=True)  # copy2 keeps mtimes
        os.rename(tmp, engine_dir)
    os.makedirs(engine_dir, exist_ok=True)


def prepare_vllm_engine(bundle_indices: list[int] | None) -> None:
    """Before an engine compiles: seed every node it runs on and fill its folder.

    Runs one task per node of the engine's placement-group bundles (the vLLM
    process's own node included) and waits for all of them.
    """
    engine_dir = os.environ.get("VLLM_CACHE_ROOT")
    if local_dir() is None or not engine_dir:
        return
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    node_ids = {ray.get_runtime_context().get_node_id()}
    pg = ray.util.get_current_placement_group()
    if pg is not None and bundle_indices:
        bundle_nodes = ray.util.placement_group_table(pg)["bundles_to_node_id"]
        node_ids.update(bundle_nodes[i] for i in bundle_indices)
    task = ray.remote(_prepare_vllm_engine_on_this_node)
    ray.get(
        [
            task.options(
                num_cpus=0,
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
                runtime_env={"py_executable": sys.executable},
            ).remote(engine_dir)
            for node_id in sorted(node_ids)
        ]
    )
