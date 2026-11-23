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
launcher's job (``cache_sync.sh sidecar-types``), except for torch.compile's
shape profiles, which ``ShapeProfiles`` saves as complete sets. See
docs/guides/persistent-jit-caches.md.
"""

import getpass
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SHARED_VAR = "NRL_COMPILE_CACHE_SHARED_DIR"
LOCAL_VAR = "NRL_COMPILE_CACHE_LOCAL_DIR"
SCRIPT_VAR = "NRL_COMPILE_CACHE_SYNC_SCRIPT"
SHAPE_PROFILES_SUFFIX = "_shape_profiles"
# Compiler cache variables and their folder under <local>/<type>.
COMPILER_DIRS = {
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "TRITON_CACHE_DIR": "triton",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "CUDA_CACHE_PATH": "cuda",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": "fa4_cute",
}
# Compiler settings one cache type's workers take from NRL_<TYPE>_<VAR>.
TUNING_VARS = (
    # Stores Triton autotune results in TRITON_CACHE_DIR; kernels whose configs
    # carry a pre_hook are still benchmarked in every process.
    "TRITON_CACHE_AUTOTUNING",
    "TRITON_PRINT_AUTOTUNING",
    # A job id that is stable across runs turns on torch.compile's per-rank shape
    # profiles, which ShapeProfiles shares between jobs.
    "TORCH_COMPILE_JOB_ID",
)


def local_dir() -> str | None:
    """The node-local cache root, or None if persistent caches are off."""
    shared, local = os.environ.get(SHARED_VAR), os.environ.get(LOCAL_VAR)
    if bool(shared) != bool(local):
        raise ValueError(
            f"Persistent compile caches need both {SHARED_VAR} and {LOCAL_VAR}"
        )
    return local or None


def type_dir(cache_type: str) -> str:
    """``cache_type``'s persistent folder: ``NRL_COMPILE_CACHE_<TYPE>_DIR`` or ``<shared>/<type>``."""
    return os.environ.get(f"NRL_COMPILE_CACHE_{cache_type.upper()}_DIR") or (
        f"{os.environ[SHARED_VAR]}/{cache_type}"
    )


def script() -> str:
    """The cache_sync.sh to run: the one next to this module unless overridden."""
    path = os.environ.get(SCRIPT_VAR) or str(Path(__file__).with_name("cache_sync.sh"))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Persistent compile cache script not found: {path}")
    return path


def env_overrides(cache_type: str) -> dict[str, str]:
    """Point the compilers at ``<local>/<cache_type>``; {} if persistent caches are off.

    A set ``NRL_<TYPE>_<VAR>`` passes one of ``TUNING_VARS`` to this type's workers.
    """
    local = local_dir()
    if local is None:
        return {}
    env = {var: f"{local}/{cache_type}/{sub}" for var, sub in COMPILER_DIRS.items()}
    env["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
    for var in TUNING_VARS:
        value = os.environ.get(f"NRL_{cache_type.upper()}_{var}")
        if value:
            env[var] = value
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


def _prepare_vllm_engine_on_this_node(engine_dir: str) -> None:
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


def warn_if_autotuning_is_shared(deterministic: bool) -> None:
    """Warn when a deterministic run takes Triton autotune results from the cache."""
    if (
        deterministic
        and local_dir() is not None
        and os.environ.get("TRITON_CACHE_AUTOTUNING") == "1"
    ):
        log.warning(
            "Triton autotune results come from the persistent compile cache: kernels "
            "use the configs of whichever job tuned them first, which a job without "
            "the cache may not pick."
        )


def shape_profile_key(job_id: str, layout: dict[str, Any]) -> str:
    """``job_id`` with the user and a digest of ``layout``, which profiles only fit."""
    try:
        user = getpass.getuser()
    except Exception:
        user = str(os.getuid())
    digest = hashlib.sha256(
        json.dumps(layout, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    return f"{job_id}-{user}-{digest}"


def newest_shape_profile_set(
    sets: Path, world_size: int
) -> tuple[Path, list[int]] | None:
    """The newest complete set in ``sets`` and the ranks it has profiles of."""
    found = []
    for manifest in sets.glob("*/manifest.json"):
        try:
            meta = json.loads(manifest.read_text())
            if meta["world_size"] == world_size:
                found.append((manifest.stat().st_mtime, manifest.parent, meta["ranks"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not found:
        return None
    _, path, ranks = max(found, key=lambda entry: entry[0])
    return path, ranks


def _replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _mtime(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


class _DefaultGroup:
    """Object collectives over torch.distributed's default group."""

    @staticmethod
    def all_gather(obj: Any) -> list[Any]:
        import torch.distributed as dist

        out: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(out, obj)
        return out

    @staticmethod
    def broadcast(obj: Any) -> Any:
        import torch.distributed as dist

        objs = [obj]
        dist.broadcast_object_list(objs, src=0)
        return objs[0]


class ShapeProfiles:
    """torch.compile's per-rank shape profiles, shared between jobs as complete sets.

    With a job id, torch.compile saves per rank which input sizes changed, and the
    next process compiles those as dynamic from the start. Ranks that start from
    profiles of different points of a run compile different graphs and can wait on
    each other's collectives, so a set holds every rank's profile after the same
    step, and a job loads all of one set or none. Sets live in
    ``<type dir>_shape_profiles/<key>/<set>``. Every method is a collective.
    """

    def __init__(self, path: Path, sets: Path, rank: int, world_size: int, group: Any):
        self.path, self.sets = path, sets
        self.rank, self.world_size = rank, world_size
        self.group = group
        self._saved: int | None = None
        self._steps = 0

    @classmethod
    def load(cls, cache_type: str, layout: dict[str, Any]) -> "ShapeProfiles | None":
        """Load the newest complete set saved for this job id, user and ``layout``.

        Call before anything compiles. None if persistent caches are off, no job id
        is set or torch keeps no local profiles.
        """
        import torch
        import torch.distributed as dist
        from torch._dynamo import pgo

        job_id = os.environ.get("TORCH_COMPILE_JOB_ID")
        if local_dir() is None or not job_id:
            return None
        # torch keeps torch.compiler.config per thread, and Ray runs the steps on
        # another thread, so torch's own profile stays under the job id it read
        # from the environment and only the sets are keyed on the layout.
        cache_key = pgo.get_cache_key()
        path = pgo.code_state_path(cache_key) if cache_key else None
        if path is None:
            return None
        world_size = dist.get_world_size()
        key = shape_profile_key(
            job_id, {**layout, "world_size": world_size, "torch": torch.__version__}
        )
        sets = Path(type_dir(cache_type) + SHAPE_PROFILES_SUFFIX) / key
        profiles = cls(Path(path), sets, dist.get_rank(), world_size, _DefaultGroup())
        profiles.load_newest()
        return profiles

    def load_newest(self) -> None:
        """Replace this rank's profile with its copy in the newest complete set."""
        self.path.unlink(missing_ok=True)
        newest = self.group.broadcast(
            newest_shape_profile_set(self.sets, self.world_size)
            if self.rank == 0
            else None
        )
        set_dir, ranks = newest if newest is not None else (None, [])
        ok = True
        if set_dir is not None and self.rank in ranks:
            try:
                _replace(self.path, (set_dir / f"{self.rank}.pkl").read_bytes())
            except OSError as e:
                log.warning("Could not load shape profile from %s: %s", set_dir, e)
                ok = False
        if not all(self.group.all_gather(ok)):
            self.path.unlink(missing_ok=True)
            if self.rank == 0:
                log.warning(
                    "A rank could not load its shape profile from %s; all ranks "
                    "start without one",
                    set_dir,
                )
        elif self.rank == 0:
            found = f"loaded {set_dir}" if set_dir else f"no set in {self.sets} yet"
            log.info("torch.compile shape profiles: %s", found)
        self._saved = _mtime(self.path)

    def step_done(self, update_successful: bool) -> None:
        """After every optimizer step; saves a set after steps 1, 2, 4, 8, ...

        Saves only if the step succeeded on every rank and some rank's profile
        changed since the load or the last save.
        """
        self._steps += 1
        if self._steps & (self._steps - 1):
            return
        mtime = _mtime(self.path)
        votes = self.group.all_gather(
            (update_successful, mtime is not None and mtime != self._saved)
        )
        if not all(ok for ok, _ in votes) or not any(new for _, new in votes):
            return
        name = self.group.broadcast(
            f"{time.strftime('%Y%m%d-%H%M%S')}-"
            f"{os.environ.get('SLURM_JOB_ID', os.getpid())}-step{self._steps}"
            if self.rank == 0
            else None
        )
        set_dir = self.sets / name
        ok = True
        if mtime is not None:
            try:
                _replace(set_dir / f"{self.rank}.pkl", self.path.read_bytes())
            except OSError as e:
                log.warning("Could not save shape profile to %s: %s", set_dir, e)
                ok = False
        saved = self.group.all_gather((ok, mtime is not None))
        if not all(ok for ok, _ in saved):
            if self.rank == 0:
                log.warning("Shape profile set %s is incomplete; not used", set_dir)
            return
        self._saved = mtime
        if self.rank != 0:
            return
        ranks = [rank for rank, (_, has) in enumerate(saved) if has]
        manifest = {"world_size": self.world_size, "ranks": ranks}
        try:
            _replace(set_dir / "manifest.json", json.dumps(manifest).encode())
        except OSError as e:
            log.warning("Could not complete shape profile set %s: %s", set_dir, e)
            return
        log.info(
            "torch.compile shape profiles: saved %d ranks' after step %d to %s",
            len(ranks),
            self._steps,
            set_dir,
        )
