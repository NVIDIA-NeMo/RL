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
"""Persistent JIT compile caches for NeMo RL's vLLM and Megatron workers (opt-in).

Each worker type keeps its caches in ``${NRL_COMPILE_CACHE_ROOT}/<type>`` on a
shared filesystem; a node uses a local copy, ``${NRL_LOCAL_COMPILE_CACHE}/<type>``,
made once per node by ``cache_sync.sh seed-once`` (``NRL_CACHE_SYNC``). A launcher
can start that copy in its node setup, in the background; ``seed()`` then just
waits for it. ``NRL_<PREFIX>_<VAR>`` points one worker type's compilers at its
copy (e.g. ``NRL_VLLM_TRITON_CACHE_DIR``). Writing new entries back is the
launcher's job (``cache_sync.sh sidecar-types``).
"""

import os
import subprocess
import time

CACHE_VARS = (
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "CUDA_CACHE_PATH",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED",
    # Stores Triton autotune results in TRITON_CACHE_DIR; kernels whose configs
    # carry a pre_hook are still benchmarked in every process.
    "TRITON_CACHE_AUTOTUNING",
    "TRITON_PRINT_AUTOTUNING",
    # A job id that is stable across runs makes torch.compile keep per-rank shape
    # profiles under TORCHINDUCTOR_CACHE_DIR/dynamo; see wait_for_shape_profiles.
    "TORCH_COMPILE_JOB_ID",
)
_SEED_VARS = ("NRL_CACHE_SYNC", "NRL_COMPILE_CACHE_ROOT", "NRL_LOCAL_COMPILE_CACHE")
# Cache types whose seed() in this process copies the shape profiles first.
_PROFILE_SEEDS: set[str] = set()


def env_overrides(prefix: str) -> dict[str, str]:
    """Map the set ``NRL_<prefix>_<VAR>`` variables to ``<VAR>``."""
    return {
        var: os.environ[f"NRL_{prefix}_{var}"]
        for var in CACHE_VARS
        if os.environ.get(f"NRL_{prefix}_{var}")
    }


def _shape_profile_dir(type_dir: str) -> str | None:
    """torch.compile's shape-profile dir relative to ``type_dir``, if inside it."""
    inductor = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not (os.environ.get("TORCH_COMPILE_JOB_ID") and inductor):
        return None
    type_dir = os.path.abspath(type_dir)
    profiles = os.path.abspath(os.path.join(inductor, "dynamo"))
    if os.path.commonpath([profiles, type_dir]) != type_dir:
        return None
    return os.path.relpath(profiles, type_dir)


def seed(cache_type: str, background: bool = False) -> None:
    """Copy ``cache_type``'s persistent cache onto this node once (no-op if unset).

    Blocking, it also waits for a copy another process on this node started.
    torch.compile's shape profiles under the type's cache are copied first.
    """
    values = [os.environ.get(var) for var in _SEED_VARS]
    if not any(values):
        return
    if not all(values):
        raise ValueError(f"Persistent compile caches need all of {_SEED_VARS}")
    script, root, local = values
    cmd = ["bash", script, "seed-once", f"{root}/{cache_type}", f"{local}/{cache_type}"]
    profiles = _shape_profile_dir(f"{local}/{cache_type}")
    if profiles:
        cmd.append(profiles)
        _PROFILE_SEEDS.add(cache_type)
    if background:
        subprocess.Popen(cmd, start_new_session=True)
    else:
        subprocess.run(cmd, check=True)


def wait_for_shape_profiles(cache_type: str, timeout_s: float = 60.0) -> None:
    """Wait up to ``timeout_s`` for ``seed()`` to place this type's shape profiles.

    torch reads a rank's profile once, at the process's first compile, so call this
    right before it. Returns at once unless this process's ``seed()`` copies them.
    """
    local = os.environ.get("NRL_LOCAL_COMPILE_CACHE")
    if cache_type not in _PROFILE_SEEDS or not local:
        return
    type_dir = f"{local}/{cache_type}"
    # .seeded also covers a copy that finished, or failed, before this call.
    markers = (f"{type_dir}.seeded-first", f"{type_dir}.seeded")
    deadline = time.monotonic() + timeout_s
    while not any(os.path.exists(m) for m in markers):
        if time.monotonic() >= deadline:
            print(
                f"[compile_cache] {cache_type}: torch.compile shape profiles not in "
                f"place after {timeout_s:.0f} s; compiling without them",
                flush=True,
            )
            return
        time.sleep(0.1)
