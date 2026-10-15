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

CACHE_VARS = (
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "CUDA_CACHE_PATH",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED",
)
_SEED_VARS = ("NRL_CACHE_SYNC", "NRL_COMPILE_CACHE_ROOT", "NRL_LOCAL_COMPILE_CACHE")


def env_overrides(prefix: str) -> dict[str, str]:
    """Map the set ``NRL_<prefix>_<VAR>`` variables to ``<VAR>``."""
    return {
        var: os.environ[f"NRL_{prefix}_{var}"]
        for var in CACHE_VARS
        if os.environ.get(f"NRL_{prefix}_{var}")
    }


def seed(cache_type: str, background: bool = False) -> None:
    """Copy ``cache_type``'s persistent cache onto this node once (no-op if unset).

    Blocking, it also waits for a copy another process on this node started.
    """
    values = [os.environ.get(var) for var in _SEED_VARS]
    if not any(values):
        return
    if not all(values):
        raise ValueError(f"Persistent compile caches need all of {_SEED_VARS}")
    script, root, local = values
    cmd = ["bash", script, "seed-once", f"{root}/{cache_type}", f"{local}/{cache_type}"]
    if background:
        subprocess.Popen(cmd, start_new_session=True)
    else:
        subprocess.run(cmd, check=True)
