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

"""Node-local and persistent cache helpers for vLLM workers."""

import fcntl
import os
import shutil


def worker_vllm_cache_root(seed: int) -> str:
    """Return the isolated cache root for one vLLM worker group."""
    cache_root_base = os.environ.get("NRL_VLLM_CACHE_ROOT_BASE")
    if cache_root_base is None:
        cache_root_base = os.path.expanduser("~/.cache")
    else:
        cache_root_base = os.path.expanduser(cache_root_base)
        if not os.path.isabs(cache_root_base):
            raise ValueError("NRL_VLLM_CACHE_ROOT_BASE must be an absolute path")
    return os.path.join(cache_root_base, f"vllm_{seed}")


def writeback_vllm_cache() -> bool:
    """Merge a node-local vLLM cache into its optional persistent seed."""
    source = os.environ.get("VLLM_CACHE_ROOT")
    destination_base = os.environ.get("NRL_VLLM_CACHE_WRITEBACK_DIR")
    if not source or not destination_base:
        return False

    source = os.path.abspath(os.path.expanduser(source))
    destination_base = os.path.abspath(os.path.expanduser(destination_base))
    if not os.path.isdir(source):
        return False

    destination = os.path.join(destination_base, os.path.basename(source))
    if source == destination:
        return False

    lock_path = os.path.join(destination_base, f".{os.path.basename(source)}.lock")
    try:
        os.makedirs(destination_base, exist_ok=True)
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                shutil.copytree(source, destination, dirs_exist_ok=True)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except OSError as error:
        print(f"vLLM cache writeback failed: {error}")
        return False

    print(f"vLLM cache written back: {source} -> {destination}")
    return True
