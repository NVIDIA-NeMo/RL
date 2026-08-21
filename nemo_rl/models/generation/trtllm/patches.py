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

"""Inject a sync, Ray-picklable ``fetch_stats_serialized`` onto TRT-LLM's BaseWorker.

Needed because ``collective_rpc("fetch_stats")`` returns unpicklable C++ bindings,
the RPC-client ``get_stats()`` path hits a stats-less engine and always returns [],
and the built-in serialized fetches are async (unusable via the sync
``RayGPUWorker.call_worker_method``). This sync method returns JSON strings that
survive the Ray RPC round-trip.

The patch reaches worker processes by appending the method definition block to
the shared-venv ``base_worker.py`` source that the separate ``RayGPUWorker``
processes import at startup.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from importlib.util import find_spec


def _get_trtllm_file(relative_path: str) -> str:
    """Return absolute path to a TRT-LLM file or raise if it cannot be found.

    The relative_path should be a POSIX-style path under the tensorrt_llm
    package root, e.g. "executor/base_worker.py".
    """
    spec = find_spec("tensorrt_llm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "tensorrt_llm package not found while attempting to patch "
            f"'{relative_path}'. Ensure TRT-LLM is installed and "
            "available in this environment."
        )

    base_dir = next(iter(spec.submodule_search_locations))
    file_path = os.path.join(base_dir, *relative_path.split("/"))

    if not os.path.exists(file_path):
        raise RuntimeError(
            "Failed to locate expected TRT-LLM file to patch. "
            f"Looked for '{relative_path}' at '{file_path}'. "
            "This likely indicates an unexpected TRT-LLM installation "
            "layout or version mismatch."
        )

    return file_path


@contextmanager
def _locked_file_patch(file_path: str):
    """Yield (content, writer) under an exclusive file lock."""
    import fcntl

    lock_path = file_path + ".patch_lock"
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        with open(file_path, "r") as f:
            content = f.read()

        def write_back(new_content: str):
            with open(file_path, "w") as f:
                f.write(new_content)

        yield content, write_back
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _patch_trtllm_fetch_stats_serialized(logger) -> None:
    """Append ``fetch_stats_serialized`` block to TRT-LLM's base_worker.py.

    Co-located replica actors share the container venv, so flock serializes
    concurrent calls. After writing, the file is read back; a warning is logged
    if the block did not persist (the failure mode that previously caused a flat
    line of zeros with no diagnostic).
    """
    marker = "# --- nemo-rl IFB metric patch ---"
    block = (
        "\n\n" + marker + "\n"
        "def _nemorl_fetch_stats_serialized(self):\n"
        "    return [self._stats_serializer(s) for s in self.fetch_stats()]\n"
        "try:\n"
        "    BaseWorker.fetch_stats_serialized = _nemorl_fetch_stats_serialized\n"
        "except Exception:\n"
        "    pass\n"
        "# --- end nemo-rl IFB metric patch ---\n"
    )
    try:
        file_path = _get_trtllm_file("executor/base_worker.py")
    except RuntimeError as e:
        logger.warning("Could not locate base_worker.py for IFB stats patch: %s", e)
        return

    with _locked_file_patch(file_path) as (content, write_back):
        if marker in content:
            logger.info("IFB stats patch already present in %s", file_path)
            return
        write_back(content + block)

    # Read back so a patch that silently failed to land is not reported as
    # applied; this is the failure mode that previously went unnoticed.
    try:
        with open(file_path) as handle:
            applied = marker in handle.read()
    except OSError as error:
        logger.warning("Could not verify IFB stats patch: %s", error)
        return

    if applied:
        logger.info("IFB stats patch appended to %s", file_path)
    else:
        logger.warning(
            "IFB stats patch did not persist to %s. Metrics will show as a "
            "flat line of zeros with no further warning from the worker processes.",
            file_path,
        )


def apply() -> bool:
    """Add ``fetch_stats_serialized`` to TRT-LLM's ``BaseWorker`` if absent.

    Returns True if the method is present after the call (patched or already
    there), False if TRT-LLM's worker module could not be imported.
    """
    try:
        from tensorrt_llm.executor.base_worker import BaseWorker
    except Exception as exc:  # pragma: no cover - TRT-LLM not importable
        print(
            f"[ifb_stats_patch] BaseWorker import failed: {exc!r}",
            file=sys.stderr,
            flush=True,
        )
        return False

    if getattr(BaseWorker, "fetch_stats_serialized", None) is not None:
        return True

    def fetch_stats_serialized(self):  # type: ignore[no-untyped-def]
        """Fetch + serialize iteration stats to JSON strings (picklable for collective_rpc)."""
        return [self._stats_serializer(s) for s in self.fetch_stats()]

    BaseWorker.fetch_stats_serialized = fetch_stats_serialized
    print(
        "[ifb_stats_patch] BaseWorker.fetch_stats_serialized installed",
        file=sys.stderr,
        flush=True,
    )
    return True
