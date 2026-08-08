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

"""Inject a sync, Ray-picklable ``fetch_stats_serialized`` onto TRT-LLM's BaseWorker.

Needed because ``collective_rpc("fetch_stats")`` returns unpicklable C++ bindings,
the RPC-client ``get_stats()`` path hits a stats-less engine and always returns [],
and the built-in serialized fetches are async (unusable via the sync
``RayGPUWorker.call_worker_method``). This sync method returns JSON strings that
survive the Ray RPC round-trip. Loaded in RayGPUWorker processes via a ``.pth``
file — those processes don't import nemo_rl otherwise.
"""

from __future__ import annotations

import sys


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


# Apply on import so a `.pth`-driven `import ..._ifb_stats_patch` at interpreter
# startup patches every process (including the RayGPUWorker engine processes).
apply()
