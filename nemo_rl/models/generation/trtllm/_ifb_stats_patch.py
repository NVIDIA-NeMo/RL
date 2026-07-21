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

"""Inject a *synchronous*, Ray-picklable iteration-stats fetch into TRT-LLM's
executor worker so the TRT-LLM in-flight-batching metrics logger can read
per-iteration stats via ``collective_rpc``.

Why this is needed (TRT-LLM 1.3.0rc21, ``_torch`` ``AsyncLLM`` + ``RayExecutor``):

* The ``_torch`` PyExecutor *does* produce iteration stats (``self.stats`` on
  the generation rank), but they are only reachable on the real engine worker
  (``RayGPUWorker.worker``) via ``collective_rpc("fetch_stats")`` — the
  RPC-client path used by ``LLM.get_stats()`` reaches a *different, stats-less*
  engine instance and always returns ``[]``.
* ``collective_rpc("fetch_stats")`` returns raw ``IterationStats`` C++ bindings
  that are **not picklable** across Ray (``TypeError: cannot pickle
  'tensorrt_llm.bindings.executor.IterationStats' object``).
* The built-in serialized fetches (``RpcWorkerMixin.fetch_stats_async`` /
  ``fetch_stats_wait_async``) are ``async def`` and cannot be dispatched through
  ``RayGPUWorker.call_worker_method`` (which is sync and returns the coroutine
  unawaited).

So we add a plain **sync** ``BaseWorker.fetch_stats_serialized`` that mirrors the
async variants' body (``[self._stats_serializer(s) for s in self.fetch_stats()]``)
and returns JSON strings — picklable, and callable via ``collective_rpc``.

This module is import-time idempotent: importing it applies the patch. It is
loaded inside the ``RayGPUWorker`` processes via a ``.pth`` file dropped into the
shared venv's ``site-packages`` (see ``trtllm_worker_async`` — the worker
processes do not import nemo_rl otherwise, and the venv is container-ephemeral).
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
        print(f"[ifb_stats_patch] BaseWorker import failed: {exc!r}", file=sys.stderr, flush=True)
        return False

    if getattr(BaseWorker, "fetch_stats_serialized", None) is not None:
        return True

    def fetch_stats_serialized(self):  # type: ignore[no-untyped-def]
        """Sync sibling of ``fetch_stats_async`` — fetch + serialize to JSON.

        ``fetch_stats`` returns raw (unpicklable) ``IterationStats`` records;
        ``_stats_serializer`` turns each into a JSON string that survives the
        Ray ``collective_rpc`` return trip.
        """
        return [self._stats_serializer(s) for s in self.fetch_stats()]

    BaseWorker.fetch_stats_serialized = fetch_stats_serialized
    print("[ifb_stats_patch] BaseWorker.fetch_stats_serialized installed", file=sys.stderr, flush=True)
    return True


# Apply on import so a `.pth`-driven `import ..._ifb_stats_patch` at interpreter
# startup patches every process (including the RayGPUWorker engine processes).
apply()
