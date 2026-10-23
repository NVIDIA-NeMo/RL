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

"""Bounded owner-loop observations; no vLLM, NumPy or Torch import is needed.

The caller enables this only while telemetry metrics export. Wrappers call the
original vLLM methods exactly once and return their original objects. A shared,
reference-counted completion wrapper observes only the processor currently in
the context, so other engines are unaffected. No route arrays are retained.
Terminal-output timing includes vLLM's route concatenation, text selection and
output-object construction; it is not an isolated np.concatenate measurement.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

logger = logging.getLogger(__name__)
_REPORT_INTERVAL_S = 30.0
_SLOW_OPERATION_S = 1.0
_SLOW_LOG_INTERVAL_S = 10.0
_LOOP_SAMPLE_INTERVAL_S = 1.0
_ACTIVE: ContextVar[EngineLoopDiagnostics | None] = ContextVar(
    "vllm_output_diagnostics", default=None
)


@dataclass
class _CompletionPatch:
    original: Callable[..., Any]
    wrapper: Callable[..., Any]
    users: int = 1


_PATCHES: WeakKeyDictionary[type, _CompletionPatch] = WeakKeyDictionary()
_PATCH_LOCK = threading.Lock()


def _install_completion_patch(state_type: type) -> None:
    with _PATCH_LOCK:
        existing = _PATCHES.get(state_type)
        if existing is not None:
            existing.users += 1
            return
        original = state_type._new_completion_output

        @functools.wraps(original)
        def completion(state: Any, *args: Any, **kwargs: Any) -> Any:
            observer = _ACTIVE.get()
            if observer is None:
                return original(state, *args, **kwargs)
            return observer.completion_output(original, state, *args, **kwargs)

        state_type._new_completion_output = completion
        _PATCHES[state_type] = _CompletionPatch(original, completion)


def _remove_completion_patch(state_type: type) -> None:
    with _PATCH_LOCK:
        patch = _PATCHES.get(state_type)
        if patch is None:
            return
        patch.users -= 1
        if patch.users:
            return
        if state_type._new_completion_output is patch.wrapper:
            state_type._new_completion_output = patch.original
        del _PATCHES[state_type]


def _log(event: str, values: dict[str, Any]) -> None:
    """Optional logging must not change output processing or its exceptions."""
    try:
        logger.info("%s %s", event, json.dumps(values, sort_keys=True))
    except Exception:
        # A broken handler is not grounds to fail generation.
        pass


class EngineLoopDiagnostics:
    """Observe the actual engine-owner loop, not the HTTP-serving thread.

    All observation state is updated on the owner loop. No per-request records,
    identity labels, background CPU threads or per-token log messages are added.
    Install/close must run on that loop; close is idempotent.
    """

    def __init__(
        self, processor: Any, state_type: type, *, chunk_size: int, torch_threads: int
    ):
        self.processor = processor
        self.state_type = state_type
        self.chunk_size = chunk_size
        self.torch_threads = torch_threads
        self._original = processor.process_outputs
        self._wrapper: Callable[..., Any] | None = None
        self._task: asyncio.Task | None = None
        self._closed = True
        self._window_started = time.time()
        self._phases: dict[str, dict[str, float]] = {}
        self._counts: dict[str, int] = {}
        self._last_slow: dict[str, float] = {}
        self._warned = False

    def _diagnostic_error(self, operation: str) -> None:
        if not self._warned:
            self._warned = True
            _log("vllm_engine_probe_error", dict(operation=operation, pid=os.getpid()))

    def _count(self, name: str, value: int) -> None:
        self._counts[name] = self._counts.get(name, 0) + value

    def _record(
        self, name: str, wall_s: float, cpu_s: float, error: bool = False
    ) -> None:
        total = self._phases.setdefault(
            name, dict(calls=0, wall_s=0, thread_cpu_s=0, max_wall_s=0, errors=0)
        )
        total["calls"] += 1
        total["wall_s"] += wall_s
        total["thread_cpu_s"] += cpu_s
        total["max_wall_s"] = max(total["max_wall_s"], wall_s)
        total["errors"] += int(error)

    def _slow(self, name: str, wall_s: float, **attributes: Any) -> None:
        now = time.monotonic()
        if (
            wall_s < _SLOW_OPERATION_S
            or now - self._last_slow.get(name, float("-inf")) < _SLOW_LOG_INTERVAL_S
        ):
            return
        self._last_slow[name] = now
        _log(
            "vllm_engine_slow",
            dict(
                phase=name,
                wall_s=wall_s,
                unix_s=time.time(),
                pid=os.getpid(),
                **attributes,
            ),
        )

    def completion_output(
        self, original: Callable[..., Any], state: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Time finished output construction, including cumulative route concat."""
        finish_reason = kwargs.get("finish_reason", args[1] if len(args) > 1 else None)
        if finish_reason is None:
            return original(state, *args, **kwargs)
        started, cpu_started = time.perf_counter(), time.thread_time()
        error = True
        result = None
        try:
            result = original(state, *args, **kwargs)
            error = False
            return result
        finally:
            elapsed = time.perf_counter() - started
            cpu_s = time.thread_time() - cpu_started
            try:
                self._record("terminal_output_build", elapsed, cpu_s, error)
                self._count("finished_outputs", 1)
                chunk_count = len(state.routed_experts_chunks)
                self._count("terminal_route_chunks", chunk_count)
                # nbytes is metadata-only; don't scan or copy route history.
                routes = None if result is None else result.routed_experts
                route_bytes = 0 if routes is None else int(routes.nbytes)
                self._count("terminal_route_bytes", route_bytes)
                self._slow(
                    "terminal_output_build",
                    elapsed,
                    route_chunks=chunk_count,
                    route_bytes=route_bytes,
                )
            except Exception:
                self._diagnostic_error("terminal_output_build")

    def start(self) -> None:
        """Attach reversible wrappers and a one-second owner-loop lag probe."""
        if not self._closed:
            return
        loop = asyncio.get_running_loop()
        _install_completion_patch(self.state_type)

        @functools.wraps(self._original)
        def process_outputs(*args: Any, **kwargs: Any) -> Any:
            batch = args[0] if args else kwargs.get("engine_core_outputs", ())
            token = _ACTIVE.set(self)
            started, cpu_started = time.perf_counter(), time.thread_time()
            error = True
            try:
                result = self._original(*args, **kwargs)
                error = False
                return result
            finally:
                elapsed = time.perf_counter() - started
                _ACTIVE.reset(token)
                try:
                    self._record(
                        "process_outputs",
                        elapsed,
                        time.thread_time() - cpu_started,
                        error,
                    )
                    self._count("output_chunks", 1)
                    self._count("request_outputs", len(batch))
                    self._slow("process_outputs", elapsed, request_outputs=len(batch))
                except Exception:
                    self._diagnostic_error("process_outputs")

        self._wrapper = process_outputs
        try:
            self.processor.process_outputs = process_outputs
            self._task = loop.create_task(
                self._monitor_loop(), name="nrl-vllm-engine-probe"
            )
        except Exception:
            if self.processor.process_outputs is process_outputs:
                self.processor.process_outputs = self._original
            _remove_completion_patch(self.state_type)
            raise
        self._closed = False
        try:
            _log(
                "vllm_engine_probe_start",
                dict(
                    pid=os.getpid(),
                    owner_thread_id=threading.get_ident(),
                    output_proc_chunk_size=self.chunk_size,
                    torch_num_threads=self.torch_threads,
                    omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                    cpu_affinity_count=len(os.sched_getaffinity(0)),
                ),
            )
        except Exception:
            self._diagnostic_error("startup_metadata")

    async def _monitor_loop(self) -> None:
        loop = asyncio.get_running_loop()
        last_report = loop.time()
        while True:
            expected_at = loop.time() + _LOOP_SAMPLE_INTERVAL_S
            await asyncio.sleep(_LOOP_SAMPLE_INTERVAL_S)
            lag = max(0.0, loop.time() - expected_at)
            try:
                self._record("owner_loop_lag", lag, 0.0)
                self._slow("owner_loop_lag", lag)
                if loop.time() - last_report >= _REPORT_INTERVAL_S:
                    self._report("periodic")
                    last_report = loop.time()
            except Exception:
                self._diagnostic_error("owner_loop_lag")

    def _report(self, reason: str) -> None:
        now = time.time()
        _log(
            "vllm_engine_probe",
            dict(
                reason=reason,
                pid=os.getpid(),
                started_unix_s=self._window_started,
                finished_unix_s=now,
                counts=dict(self._counts),
                phases=self._phases,
            ),
        )
        self._window_started = now
        self._phases = {}
        self._counts = {}

    async def aclose(self) -> None:
        """Stop the loop probe and restore only wrappers this observer owns."""
        if self._closed:
            return
        self._closed = True
        if self.processor.process_outputs is self._wrapper:
            self.processor.process_outputs = self._original
        _remove_completion_patch(self.state_type)
        if self._task is not None:
            self._task.cancel()
            await asyncio.wait({self._task})
            self._task = None
        self._report("shutdown")
