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
"""Measurement-only extension of the production #5176 generation worker."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace
from typing import Any

import ray

from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


@ray.remote
class PrefixCheckpointBenchmarkWorker(
    VllmAsyncGenerationWorkerImpl
):  # pragma: no cover
    """Observe native cuts; no extra control threads, queues, fences or encoders."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._bench_lock = threading.Lock()
        self._bench_arrivals: dict[tuple[str, str], float] = {}
        self._bench_events: list[dict[str, Any]] = []
        self._bench_pending: set[asyncio.Task[Any]] = set()
        super().__init__(*args, **kwargs)

    async def setup_token_capture(self, *args: Any, **kwargs: Any) -> Any:
        result = await super().setup_token_capture(*args, **kwargs)
        if not self.is_model_owner:
            return result
        recovery = self._prefix_recovery
        original_run = recovery.run_cut
        original_steps = recovery._checkpoint_steps
        sink = recovery._parts.sink
        original_stage = sink.stage_generation_prefix_batch

        async def run(inventory: Any) -> Any:
            with self._bench_lock:
                self._bench_arrivals[
                    (inventory.checkpoint_id, inventory.inventory_digest)
                ] = time.perf_counter()
            # HTTP cancellation must not hide still-admitted native writes.
            task = asyncio.create_task(original_run(inventory))
            self._bench_pending.add(task)
            task.add_done_callback(self._bench_pending.discard)
            return await asyncio.shield(task)

        def steps(inventory: Any, *, deadline: float | None = None) -> Any:
            started = time.perf_counter()
            with self._bench_lock:
                arrived = self._bench_arrivals.pop(
                    (inventory.checkpoint_id, inventory.inventory_digest)
                )
            event: dict[str, Any] = {
                "phase": "inventory",
                "checkpoint_id": inventory.checkpoint_id,
                "calls": len(inventory.active_prefixes),
                "queue_s": started - arrived,
                "batch_size": recovery._parts.generation_prefix_batch_size,
                "prepare_wall_s": 0.0,
                "thread_cpu_s": 0.0,
                "success": False,
            }
            iterator: Any = original_steps(inventory, deadline=deadline)
            reply = None
            try:
                while True:
                    before, cpu = time.perf_counter(), time.thread_time()
                    try:
                        request = iterator.send(reply)
                    except StopIteration as finished:
                        event["success"] = True
                        return finished.value
                    finally:
                        event["prepare_wall_s"] += time.perf_counter() - before
                        event["thread_cpu_s"] += time.thread_time() - cpu
                    reply = yield request
            finally:
                iterator.close()
                event["elapsed_s"] = time.perf_counter() - started
                with self._bench_lock:
                    self._bench_events.append(event)

        def stage(records: Any, **stage_kwargs: Any) -> Any:
            started, cpu = time.perf_counter(), time.thread_time()
            event = {
                "phase": "prefix_stage_batch",
                "checkpoint_id": stage_kwargs["checkpoint_id"],
                "rows": len(records),
                "token_positions": sum(len(row.token_ids_delta) for row in records),
                "success": False,
            }
            try:
                keys = original_stage(records, **stage_kwargs)
                event["success"] = True
                return keys
            finally:
                event.update(
                    elapsed_s=time.perf_counter() - started,
                    thread_cpu_s=time.thread_time() - cpu,
                )
                with self._bench_lock:
                    self._bench_events.append(event)

        recovery.run_cut = run
        recovery._checkpoint_steps = steps
        sink.stage_generation_prefix_batch = stage
        return result

    async def benchmark_calls(self, call_ids: list[str]) -> list[str]:
        """Select admitted calls without iterating an unlocked registry."""
        return [
            identity
            for identity in call_ids
            if self._capture_registry.get_by_model_call_id(identity) is not None
        ]

    async def benchmark_batch_size(self, size: int) -> bool:
        """Change only the public runtime batch knob after previous cuts settle."""
        recovery = self._prefix_recovery
        recovery._parts = replace(recovery._parts, generation_prefix_batch_size=size)
        return True

    async def benchmark_drain(self) -> dict[str, Any]:
        """Wait for every admitted cut, including shielded pipelined writes."""
        recovery = self._prefix_recovery
        while self._bench_pending:
            await asyncio.gather(
                *(asyncio.shield(task) for task in tuple(self._bench_pending))
            )
        await asyncio.wrap_future(recovery._control_executor.submit(lambda: None))
        pipeline = recovery._pipeline
        metrics = {} if pipeline is None else pipeline.stats()
        if metrics.get("pending_inventories", 0):
            raise RuntimeError("Prefix pipeline still has admitted inventories")
        store = recovery._parts.sink._prefix_store
        writer = {} if store is None else store._dp_client.snapshot()
        with self._bench_lock:
            events, self._bench_events = self._bench_events, []
        return {
            "events": events,
            "progress_counts": recovery.counts(),
            "pipeline_cumulative": metrics,
            "writer_cumulative": writer,
        }

    async def benchmark_token_volume(self, call_ids: list[str]) -> dict[str, int]:
        """Read lengths only, without copying token histories or route arrays."""

        def sample() -> dict[str, int]:
            headers = []
            for identity in call_ids:
                state = self._capture_registry.get_by_model_call_id(identity)
                if state is None:
                    continue
                with state._lock:
                    if state._terminal_started:
                        continue
                    headers.append(
                        (
                            identity,
                            state._staged_token_count,
                            0 if state._staging_keys else len(state.prompt_token_ids),
                        )
                    )
            reader = self._progress_reader
            result = {"calls": len(headers), "generated_tokens": 0, "new_positions": 0}
            with reader._lock:
                processor = reader._client.output_processor
                request_ids = {
                    model_call_id: request_id
                    for request_id, model_call_id in reader._client.captured_request_ids().items()
                }
                for identity, staged, prompt in headers:
                    request_id = request_ids.get(identity)
                    if request_id is None:
                        continue
                    finished = reader._finished_requests.get(request_id)
                    if finished is not None:
                        engine_state = finished.request_state
                    else:
                        internal_ids = processor.external_req_ids.get(request_id)
                        if not internal_ids or len(internal_ids) != 1:
                            continue
                        engine_state = processor.request_states.get(internal_ids[0])
                    if engine_state is None or engine_state.detokenizer is None:
                        continue
                    generated = len(engine_state.detokenizer.output_token_ids)
                    if generated < staged:
                        raise ValueError(
                            "Engine output is shorter than its sealed prefix"
                        )
                    result["generated_tokens"] += generated
                    if generated:
                        result["new_positions"] += generated - staged + prompt
            return result

        return await asyncio.to_thread(sample)
