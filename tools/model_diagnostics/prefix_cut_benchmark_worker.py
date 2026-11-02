# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Instrumented production worker used only by the generation-only benchmark."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import math
import os
import sys
from typing import Any

import ray
from vllm.logprobs import FlatLogprobs

from nemo_rl.models.generation.vllm import vllm_worker_async as worker_module
from nemo_rl.models.generation.vllm.packed_logprobs import PackedSelectedLogprobs
from nemo_rl.models.generation.vllm.utils import install_owned_numpy_array_decode
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)
from tools.model_diagnostics.prefix_cut_benchmark_diagnostics import (
    BenchmarkDiagnostics,
    ObservedStateLock,
)


def configure_gil_switch_interval(interval_ms: float | None) -> float:
    """Optionally set this benchmark actor process's GIL interval, in ms.

    ``None`` leaves interpreter policy untouched. This affects every Python
    thread in this process; the driver, TQ servers, and engine-core processes
    have their own interpreters and are not changed here.
    """
    if interval_ms is not None:
        seconds = interval_ms / 1000
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("GIL switch interval must be finite and positive")
        sys.setswitchinterval(seconds)
    return sys.getswitchinterval() * 1000


def install_owned_array_decode(
    decoder: Any, diagnostics: BenchmarkDiagnostics
) -> dict[str, int]:
    """Copy NumPy payloads before queueing, preserving the live decoder object.

    vLLM's dec_hook looks up _decode_ndarray on this instance for each array.
    Copy after shaping so the returned array owns its storage and the existing
    route-publication hook does not copy it again. Tensor IPC is unchanged.
    """
    stats = install_owned_numpy_array_decode(decoder)
    original_decode = decoder.decode

    def decode_observed(frames: Any) -> Any:
        with diagnostics.phase("engine.decode_owned_arrays"):
            return original_decode(frames)

    decoder.decode = decode_observed
    diagnostics.event("owned_array_decode_enabled", tensor_ipc_unchanged=True)
    return stats


@ray.remote
class PrefixCutBenchmarkWorker(VllmAsyncGenerationWorkerImpl):  # pragma: no cover
    """Keep production behavior, adding per-loop/phase/lock/GC observations."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._bench_diagnostics: BenchmarkDiagnostics | None = None
        self._bench_engine_heartbeat: asyncio.Task[None] | None = None
        self._bench_http_heartbeat: asyncio.Task[None] | None = None
        self._bench_decoder_copy_stats: dict[str, int] = {}
        super().__init__(*args, **kwargs)

    async def post_init_async(self) -> None:
        self._bench_diagnostics = BenchmarkDiagnostics(
            os.environ["PREFIX_CUT_BENCH_OUTPUT"]
        )
        requested_ms = self.cfg.get("_benchmark_gil_switch_interval_ms")
        original_ms = sys.getswitchinterval() * 1000
        effective_ms = configure_gil_switch_interval(requested_ms)
        self._bench_diagnostics.event(
            "gil_switch_interval",
            requested_ms=requested_ms,
            original_ms=original_ms,
            effective_ms=effective_ms,
            scope="benchmark_generation_actor_process",
        )
        for name in (
            "attach_token_information_to_chat_response_choices",
            "attach_routed_experts_to_chat_response_choices",
            "model_dump_chat_response_with_dynamic_message_fields",
        ):
            self._wrap_preparation(name)
        self._wrap_preparation("own_routed_expert_fragments", phase_prefix="engine")
        if (
            os.environ.get("PREFIX_CUT_BENCH_COPY_DECODED_ARRAYS") == "1"
            and self.llm is not None
        ):
            self._bench_decoder_copy_stats = install_owned_array_decode(
                self.llm.engine_core.decoder, self._bench_diagnostics
            )
        await super().post_init_async()
        self._bench_engine_heartbeat = asyncio.create_task(
            self._bench_diagnostics.heartbeat("engine")
        )

    def _wrap_preparation(self, name: str, *, phase_prefix: str = "http") -> None:
        original = getattr(worker_module, name)
        diagnostics = self._bench_diagnostics
        assert diagnostics is not None

        @functools.wraps(original)
        def observed(*args: Any, **kwargs: Any) -> Any:
            with diagnostics.phase(f"{phase_prefix}.{name}"):
                return original(*args, **kwargs)

        setattr(worker_module, name, observed)

    def _setup_vllm_openai_api_server(self, app: Any) -> Any:
        app = super()._setup_vllm_openai_api_server(app)

        @contextlib.asynccontextmanager
        async def lifespan(application: Any) -> Any:
            async with original_lifespan(application) as state:
                assert self._bench_diagnostics is not None
                self._bench_http_heartbeat = asyncio.create_task(
                    self._bench_diagnostics.heartbeat("http")
                )
                try:
                    yield state
                finally:
                    self._bench_http_heartbeat.cancel()
                    await asyncio.gather(
                        self._bench_http_heartbeat, return_exceptions=True
                    )

        original_lifespan = app.router.lifespan_context
        app.router.lifespan_context = lifespan
        return app

    def _begin_request_capture(self, *args: Any, **kwargs: Any) -> None:
        super()._begin_request_capture(*args, **kwargs)
        request = args[0] if args else kwargs["request"]
        state = self._get_request_capture(request)
        if state is not None:
            assert self._bench_diagnostics is not None
            state.lock = ObservedStateLock(state.lock, self._bench_diagnostics)

    def _observe_request_capture(self, request: Any, request_output: Any) -> None:
        assert self._bench_diagnostics is not None
        with self._bench_diagnostics.phase("http.observe_final_capture"):
            return super()._observe_request_capture(request, request_output)

    async def benchmark_status(self) -> dict[str, Any]:
        assert self._bench_diagnostics is not None
        with self._capture_registry_lock:
            active_calls = len(self._capture_calls_by_model_call_id)
        # Diagnostic snapshots only. Admission ownership stays on the HTTP loop.
        result = self._bench_diagnostics.summary()
        result.update(
            gil_switch_interval_ms=sys.getswitchinterval() * 1000,
            active_calls=active_calls,
            active_inventories=len(self._generation_cut_control._active),
            pending_inventories=len(self._generation_cut_control._pending),
            slots=self._generation_cut_control._max_workers,
            decoder_copy=dict(self._bench_decoder_copy_stats),
            output_queue_depth=self.llm.engine_core.outputs_queue.qsize(),
            compact_output=(
                self._compact_output_pipeline.stats()
                if self._compact_output_pipeline is not None
                else {}
            ),
            numeric_capture=(
                self._http_engine_client._numeric_capture_hook.stats()
                if self._http_engine_client._numeric_capture_hook is not None
                else {}
            ),
            inventory_grouping=(
                self._generation_cut_grouper.stats()
                if self._generation_cut_grouper is not None
                else {}
            ),
            write_batching=(
                self._generation_prefix_write_batcher.stats()
                if self._generation_prefix_write_batcher is not None
                else {}
            ),
        )
        return result

    async def benchmark_active_calls(self) -> list[str]:
        with self._capture_registry_lock:
            return list(self._capture_calls_by_model_call_id)

    async def benchmark_heap_summary(self) -> dict[str, Any]:
        """Collect heap and engine retention evidence outside snapshot windows."""
        assert self._bench_diagnostics is not None
        # Read engine-owned structures on their owner loop, before any await.
        states = self.llm.output_processor.request_states.values()
        retained = {
            "request_states": len(self.llm.output_processor.request_states),
            "output_queue_depth": self.llm.engine_core.outputs_queue.qsize(),
            "detokenizer_output_tokens": 0,
            "routed_expert_fragments": 0,
            "flat_logprob_positions": 0,
            "flat_logprob_values": 0,
        }
        for state in states:
            if state.detokenizer is not None:
                retained["detokenizer_output_tokens"] += len(
                    state.detokenizer.output_token_ids
                )
            retained["routed_expert_fragments"] += len(state.routed_experts_chunks)
            processor = state.logprobs_processor
            if processor is not None and isinstance(
                processor.logprobs, PackedSelectedLogprobs
            ):
                retained["flat_logprob_positions"] += len(processor.logprobs)
                retained["flat_logprob_values"] += len(processor.logprobs)
            elif processor is not None and isinstance(processor.logprobs, FlatLogprobs):
                retained["flat_logprob_positions"] += len(
                    processor.logprobs.start_indices
                )
                retained["flat_logprob_values"] += len(processor.logprobs.logprobs)
        with self._capture_registry_lock:
            retained["active_capture_calls"] = len(self._capture_calls)
            retained["completed_capture_calls"] = len(self._completed_capture_calls)
        result = await asyncio.to_thread(self._bench_diagnostics.heap_summary)
        result["engine_retained"] = retained
        return result

    async def benchmark_begin_cut_window(self) -> bool:
        assert self._bench_diagnostics is not None
        self._bench_diagnostics.event("cut_window_started", native_timer_enabled=False)
        return True

    async def benchmark_end_cut_window(self) -> bool:
        assert self._bench_diagnostics is not None
        self._bench_diagnostics.event("cut_window_stopped", native_timer_enabled=False)
        return True

    async def shutdown(self) -> Any:
        try:
            return await super().shutdown()
        finally:
            if self._bench_engine_heartbeat is not None:
                self._bench_engine_heartbeat.cancel()
                await asyncio.gather(
                    self._bench_engine_heartbeat, return_exceptions=True
                )
            if self._bench_diagnostics is not None:
                await asyncio.to_thread(self._bench_diagnostics.close)
