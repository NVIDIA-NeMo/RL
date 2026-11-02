# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Keep generation output payloads encoded until their consumer is ready.

The vLLM 0.29 AsyncMPClient receiver still owns the socket and utility replies.
Only generation data is represented by a small packet in its existing queue.
The original typed decoder is used at consumption, preserving wire semantics.
"""

from __future__ import annotations

import functools
import importlib.metadata
import time
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import msgspec

from nemo_rl.models.generation.vllm.packed_logprobs import PackedSelectedLogprobs


class _Envelope(msgspec.Struct, array_like=True, gc=False):
    """vLLM 0.29 EngineCoreOutputs layout, with nested objects left raw."""

    engine_index: int = 0
    outputs: msgspec.Raw = msgspec.Raw(b"\x90")
    scheduler_stats: msgspec.Raw = msgspec.Raw(b"\xc0")
    timestamp: float = 0.0
    utility_output: msgspec.Raw = msgspec.Raw(b"\xc0")
    finished_requests: msgspec.Raw = msgspec.Raw(b"\xc0")
    wave_complete: int | None = None
    start_wave: int | None = None


def _is_nil(value: msgspec.Raw) -> bool:
    return len(value) == 1 and memoryview(value)[0] == 0xC0


@dataclass(slots=True)
class _Packet:
    frames: tuple[Any, ...]
    size: int
    engine_index: int
    owner: CompactOutputPipeline
    # The original receiver only tests these fields before queueing.
    outputs: bool = True
    utility_output: None = None
    scheduler_stats: None = None


class CompactOutputPipeline:
    """Install before requests, on one AsyncMPClient instance.

    Bounds are hard safety limits, not a promise of adequate throughput. An
    overflow becomes an explicit receiver error; no output is silently dropped
    or acknowledged. Utilities bypass data queue accounting and decode eagerly.
    """

    def __init__(self, client: Any, *, max_bytes: int, max_messages: int) -> None:
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("compact output max_bytes must be positive")
        if type(max_messages) is not int or max_messages < 1:
            raise ValueError("compact output max_messages must be positive")
        if not client.outputs_queue.empty():
            raise RuntimeError("Install compact outputs before the output queue fills")
        self._client = client
        self._decode = client.decoder.decode
        self._get = client.get_output_async
        self._header = msgspec.msgpack.Decoder(_Envelope)
        self._max_bytes = max_bytes
        self._max_messages = max_messages
        self._stats: dict[str, int | float] = {
            "queued_bytes": 0,
            "queued_messages": 0,
            "peak_queued_bytes": 0,
            "peak_queued_messages": 0,
            "received_messages": 0,
            "decoded_messages": 0,
            "eager_control_messages": 0,
            "limit_errors": 0,
            "header_wall_s": 0.0,
            "decode_wall_s": 0.0,
            "decode_thread_cpu_s": 0.0,
        }
        client.decoder.decode = self.decode_envelope
        client.get_output_async = self.get_output_async

    def stats(self) -> dict[str, int | float]:
        """Read on the engine owner loop; no heap traversal or payload copies."""
        return dict(self._stats)

    def decode_envelope(self, frames: Sequence[Any]) -> Any:
        """Leave bulk fields raw; original vLLM code handles control replies."""
        started = time.perf_counter()
        header = self._header.decode(frames[0])
        self._stats["header_wall_s"] += time.perf_counter() - started
        self._stats["received_messages"] += 1
        empty_outputs = (
            len(header.outputs) == 1 and memoryview(header.outputs)[0] == 0x90
        )
        if not _is_nil(header.utility_output) or (
            empty_outputs and _is_nil(header.scheduler_stats)
        ):
            self._stats["eager_control_messages"] += 1
            return self._decode(frames)
        size = sum(memoryview(frame).nbytes for frame in frames)
        if (
            self._stats["queued_bytes"] + size > self._max_bytes
            or self._stats["queued_messages"] >= self._max_messages
        ):
            self._stats["limit_errors"] += 1
            raise BufferError(
                "Compact output queue exceeded its configured byte/message limit: "
                f"queued_bytes={self._stats['queued_bytes']}, incoming_bytes={size}, "
                f"queued_messages={self._stats['queued_messages']}"
            )
        packet = _Packet(tuple(frames), size, header.engine_index, self)
        self._stats["queued_bytes"] += size
        self._stats["queued_messages"] += 1
        self._stats["peak_queued_bytes"] = max(
            self._stats["peak_queued_bytes"], self._stats["queued_bytes"]
        )
        self._stats["peak_queued_messages"] = max(
            self._stats["peak_queued_messages"], self._stats["queued_messages"]
        )
        return packet

    async def get_output_async(self) -> Any:
        output = await self._get()
        if not isinstance(output, _Packet):
            return output
        if output.owner is not self:
            raise RuntimeError("Compact output packet belongs to another client")
        self._stats["queued_bytes"] -= output.size
        self._stats["queued_messages"] -= 1
        started, cpu_started = time.perf_counter(), time.thread_time()
        try:
            result = self._decode(output.frames)
            self._stats["decoded_messages"] += 1
            return result
        finally:
            self._stats["decode_wall_s"] += time.perf_counter() - started
            self._stats["decode_thread_cpu_s"] += time.thread_time() - cpu_started
            output.frames = ()


def install_compact_output_pipeline(
    client: Any, *, max_bytes: int, max_messages: int
) -> CompactOutputPipeline:
    """Guard the exact upstream protocol and client implementation at startup."""
    # vLLM is optional outside the generation worker's environment.
    from vllm.v1.engine.core_client import AsyncMPClient

    if importlib.metadata.version("vllm").split("+")[0] != "0.29.0":
        raise RuntimeError("Compact output decoding is verified on vLLM 0.29.0")
    if type(client) is not AsyncMPClient:
        raise RuntimeError("Compact outputs currently require the DP=1 AsyncMPClient")
    return CompactOutputPipeline(client, max_bytes=max_bytes, max_messages=max_messages)


def append_numeric_logprobs(processor: Any, lists: Any) -> int:
    """Append selected-token arrays without constructing decoded token strings.

    Restricted to flat sample logprobs with top_logprobs=0. Cumulative sums use
    the same sequential Python float additions as upstream, not a vector sum.
    Normal response-text detokenization is independent and remains unchanged.
    """
    token_ids, logprobs, ranks, _ = lists
    if isinstance(processor.logprobs, PackedSelectedLogprobs):
        target = processor.logprobs
        before = len(target)
        target.append_arrays(token_ids, logprobs, ranks)
        processor.cumulative_logprob = target.cumulative_after(
            processor.cumulative_logprob, start=before
        )
        return len(target) - before
    if (
        token_ids.ndim != 2
        or logprobs.shape != token_ids.shape
        or token_ids.shape[1] != 1
        or ranks.shape != (token_ids.shape[0],)
    ):
        raise ValueError("Expected one selected token and rank per logprob position")
    count = token_ids.shape[0]
    ids = token_ids[:, 0].tolist()
    values = logprobs[:, 0].tolist()
    rank_values = ranks.tolist()
    target = processor.logprobs
    start = len(target.token_ids)
    target.start_indices.extend(range(start, start + count))
    target.end_indices.extend(range(start + 1, start + count + 1))
    target.token_ids.extend(ids)
    target.logprobs.extend(values)
    target.ranks.extend(rank_values)
    target.decoded_tokens.extend([None] * count)
    for value in values:
        processor.cumulative_logprob += value
    return count


class NumericCaptureLogprobs:
    """Apply the numeric path only when adding a captured request on its loop."""

    def __init__(
        self, processor: Any, is_captured: Callable[[], bool], *, packed: bool = False
    ) -> None:
        # Optional GPU dependency: only installed in the vLLM worker.
        from vllm.logprobs import FlatLogprobs, Logprob
        from vllm.sampling_params import RequestOutputKind

        self._processor = processor
        self._original = processor.add_request
        self._stats = {"requests": 0, "positions": 0, "packed_requests": 0}
        self._packed_states: weakref.WeakSet[PackedSelectedLogprobs] = weakref.WeakSet()
        self._closed = False

        @functools.wraps(self._original)
        def add_request(request: Any, *args: Any, **kwargs: Any) -> Any:
            result = self._original(request, *args, **kwargs)
            if not is_captured():
                return result
            state = processor.request_states[request.request_id]
            lp = state.logprobs_processor
            if (
                lp is None
                or not isinstance(lp.logprobs, FlatLogprobs)
                or lp.num_logprobs != 0
                or lp.num_prompt_logprobs is not None
                or state.output_kind != RequestOutputKind.FINAL_ONLY
            ):
                return result
            if packed:
                if len(lp.logprobs):
                    raise RuntimeError(
                        "Packed capture must be installed before the first token"
                    )
                lp.logprobs = PackedSelectedLogprobs(Logprob)
                self._packed_states.add(lp.logprobs)
                self._stats["packed_requests"] += 1
            reference = weakref.ref(lp)

            def numeric_update(lists: Any) -> None:
                current = reference()
                if current is None:
                    raise RuntimeError("Captured logprob processor has been released")
                self._stats["positions"] += append_numeric_logprobs(current, lists)

            lp._update_sample_logprobs = numeric_update
            self._stats["requests"] += 1
            return result

        self._wrapper = add_request
        processor.add_request = add_request

    def stats(self) -> dict[str, int]:
        result = dict(self._stats)
        result["packed_live_requests"] = len(self._packed_states)
        result["packed_live_positions"] = sum(
            len(state) for state in self._packed_states
        )
        result["packed_live_payload_bytes"] = sum(
            state.payload_bytes for state in self._packed_states
        )
        return result

    def close(self) -> None:
        if not self._closed and self._processor.add_request is self._wrapper:
            self._processor.add_request = self._original
        self._closed = True
