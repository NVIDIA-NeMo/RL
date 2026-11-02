# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Real vLLM wire-format parity without starting a GPU engine."""

from __future__ import annotations

import ast
import asyncio
import enum
import gc
import inspect
import itertools
import logging
import os
import runpy
import sys
import time
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, NamedTuple

import msgspec
import numpy as np
import pytest
import torch

from nemo_rl.models.generation.vllm.compact_output import (
    CompactOutputPipeline,
    NumericCaptureLogprobs,
    append_numeric_logprobs,
)
from nemo_rl.models.generation.vllm.packed_logprobs import PackedSelectedLogprobs
from nemo_rl.models.generation.vllm.utils import (
    _extract_selected_token_logprobs,
    own_routed_expert_fragments,
)

G_SERIAL = Path(
    os.environ.get(
        "NRL_TEST_VLLM_SERIAL_UTILS_SOURCE", "/tmp/nano35-vllm029-serial-utils.py"
    )
)
G_ENGINE = Path(
    os.environ.get(
        "NRL_TEST_VLLM_ENGINE_TYPES_SOURCE", "/tmp/nano35-vllm029-engine-types.py"
    )
)
G_OUTPUTS = Path(
    os.environ.get(
        "NRL_TEST_VLLM_V1_OUTPUTS_SOURCE", "/tmp/nano35-vllm029-v1-outputs.py"
    )
)
G_LOGPROBS = Path(
    os.environ.get(
        "NRL_TEST_VLLM_ENGINE_LOGPROBS_SOURCE", "/tmp/nano35-vllm029-engine-logprobs.py"
    )
)
G_FLAT = Path(
    os.environ.get(
        "NRL_TEST_VLLM_LOGPROBS_SOURCE", "/tmp/nano35-vllm-029-vllm-logprobs.py"
    )
)


def execute_nodes(path: Path, names: set[str], namespace: dict[str, Any]) -> None:
    if not path.exists():
        pytest.skip(f"Provide actual installed vLLM source at {path}")
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> Any:
    module = ModuleType("nrl_compact_wire_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = module.__dict__
    dummy = type("UnusedMultimodalType", (), {})
    namespace.update(
        msgspec=msgspec,
        msgpack=msgspec.msgpack,
        np=np,
        torch=torch,
        Any=Any,
        Mapping=Mapping,
        Sequence=Sequence,
        NamedTuple=NamedTuple,
        FINISH_REASON_STRINGS=("stop", "length", "abort", "error", "repetition"),
        enum=enum,
        time=time,
        dataclass=dataclass,
        itertools=itertools,
        isclass=inspect.isclass,
        bytestr=(bytes, bytearray, memoryview),
        PIN_MEMORY=False,
        CUSTOM_TYPE_RAW_VIEW=3,
        envs=SimpleNamespace(
            VLLM_MSGPACK_ZERO_COPY_THRESHOLD=256,
            VLLM_ALLOW_INSECURE_SERIALIZATION=False,
        ),
        MultiModalKwargsItem=dummy,
        MultiModalKwargsItems=dummy,
        PrefillStats=Any,
        SchedulerStats=Any,
        RequestSpecDecodeMetrics=Any,
        LogprobsTensors=Any,
        TokenizerLike=Any,
        logger=logging.getLogger(__name__),
        NONES=itertools.repeat(None),
        convert_ids_list_to_tokens=lambda tokenizer, ids: [str(x) for x in ids],
    )
    execute_nodes(G_OUTPUTS, {"LogprobsLists", "SamplingMaskLists"}, namespace)
    execute_nodes(
        G_SERIAL, {"UtilityResult", "MsgpackEncoder", "MsgpackDecoder"}, namespace
    )
    execute_nodes(
        G_ENGINE,
        {
            "FinishReason",
            "EngineCoreEventType",
            "EngineCoreEvent",
            "EngineCoreOutput",
            "UtilityOutput",
            "EngineCoreOutputs",
        },
        namespace,
    )
    module_name = module.__name__
    namespace.update(runpy.run_path(str(G_FLAT)))
    # runpy's namespace has its own name; dataclass below needs this registered module.
    namespace["__name__"] = module_name
    execute_nodes(G_LOGPROBS, {"LogprobsProcessor"}, namespace)
    return SimpleNamespace(**namespace)


def make_client(native: Any, *, threshold: int = 256) -> Any:
    decoder = native.MsgpackDecoder(native.EngineCoreOutputs)
    queue: asyncio.Queue = asyncio.Queue()

    async def get() -> Any:
        value = await queue.get()
        if isinstance(value, Exception):
            raise value
        return value

    return SimpleNamespace(
        decoder=decoder,
        outputs_queue=queue,
        get_output_async=get,
        encoder=native.MsgpackEncoder(size_threshold=threshold),
    )


def make_output(native: Any, *, positions: int = 8, routes: bool = True) -> Any:
    ids = np.arange(positions, dtype=np.int32).reshape(-1, 1) + 1
    probs = np.linspace(-2.0, -0.1, positions, dtype=np.float32).reshape(-1, 1)
    ranks = np.ones(positions, dtype=np.int32)
    item = native.EngineCoreOutput(
        request_id="call",
        new_token_ids=ids[:, 0].tolist(),
        new_logprobs=native.LogprobsLists(ids, probs, ranks, None),
        routed_experts=(
            np.arange(positions * 156, dtype=np.int16).reshape(positions, 26, 6).copy()
            if routes
            else None
        ),
    )
    return native.EngineCoreOutputs(outputs=[item], timestamp=12.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [0, 1_000_000])
@pytest.mark.parametrize("routes", [True, False])
async def test_native_wire_parity_and_fifo(
    native: Any, threshold: int, routes: bool
) -> None:
    client = make_client(native, threshold=threshold)
    original = client.decoder.decode
    pipeline = CompactOutputPipeline(client, max_bytes=1_000_000, max_messages=10)
    expected_messages = []
    for positions in [1, 8, 17]:
        message = make_output(native, positions=positions, routes=routes)
        frames = client.encoder.encode(message)
        expected_messages.append(original(frames))
        packet = client.decoder.decode(frames)
        assert packet is not message
        client.outputs_queue.put_nowait(packet)
    assert pipeline.stats()["decoded_messages"] == 0
    assert pipeline.stats()["queued_messages"] == 3
    for expected in expected_messages:
        result = await client.get_output_async()
        assert result.timestamp == expected.timestamp
        assert result.outputs[0].new_token_ids == expected.outputs[0].new_token_ids
        for actual, baseline in zip(
            result.outputs[0].new_logprobs[:3],
            expected.outputs[0].new_logprobs[:3],
            strict=True,
        ):
            np.testing.assert_array_equal(actual, baseline)
        if routes:
            np.testing.assert_array_equal(
                result.outputs[0].routed_experts, expected.outputs[0].routed_experts
            )
        else:
            assert result.outputs[0].routed_experts is None
    assert pipeline.stats()["decoded_messages"] == 3
    assert pipeline.stats()["queued_bytes"] == 0
    assert pipeline.stats()["queued_messages"] == 0


@pytest.mark.asyncio
async def test_no_bulk_decode_before_consumption_and_utility_bypasses_full_queue(
    native: Any,
) -> None:
    client = make_client(native)
    decoded = []
    original = client.decoder.decode

    def observe(frames: Any) -> Any:
        decoded.append(True)
        return original(frames)

    client.decoder.decode = observe
    pipeline = CompactOutputPipeline(client, max_bytes=1_000_000, max_messages=1)
    frames = client.encoder.encode(make_output(native))
    packet = client.decoder.decode(frames)
    client.outputs_queue.put_nowait(packet)
    assert not decoded
    utility = native.EngineCoreOutputs(
        utility_output=native.UtilityOutput(
            42, result=native.UtilityResult({"ok": True})
        )
    )
    result = client.decoder.decode(client.encoder.encode(utility))
    assert result.utility_output.call_id == 42
    assert result.utility_output.result.result == {"ok": True}
    assert len(decoded) == 1
    assert pipeline.stats()["queued_messages"] == 1
    result = await client.get_output_async()
    assert result.outputs[0].request_id == "call" and len(decoded) == 2


@pytest.mark.asyncio
async def test_raw_frames_keep_routes_alive_until_decode_and_publication(
    native: Any,
) -> None:
    client = make_client(native, threshold=0)
    pipeline = CompactOutputPipeline(client, max_bytes=1_000_000, max_messages=10)
    message = make_output(native)
    reference = weakref.ref(message.outputs[0].routed_experts)
    frames = client.encoder.encode(message)
    packet = client.decoder.decode(frames)
    client.outputs_queue.put_nowait(packet)
    del message, frames
    gc.collect()
    assert reference() is not None
    result = await client.get_output_async()
    assert packet.frames == ()
    own_routed_expert_fragments(result.outputs)
    gc.collect()
    assert reference() is None
    assert result.outputs[0].routed_experts.flags.owndata
    assert pipeline.stats()["queued_bytes"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["bytes", "messages"])
async def test_queue_limit_is_explicit_and_consumption_releases_capacity(
    native: Any, limit: str
) -> None:
    client = make_client(native)
    frames = client.encoder.encode(make_output(native))
    size = sum(memoryview(x).nbytes for x in frames)
    pipeline = CompactOutputPipeline(
        client,
        max_bytes=size if limit == "bytes" else size * 10,
        max_messages=1 if limit == "messages" else 10,
    )
    client.outputs_queue.put_nowait(client.decoder.decode(frames))
    with pytest.raises(BufferError, match="exceeded"):
        client.decoder.decode(frames)
    assert pipeline.stats()["limit_errors"] == 1
    await client.get_output_async()
    client.outputs_queue.put_nowait(client.decoder.decode(frames))
    await client.get_output_async()
    assert pipeline.stats()["queued_bytes"] == 0


@pytest.mark.asyncio
async def test_cancelled_wait_does_not_consume_a_future_packet(native: Any) -> None:
    client = make_client(native)
    pipeline = CompactOutputPipeline(client, max_bytes=1_000_000, max_messages=10)
    task = asyncio.create_task(client.get_output_async())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    client.outputs_queue.put_nowait(
        client.decoder.decode(client.encoder.encode(make_output(native)))
    )
    assert (await client.get_output_async()).outputs[0].request_id == "call"
    assert pipeline.stats()["queued_messages"] == 0


@pytest.mark.asyncio
async def test_malformed_bulk_payload_releases_accounting_and_frames(
    native: Any,
) -> None:
    client = make_client(native)
    pipeline = CompactOutputPipeline(client, max_bytes=1000, max_messages=10)
    # Envelope is valid but the typed EngineCoreOutput is not.
    frames = [msgspec.msgpack.encode([0, [123], None, 1.0])]
    packet = client.decoder.decode(frames)
    client.outputs_queue.put_nowait(packet)
    with pytest.raises(msgspec.ValidationError):
        await client.get_output_async()
    assert packet.frames == ()
    assert pipeline.stats()["queued_bytes"] == 0
    assert pipeline.stats()["queued_messages"] == 0


@pytest.mark.asyncio
async def test_stats_and_finished_metadata_preserved(native: Any) -> None:
    client = make_client(native)
    pipeline = CompactOutputPipeline(client, max_bytes=10000, max_messages=10)
    message = native.EngineCoreOutputs(
        scheduler_stats={"running": 3},
        timestamp=5.0,
        finished_requests={"done"},
        wave_complete=2,
        start_wave=3,
    )
    client.outputs_queue.put_nowait(
        client.decoder.decode(client.encoder.encode(message))
    )
    result = await client.get_output_async()
    assert not result.outputs
    assert result.scheduler_stats == {"running": 3}
    assert result.finished_requests == {"done"}
    assert result.wave_complete == 2 and result.start_wave == 3
    assert result.timestamp == 5.0
    assert pipeline.stats()["decoded_messages"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("overflow", [False, True])
async def test_actual_receiver_dispatches_utility_and_propagates_overflow(
    native: Any, overflow: bool
) -> None:
    source = Path("/tmp/nano35-vllm029-core-client.py")
    if not source.exists():
        pytest.skip("Provide installed vLLM core_client source")
    tree = ast.parse(source.read_text())
    client_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AsyncMPClient"
    )
    methods = [
        node
        for node in client_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"_ensure_output_queue_task", "get_output_async"}
    ]
    client_node.bases = []
    client_node.decorator_list = []
    client_node.body = methods
    namespace = dict(
        vars(native),
        asyncio=asyncio,
        weakref=weakref,
        EEP_NOTIFICATION_CALL_ID=-1,
        FT_STATUS_CALL_ID=-2,
        EngineDeadError=RuntimeError,
    )
    execute_nodes(source, {"_process_utility_output"}, namespace)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, client_node], type_ignores=[])
            ),
            str(source),
            "exec",
        ),
        namespace,
    )
    client = namespace["AsyncMPClient"]()
    socket_queue = asyncio.Queue()

    async def receive(**kwargs: Any) -> Any:
        return await socket_queue.get()

    client.resources = SimpleNamespace(
        output_queue_task=None,
        output_socket=SimpleNamespace(recv_multipart=receive),
        validate_alive=lambda frames: None,
    )
    client.outputs_queue = asyncio.Queue()
    client.utility_results = {42: asyncio.get_running_loop().create_future()}
    reply = client.utility_results[42]
    client.decoder = native.MsgpackDecoder(native.EngineCoreOutputs)
    client._format_exception = lambda error: error
    pipeline = CompactOutputPipeline(client, max_bytes=100000, max_messages=1)
    encoder = native.MsgpackEncoder()
    socket_queue.put_nowait(encoder.encode(make_output(native)))
    socket_queue.put_nowait(
        encoder.encode(
            native.EngineCoreOutputs(
                utility_output=native.UtilityOutput(
                    42, result=native.UtilityResult("ack")
                )
            )
        )
    )
    if overflow:
        socket_queue.put_nowait(encoder.encode(make_output(native)))
    client._ensure_output_queue_task()
    try:
        assert await asyncio.wait_for(reply, 1) == "ack"
        assert pipeline.stats()["decoded_messages"] == 0
        assert (await client.get_output_async()).outputs[0].request_id == "call"
        if overflow:
            with pytest.raises(BufferError):
                await client.get_output_async()
            assert pipeline.stats()["limit_errors"] == 1
    finally:
        client.resources.output_queue_task.cancel()
        await asyncio.gather(client.resources.output_queue_task, return_exceptions=True)


@pytest.mark.parametrize("positions", [0, 1, 8, 257])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numeric_append_matches_native_values_indices_ranks_and_sequential_sum(
    native: Any, positions: int, dtype: Any
) -> None:
    def processor() -> Any:
        return native.LogprobsProcessor(
            tokenizer=None,
            logprobs=native.FlatLogprobs(),
            prompt_logprobs=None,
            cumulative_logprob=0.0,
            num_logprobs=0,
            num_prompt_logprobs=None,
        )

    expected, actual = processor(), processor()
    for count in [3, positions, 2]:
        values = native.LogprobsLists(
            np.arange(count, dtype=np.int32).reshape(-1, 1),
            np.linspace(-10.0, -0.01, count, dtype=dtype).reshape(-1, 1),
            np.ones(count, dtype=np.int32),
            None,
        )
        expected._update_sample_logprobs(values)
        assert append_numeric_logprobs(actual, values) == count
        for field in [
            "start_indices",
            "end_indices",
            "token_ids",
            "logprobs",
            "ranks",
            "decoded_tokens",
        ]:
            assert getattr(actual.logprobs, field) == getattr(expected.logprobs, field)
        assert actual.cumulative_logprob == expected.cumulative_logprob


def test_numeric_bad_shape_does_not_partially_mutate(native: Any) -> None:
    processor = SimpleNamespace(logprobs=native.FlatLogprobs(), cumulative_logprob=0.0)
    bad = (
        np.ones((2, 2), dtype=np.int32),
        np.ones((2, 2), dtype=np.float32),
        np.ones(2, dtype=np.int32),
        None,
    )
    with pytest.raises(ValueError):
        append_numeric_logprobs(processor, bad)
    assert len(processor.logprobs) == 0 and processor.cumulative_logprob == 0.0


@pytest.mark.parametrize(
    "captured,flat,topk,prompt,final,enabled",
    [
        (True, True, 0, None, True, True),
        (False, True, 0, None, True, False),
        (True, False, 0, None, True, False),
        (True, True, 1, None, True, False),
        (True, True, 0, 1, True, False),
        (True, True, 0, None, False, False),
    ],
)
@pytest.mark.parametrize("packed", [False, True])
def test_numeric_hook_only_changes_eligible_capture(
    native: Any,
    monkeypatch: pytest.MonkeyPatch,
    captured: bool,
    flat: bool,
    topk: int,
    prompt: int | None,
    final: bool,
    enabled: bool,
    packed: bool,
) -> None:
    logprobs = ModuleType("vllm.logprobs")
    logprobs.FlatLogprobs = native.FlatLogprobs
    logprobs.Logprob = native.Logprob
    params = ModuleType("vllm.sampling_params")
    params.RequestOutputKind = SimpleNamespace(FINAL_ONLY="final")
    monkeypatch.setitem(sys.modules, "vllm.logprobs", logprobs)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", params)
    lp = native.LogprobsProcessor(
        tokenizer=None,
        logprobs=native.FlatLogprobs() if flat else [],
        prompt_logprobs=None,
        cumulative_logprob=0.0,
        num_logprobs=topk,
        num_prompt_logprobs=prompt,
    )
    state = SimpleNamespace(
        logprobs_processor=lp, output_kind="final" if final else "stream"
    )
    original_update = lp._update_sample_logprobs
    marker = object()
    processor = SimpleNamespace(
        request_states={"call": state}, add_request=lambda *args, **kwargs: marker
    )
    original_add = processor.add_request
    hook = NumericCaptureLogprobs(processor, lambda: captured, packed=packed)
    assert processor.add_request(SimpleNamespace(request_id="call")) is marker
    assert (lp._update_sample_logprobs != original_update) is enabled
    assert hook.stats()["requests"] == int(enabled)
    assert isinstance(lp.logprobs, PackedSelectedLogprobs) is (enabled and packed)
    assert hook.stats()["packed_requests"] == int(enabled and packed)
    if enabled:
        lists = native.LogprobsLists(
            np.array([[7]], dtype=np.int32),
            np.array([[-0.125]], dtype=np.float32),
            np.array([2], dtype=np.int32),
            None,
        )
        lp._update_sample_logprobs(lists)
        assert _extract_selected_token_logprobs([7], lp.logprobs) == [-0.125]
        assert hook.stats()["positions"] == 1
        if packed:
            assert hook.stats()["packed_live_positions"] == 1
            assert hook.stats()["packed_live_payload_bytes"] == 24

    hook.close()
    assert processor.add_request is original_add
    reference = weakref.ref(lp)
    del state, lp, original_update
    processor.request_states.clear()
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("count", [0, 1, 8, 257])
def test_packed_matches_native_append_and_frozen_prefix(
    native: Any, dtype: Any, count: int
) -> None:
    expected = native.LogprobsProcessor(
        tokenizer=None,
        logprobs=native.FlatLogprobs(),
        prompt_logprobs=None,
        cumulative_logprob=0.0,
        num_logprobs=0,
        num_prompt_logprobs=None,
    )
    packed = PackedSelectedLogprobs(native.Logprob)
    actual = SimpleNamespace(logprobs=packed, cumulative_logprob=0.0)
    for n in [3, count, 2]:
        # Noncontiguous inputs exercise bulk conversion independently of native wire layout.
        ids = np.arange(n * 2, dtype=np.int64).reshape(n, 2)[:, :1]
        values = np.linspace(-99999.0, -0.000001, n * 2, dtype=dtype).reshape(n, 2)[
            :, :1
        ]
        ranks = np.arange(n * 2, dtype=np.int64)[::2] + 1
        lists = native.LogprobsLists(ids, values, ranks, None)
        expected._update_sample_logprobs(lists)
        frozen = len(packed)
        frozen_ids = expected.logprobs.token_ids[:frozen]
        assert append_numeric_logprobs(actual, lists) == n
        assert actual.cumulative_logprob == expected.cumulative_logprob
        assert len(packed) == len(expected.logprobs)
        assert list(packed) == list(expected.logprobs)
        ids_all = expected.logprobs.token_ids
        assert _extract_selected_token_logprobs(
            ids_all, packed
        ) == _extract_selected_token_logprobs(ids_all, expected.logprobs)
        assert _extract_selected_token_logprobs(
            frozen_ids, packed, stop=frozen
        ) == _extract_selected_token_logprobs(
            frozen_ids, expected.logprobs, stop=frozen
        )
        assert packed.payload_bytes == 24 * len(packed)
        assert gc.get_referents(packed._values) == [type(packed._values)]
    for selection in [
        slice(None),
        slice(1, 4),
        slice(None, None, -1),
        slice(99, 100),
        slice(None, None, 2),
        slice(-2, None),
    ]:
        assert list(packed[selection]) == list(expected.logprobs)[selection]
    assert packed[-1] == expected.logprobs[-1]
    with pytest.raises(IndexError):
        packed[len(packed)]
    with pytest.raises(RuntimeError, match="selected token"):
        _extract_selected_token_logprobs([999] * len(packed), packed)


def test_packed_rejects_bad_inputs_without_partial_append(native: Any) -> None:
    packed = PackedSelectedLogprobs(native.Logprob)
    packed.append_arrays(
        np.array([[1]], dtype=np.int32),
        np.array([[-0.1]], dtype=np.float64),
        np.array([1], dtype=np.int32),
    )
    before = list(packed)
    invalid = [
        (np.array([[2, 3]]), np.array([[-1.0, -2.0]]), np.array([1])),
        (np.array([[2.5]]), np.array([[-1.0]]), np.array([1])),
        (np.array([[2]]), np.array([[-1.0]]), np.array([1.5])),
        (np.array([[2**63]], dtype=np.uint64), np.array([[-1.0]]), np.array([1])),
        (np.array([[2]]), np.array([[-1.0]]), np.array([2**63], dtype=np.uint64)),
    ]
    for args in invalid:
        with pytest.raises((ValueError, OverflowError)):
            packed.append_arrays(*args)
        assert list(packed) == before
    with pytest.raises(ValueError):
        packed.selected_values([1], start=0, stop=2, floor=-9999.0)


def test_packed_copies_input_buffers_and_preserves_special_floats(native: Any) -> None:
    import math

    packed = PackedSelectedLogprobs(native.Logprob)
    ids = np.array([[1], [2], [3], [4]], dtype=np.int32)
    values = np.array([[-0.0], [-np.inf], [np.inf], [np.nan]], dtype=np.float64)
    ranks = np.array([1, 2, 3, 4], dtype=np.int32)
    actual = SimpleNamespace(logprobs=packed, cumulative_logprob=0.0)
    append_numeric_logprobs(actual, native.LogprobsLists(ids, values, ranks, None))
    ids[:] = 99
    values[:] = 99
    ranks[:] = 99
    assert next(iter(packed[0])) == 1
    assert math.copysign(1.0, packed[0][1].logprob) == -1.0
    assert packed[1][2].logprob == -math.inf
    assert packed[2][3].logprob == math.inf
    assert math.isnan(packed[3][4].logprob) and math.isnan(actual.cumulative_logprob)
    assert packed[3][4].rank == 4 and packed[3][4].decoded_token is None


def test_packed_partial_append_failure_rolls_back(native: Any) -> None:
    from array import array

    class FailingValues(array):
        def append(self, value: float) -> None:
            raise MemoryError("injected allocation failure")

    packed = PackedSelectedLogprobs(native.Logprob)
    packed._values = FailingValues("d")
    with pytest.raises(MemoryError, match="injected"):
        packed.append_arrays(np.array([[7]]), np.array([[-0.1]]), np.array([2]))
    assert (
        len(packed)
        == len(packed._ids)
        == len(packed._values)
        == len(packed._ranks)
        == 0
    )


def test_native_final_output_keeps_packed_history_without_materialization(
    native: Any,
) -> None:
    source = Path("/tmp/nano35-vllm029-output-processor.py")
    tree = ast.parse(source.read_text())
    state_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RequestState"
    )
    method = next(
        node
        for node in state_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_new_completion_output"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    namespace = {
        "RequestOutputKind": SimpleNamespace(DELTA="delta"),
        "CompletionOutput": SimpleNamespace,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, method], type_ignores=[])
            ),
            str(source),
            "exec",
        ),
        namespace,
    )
    packed = PackedSelectedLogprobs(native.Logprob)
    packed.append_arrays(
        np.array([[7], [8]]), np.array([[-0.1], [-0.2]]), np.array([2, 1])
    )
    state = SimpleNamespace(
        detokenizer=SimpleNamespace(
            output_token_ids=[7, 8],
            get_next_output_text=lambda finished, delta: "response",
        ),
        logprobs_processor=SimpleNamespace(logprobs=packed, cumulative_logprob=-0.3),
        output_kind="final",
        sampling_mask_chunks=[],
        routed_experts_chunks=[],
        request_index=0,
        spec_decode_metrics=None,
    )
    output = namespace["_new_completion_output"](
        state, [8], native.FinishReason.LENGTH, None
    )
    assert output.logprobs is packed and output.token_ids == [7, 8]
    assert output.text == "response" and output.finish_reason == "length"
    assert _extract_selected_token_logprobs(output.token_ids, output.logprobs) == [
        -0.1,
        -0.2,
    ]


@pytest.mark.parametrize("enabled,numeric", [(False, True), (True, False)])
def test_packed_config_rejects_missing_prerequisites(
    enabled: bool, numeric: bool
) -> None:
    from nemo_rl.models.generation.vllm.config import validate_vllm_async_actor_settings

    with pytest.raises(ValueError, match="packed capture requires"):
        validate_vllm_async_actor_settings(
            {
                "vllm_cfg": {
                    "compact_output": {
                        "enabled": enabled,
                        "numeric_capture_logprobs": numeric,
                        "packed_capture_logprobs": True,
                    }
                }
            }
        )
