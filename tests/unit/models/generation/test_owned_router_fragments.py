# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Owned router fragments release decoded buffers before append-only publication."""

import ast
import gc
import logging
import os
import threading
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from nemo_rl.models.generation.vllm.config import validate_vllm_async_actor_settings
from nemo_rl.models.generation.vllm.utils import (
    install_owned_numpy_array_decode,
    own_routed_expert_fragments,
)
from nemo_rl.utils.routed_experts_codec import encode_routed_experts

_WORKER = (
    Path(__file__).resolve().parents[4]
    / "nemo_rl/models/generation/vllm/vllm_worker_async.py"
)


@pytest.fixture
def installed_hook():
    """Execute the production publication wrapper without starting a GPU actor."""
    tree = ast.parse(_WORKER.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "enable_token_snapshots"
    )
    namespace = {
        "Any": Any,
        "own_routed_expert_fragments": own_routed_expert_fragments,
        "LOGGER": logging.getLogger(__name__),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(_WORKER), "exec"),
        namespace,
    )

    def install(original):
        processor = SimpleNamespace(process_outputs=original)
        client = SimpleNamespace(
            _token_snapshots_enabled=False,
            _token_snapshot_lock=threading.RLock(),
            _engine_client=SimpleNamespace(output_processor=processor),
            check_route_snapshot_support=lambda: None,
            _direct_route_snapshot_enabled=lambda: True,
        )
        namespace["enable_token_snapshots"](client)
        return client, processor, namespace

    return install


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32])
@pytest.mark.parametrize("rows", [0, 1, 32, 1024])
def test_detach_preserves_dtype_shape_route_envelope_and_token_references(dtype, rows):
    backing = np.arange(max(1, rows) * 26 * 6, dtype=np.int32).astype(dtype)
    raw = np.frombuffer(memoryview(backing), dtype=dtype)[: rows * 26 * 6].reshape(
        rows, 26, 6
    )
    token_ids = [1, 2]
    logprobs = [-0.25, -0.5]
    output = SimpleNamespace(
        routed_experts=raw, new_token_ids=token_ids, logprobs=logprobs
    )
    before = encode_routed_experts(torch.from_numpy(raw).to(torch.int16))
    own_routed_expert_fragments([output])
    assert output.routed_experts.flags.owndata
    assert output.routed_experts.base is None
    assert output.routed_experts.dtype == dtype
    assert np.array_equal(output.routed_experts, raw)
    assert output.new_token_ids is token_ids and output.logprobs is logprobs
    assert (
        encode_routed_experts(torch.from_numpy(output.routed_experts).to(torch.int16))
        == before
    )
    owned = output.routed_experts
    own_routed_expert_fragments([output])
    assert output.routed_experts is owned


def test_original_buffer_can_be_released_while_history_keeps_owned_routes():
    backing = np.arange(312, dtype=np.uint8)
    reference = weakref.ref(backing)
    view = np.frombuffer(memoryview(backing), dtype=np.uint8).reshape(2, 26, 6)
    assert gc.is_tracked(view.base.base)
    output = SimpleNamespace(routed_experts=view)
    own_routed_expert_fragments([output])
    history = [output.routed_experts]
    del view, backing
    gc.collect()
    assert reference() is None
    assert history[0].base is None
    assert not gc.is_tracked(history[0])


def test_owned_arrays_and_absent_routes_are_kept_as_is():
    owned = np.zeros((1, 26, 6), dtype=np.uint8)
    outputs = [
        SimpleNamespace(routed_experts=owned),
        SimpleNamespace(routed_experts=None),
    ]
    own_routed_expert_fragments(outputs)
    assert outputs[0].routed_experts is owned
    assert outputs[1].routed_experts is None


def test_noncontiguous_fragment_is_detached_without_changing_order():
    raw = np.arange(4 * 26 * 12, dtype=np.int16).reshape(4, 26, 12)[:, :, ::2]
    output = SimpleNamespace(routed_experts=raw)
    own_routed_expert_fragments([output])
    assert output.routed_experts.flags.owndata
    assert output.routed_experts.flags.c_contiguous
    assert np.array_equal(output.routed_experts, raw)


def test_copies_complete_before_publication_lock_and_preserve_call_contract(
    installed_hook,
):
    copied, published = threading.Event(), threading.Event()
    timestamps, stats = object(), object()
    token_ids = [4]
    raw = np.frombuffer(bytes(156), dtype=np.uint8).reshape(1, 26, 6)
    batch = [SimpleNamespace(routed_experts=raw, new_token_ids=token_ids)]
    result = object()

    def publish(outputs, engine_core_timestamp=None, iteration_stats=None):
        assert outputs is batch
        assert engine_core_timestamp is timestamps and iteration_stats is stats
        assert outputs[0].routed_experts.flags.owndata
        assert outputs[0].new_token_ids is token_ids
        published.set()
        return result

    client, processor, namespace = installed_hook(publish)

    def copying(outputs):
        own_routed_expert_fragments(outputs)
        copied.set()

    namespace["own_routed_expert_fragments"] = copying
    results = []
    failures = []

    def writer():
        try:
            results.append(
                processor.process_outputs(batch, timestamps, iteration_stats=stats)
            )
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=writer, daemon=True)
    with client._token_snapshot_lock:
        thread.start()
        assert copied.wait(3), "Route copying blocked on the publication lock"
        assert not published.is_set()
    thread.join(3)
    assert not thread.is_alive() and not failures
    assert results == [result]


def test_published_history_keeps_identity_when_later_batch_arrives(installed_hook):
    history = []

    def publish(outputs, **kwargs):
        history.extend(
            output.routed_experts
            for output in outputs
            if output.routed_experts is not None
        )

    _, processor, _ = installed_hook(publish)
    first = SimpleNamespace(
        routed_experts=np.frombuffer(bytes(156), dtype=np.uint8).reshape(1, 26, 6)
    )
    processor.process_outputs(engine_core_outputs=[first])
    old_piece = history[0]
    second = SimpleNamespace(
        routed_experts=np.frombuffer(bytes([1]) * 156, dtype=np.uint8).reshape(1, 26, 6)
    )
    processor.process_outputs([second])
    assert history[0] is old_piece and history[1] is second.routed_experts
    assert old_piece.sum() == 0 and history[1].sum() == 156


def test_copy_failure_does_not_publish_or_leave_cut_lock_held(installed_hook):
    class BrokenArray(np.ndarray):
        def copy(self, order="C"):
            raise MemoryError("copy failed")

    calls = []
    client, processor, _ = installed_hook(lambda outputs: calls.append(outputs))
    raw = np.zeros((1, 26, 6), dtype=np.uint8).view(BrokenArray)
    with pytest.raises(MemoryError, match="copy failed"):
        processor.process_outputs([SimpleNamespace(routed_experts=raw)])
    assert not calls
    assert client._token_snapshot_lock.acquire(blocking=False)
    client._token_snapshot_lock.release()


def test_real_vllm_decoded_inline_view_loses_buffer_chain():
    path = os.environ.get("NRL_TEST_VLLM_SERIAL_UTILS_SOURCE")
    if path is None:
        module = pytest.importorskip("vllm.v1.serial_utils")
        path = module.__file__
    tree = ast.parse(Path(path).read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_decode_ndarray"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    namespace = {"np": np}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, method], type_ignores=[])
            ),
            path,
            "exec",
        ),
        namespace,
    )
    backing = np.arange(156, dtype=np.uint8)
    decoded = namespace["_decode_ndarray"](
        SimpleNamespace(aux_buffers=[], share_mem=True),
        ("|u1", (1, 26, 6), memoryview(backing)),
    )
    assert decoded.base is not None
    output = SimpleNamespace(routed_experts=decoded)
    own_routed_expert_fragments([output])
    assert output.routed_experts.base is None
    assert np.array_equal(output.routed_experts, decoded)
    backing[:] = 0
    assert output.routed_experts.sum() == sum(range(156))


@pytest.fixture
def owned_array_decoder() -> Any:
    """Use the installed vLLM ndarray decoder with the benchmark's live hook."""
    serial_path = os.environ.get("NRL_TEST_VLLM_SERIAL_UTILS_SOURCE")
    if serial_path is None:
        module = pytest.importorskip("vllm.v1.serial_utils")
        serial_path = module.__file__
    tree = ast.parse(Path(serial_path).read_text())
    decode_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_decode_ndarray"
    )
    source = (
        _WORKER.parents[4] / "tools/model_diagnostics/prefix_cut_benchmark_worker.py"
    )
    hook = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "install_owned_array_decode"
    )
    namespace = {
        "np": np,
        "Any": Any,
        "BenchmarkDiagnostics": Any,
        "install_owned_numpy_array_decode": install_owned_numpy_array_decode,
    }
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, decode_method, hook], type_ignores=[])
            ),
            str(source),
            "exec",
        ),
        namespace,
    )

    class Decoder:
        share_mem = True
        _decode_ndarray = namespace["_decode_ndarray"]

        def __init__(self) -> None:
            self.aux_buffers: list[Any] = []

        def decode(self, encoded: Any) -> np.ndarray:
            # Native dec_hook also resolves _decode_ndarray on the live instance.
            return self._decode_ndarray(encoded)

    return Decoder, namespace["install_owned_array_decode"]


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32, np.float32])
@pytest.mark.parametrize("shape", [(0, 26, 6), (1, 26, 6), (32, 1)])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_decoder_copy_releases_buffers_before_queueing_without_router_recopy(
    owned_array_decoder: Any, dtype: Any, shape: tuple[int, ...], auxiliary: bool
) -> None:
    decoder_type, install = owned_array_decoder
    decoder = decoder_type()
    events = []
    diagnostics = SimpleNamespace(
        phase=lambda name: nullcontext(),
        event=lambda name, **fields: events.append((name, fields)),
    )
    stats = install(decoder, diagnostics)
    backing = np.arange(int(np.prod(shape)), dtype=np.int32).astype(dtype)
    backing_ref = weakref.ref(backing)
    expected = backing.reshape(shape).copy()
    if auxiliary:
        decoder.aux_buffers = [memoryview(backing)]
        payload = (backing.dtype.str, shape, 0)
    else:
        payload = (backing.dtype.str, shape, memoryview(backing))
    queued = decoder.decode(payload)
    assert queued.flags.owndata and queued.base is None
    assert queued.dtype == expected.dtype and queued.shape == expected.shape
    assert queued.tobytes() == expected.tobytes()
    assert stats == {"arrays": 1, "bytes": expected.nbytes}
    output = SimpleNamespace(routed_experts=queued)
    own_routed_expert_fragments([output])
    assert output.routed_experts is queued
    decoder.aux_buffers = []
    del payload, backing
    gc.collect()
    assert backing_ref() is None
    assert queued.tobytes() == expected.tobytes()
    assert events == [("owned_array_decode_enabled", {"tensor_ipc_unchanged": True})]


def test_decoder_copy_rejects_an_already_copying_decoder(
    owned_array_decoder: Any,
) -> None:
    decoder_type, install = owned_array_decoder
    decoder = decoder_type()
    decoder.share_mem = False
    with pytest.raises(RuntimeError, match="original shared-buffer decoder"):
        install(decoder, None)


@pytest.mark.parametrize("async_engine", [False, True])
def test_copying_decoder_requires_async_engine(async_engine: bool) -> None:
    config = {"vllm_cfg": {"copy_decoded_arrays": True, "async_engine": async_engine}}
    if async_engine:
        validate_vllm_async_actor_settings(config)
    else:
        with pytest.raises(
            ValueError, match="copy_decoded_arrays requires async_engine"
        ):
            validate_vllm_async_actor_settings(config)
