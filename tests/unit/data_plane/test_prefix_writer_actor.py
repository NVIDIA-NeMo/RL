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
"""CPU coverage of prefix transport, durability barriers and cleanup routing."""

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.prefix_writer_actor import (
    PackedColumn,
    PrefixPut,
    PrefixWriterService,
    PrefixWriteReply,
    RayPrefixWriterClient,
    create_prefix_writer,
    pack_columns,
    unpack_columns,
    validate_prefix_writer_setup,
)
from nemo_rl.data_plane.tq_token_sink import TQTokenSink
from tests.unit.models.generation.test_vllm_token_capture_hosting import (
    _batch_worker_fixture,
    _fake_worker,
)


def _fields() -> TensorDict:
    return TensorDict(
        {
            "tokens": torch.nested.nested_tensor(
                [torch.tensor([1, 2]), torch.tensor([3, 4, 5])], layout=torch.jagged
            ),
            "routes": torch.nested.nested_tensor(
                [
                    torch.ones((2, 3, 2), dtype=torch.int16),
                    torch.zeros((3, 3, 2), dtype=torch.int16),
                ],
                layout=torch.jagged,
            ),
            "metadata": torch.tensor([[7, 8], [9, 10]], dtype=torch.uint8),
            "bf16": torch.tensor([[1.5], [-2.25]], dtype=torch.bfloat16),
            "empty": torch.empty((2, 0), dtype=torch.float32),
        },
        batch_size=[2],
    )


def _batch() -> PrefixPut:
    return PrefixPut(
        ("__generation_cut__/s/a/c/0", "__generation_cut__/s/b/d/0"),
        "staging",
        pack_columns(_fields()),
        ({"a": 1}, {"b": 2}),
    )


def test_transport_preserves_ragged_routes_bf16_and_empty_columns() -> None:
    original = _fields()
    packed = pack_columns(original)
    restored = unpack_columns(packed, 2)
    for name, tensor in original.items():
        for index in range(2):
            torch.testing.assert_close(
                restored[name][index], tensor[index], rtol=0, atol=0
            )
    assert all(column.data.dtype == np.uint8 for column in packed.values())


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
@pytest.mark.parametrize("layout", ["dense", "strided", "empty", "jagged"])
def test_packed_columns_preserve_exact_bytes(dtype: torch.dtype, layout: str) -> None:
    values = torch.arange(6).to(dtype)
    if layout == "jagged":
        tensor = torch.nested.nested_tensor_from_jagged(
            values, offsets=torch.tensor([0, 2, 6])
        )
    elif layout == "strided":
        tensor = values.reshape(3, 2).t()
    elif layout == "empty":
        tensor = torch.empty((2, 0), dtype=dtype)
    else:
        tensor = values.reshape(2, 3)
    batch = TensorDict({"column": tensor}, batch_size=[2])
    column = pack_columns(batch)["column"]
    source = tensor.values() if tensor.is_nested else tensor
    expected = source.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
    np.testing.assert_array_equal(column.data, expected)
    assert column.shape == tuple(source.shape)
    assert column.dtype == str(dtype)
    assert column.data.flags.c_contiguous
    if tensor.is_nested:
        np.testing.assert_array_equal(column.offsets, tensor.offsets().numpy())
    else:
        assert column.offsets is None
    restored = unpack_columns({"column": column}, 2)["column"]
    for row in range(2):
        torch.testing.assert_close(restored[row], tensor[row], rtol=0, atol=0)


def test_packing_detaches_grad_and_preserves_float_bit_patterns() -> None:
    bits = torch.tensor(
        [[0, -2147483648, 2139095040], [2143289345, -4194303, -8388608]],
        dtype=torch.int32,
    )
    values = bits.view(torch.float32).requires_grad_()
    batch = TensorDict({"column": values}, batch_size=[2])
    column = pack_columns(batch)["column"]
    np.testing.assert_array_equal(
        column.data, bits.reshape(-1).view(torch.uint8).numpy()
    )
    restored = unpack_columns({"column": column}, 2)["column"]
    assert not restored.requires_grad
    assert torch.equal(restored.view(torch.int32), bits)
    assert values.requires_grad


@pytest.mark.parametrize("defect", ["size", "dtype", "offsets", "rows"])
def test_bad_transport_is_rejected_before_put(defect: str) -> None:
    batch = _batch()
    columns = dict(batch.columns)
    if defect == "size":
        columns["metadata"] = replace(
            columns["metadata"], data=np.zeros(1, dtype=np.uint8)
        )
    elif defect == "dtype":
        columns["metadata"] = replace(columns["metadata"], dtype="bogus")
    elif defect == "offsets":
        columns["tokens"] = replace(columns["tokens"], offsets=np.array([0, 4, 2]))
    else:
        columns["metadata"] = PackedColumn(
            np.zeros(2, dtype=np.uint8), "torch.uint8", (1, 2), None
        )
    client = MagicMock()
    with pytest.raises(ValueError):
        PrefixWriterService(client, "staging").put(replace(batch, columns=columns))
    client.put_samples.assert_not_called()


@pytest.mark.parametrize("defect", ["partition", "canonical", "duplicate", "tags"])
def test_service_rejects_wrong_identity(defect: str) -> None:
    batch = _batch()
    if defect == "partition":
        batch = replace(batch, partition_id="another")
    elif defect == "canonical":
        batch = replace(batch, sample_ids=("a/c", "b/d"))
    elif defect == "duplicate":
        batch = replace(batch, sample_ids=(batch.sample_ids[0],) * 2)
    else:
        batch = replace(batch, tags=())
    client = MagicMock()
    with pytest.raises(ValueError):
        PrefixWriterService(client, "staging").put(batch)
    client.put_samples.assert_not_called()


def test_service_ack_waits_for_tq_and_preserves_keys_and_tags() -> None:
    entered, release = threading.Event(), threading.Event()
    client = MagicMock()

    def put(**kwargs: object) -> None:
        entered.set()
        assert release.wait(5)

    client.put_samples.side_effect = put
    batch = _batch()
    replies = []
    thread = threading.Thread(
        target=lambda: replies.append(PrefixWriterService(client, "staging").put(batch))
    )
    thread.start()
    try:
        assert entered.wait(5)
        assert not replies
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert replies[0].sample_ids == batch.sample_ids
    assert client.put_samples.call_args.kwargs["tags"] == list(batch.tags)
    assert replies[0].put_s >= replies[0].put_cpu_s - 0.01


def test_client_refuses_bad_ack_and_closed_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = SimpleNamespace(put=MagicMock())
    client = RayPrefixWriterClient(actor, writer_pid=99)
    monkeypatch.setattr("ray.put", lambda batch: batch)
    monkeypatch.setattr(
        "ray.get", lambda ref: PrefixWriteReply(("wrong",), 0, 0, 0, 0, 0)
    )
    kill = MagicMock()
    monkeypatch.setattr("ray.kill", kill)
    with pytest.raises(RuntimeError, match="different durable keys"):
        client.put_samples(
            sample_ids=list(_batch().sample_ids),
            partition_id="staging",
            fields=_fields(),
            tags=[{}, {}],
        )
    assert "batches" not in client.snapshot()
    client.close()
    client.close()
    kill.assert_called_once_with(actor, no_restart=True)
    with pytest.raises(RuntimeError, match="closed"):
        client.put_samples(
            sample_ids=["anything"], partition_id="staging", fields=_fields(), tags=[]
        )


def test_client_records_timings_and_orders_failed_write_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = MagicMock()
    service = PrefixWriterService(data, "staging")
    actor = SimpleNamespace(
        put=SimpleNamespace(remote=service.put),
        clear=SimpleNamespace(remote=service.clear),
    )
    monkeypatch.setattr("ray.put", lambda batch: batch)
    monkeypatch.setattr("ray.get", lambda result: result)
    client = RayPrefixWriterClient(actor, writer_pid=99)
    batch = _batch()
    client.put_samples(
        sample_ids=list(batch.sample_ids),
        partition_id="staging",
        fields=_fields(),
        tags=list(batch.tags),
    )
    client.clear_samples(sample_ids=list(batch.sample_ids), partition_id="staging")
    assert client.snapshot()["batches"] == 1
    assert client.snapshot()["rows"] == 2
    assert client.snapshot()["payload_bytes"] == batch.payload_bytes
    assert data.method_calls[0][0] == "put_samples"
    assert data.method_calls[1][0] == "clear_samples"


@pytest.mark.parametrize("fail", [False, True])
def test_native_cut_uses_prefix_client_and_terminal_remains_local(fail: bool) -> None:
    recovery, memory, _, inventory = _batch_worker_fixture([2, 3], batch_size=2)

    def client() -> SimpleNamespace:
        return SimpleNamespace(
            put_samples=MagicMock(spec=lambda **kwargs: None),
            clear_samples=MagicMock(spec=lambda **kwargs: None),
        )

    local, prefix = client(), client()
    sink = TQTokenSink(local, staging_partition="staging")
    sink.set_prefix_writer(prefix)
    recovery._parts = replace(recovery._parts, sink=sink)
    if fail:
        prefix.put_samples.side_effect = RuntimeError("injected failed PUT")
    try:
        receipt = recovery.checkpoint_cut(inventory)
        expected = "durable_failure" if fail else "durable_prefix"
        assert all(ack.disposition == expected for ack in receipt.prefixes)
        prefix.put_samples.assert_called()
        local.put_samples.assert_not_called()
        if fail:
            prefix.clear_samples.assert_called()
            local.clear_samples.assert_not_called()
    finally:
        asyncio.run(recovery.aclose())
        recovery.shutdown()
    other, memory, _, original = _batch_worker_fixture([2])
    try:
        other.checkpoint_cut(original)
        record = memory.generation_prefix_records[0][1]
        assert sink.stage(record).ok
        local.put_samples.assert_called_once()
    finally:
        asyncio.run(other.aclose())
        other.shutdown()


@pytest.mark.parametrize(
    "prefix_cuts,capture_media,backend",
    [(False, False, "simple"), (True, True, "simple"), (True, False, "mooncake_cpu")],
)
def test_unsupported_deployment_is_rejected(
    prefix_cuts: bool, capture_media: bool, backend: str
) -> None:
    with pytest.raises(ValueError):
        validate_prefix_writer_setup(
            {"impl": "transfer_queue", "backend": backend},
            prefix_cuts=prefix_cuts,
            capture_media=capture_media,
        )


@pytest.mark.parametrize("failure", [None, "wrong_node", "startup_error"])
def test_factory_pins_writer_and_cleans_failed_startup(
    failure: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_id = "a" * 56
    actor = SimpleNamespace(
        ready=SimpleNamespace(
            remote=AsyncMock(
                return_value={
                    "pid": 99,
                    "node_id": "b" * 56 if failure == "wrong_node" else node_id,
                },
                side_effect=RuntimeError("startup")
                if failure == "startup_error"
                else None,
            )
        )
    )
    descriptor = MagicMock()
    descriptor.options.return_value.remote.return_value = actor
    monkeypatch.setattr("ray.remote", lambda cls: descriptor)
    monkeypatch.setattr(
        "ray.get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: node_id)
    )
    kill = MagicMock()
    monkeypatch.setattr("ray.kill", kill)
    if failure:
        with pytest.raises(RuntimeError):
            asyncio.run(
                create_prefix_writer(
                    {"impl": "transfer_queue", "backend": "simple"},
                    partition_id="staging",
                )
            )
        kill.assert_called_once_with(actor, no_restart=True)
    else:
        client = asyncio.run(
            create_prefix_writer(
                {"impl": "transfer_queue", "backend": "simple"}, partition_id="staging"
            )
        )
        assert client.snapshot()["writer_pid"] == 99
        kill.assert_not_called()
    options = descriptor.options.call_args.kwargs
    assert options["scheduling_strategy"].node_id == node_id
    assert options["max_concurrency"] == 1
    assert options["max_restarts"] == options["max_task_retries"] == 0


@pytest.mark.parametrize(
    "writer_enabled,pipeline_enabled",
    [(True, False), (False, False), (False, True), (True, True)],
)
def test_worker_setup_installs_only_prefix_transport(
    monkeypatch: pytest.MonkeyPatch,
    writer_enabled: bool,
    pipeline_enabled: bool,
) -> None:
    worker = _fake_worker()
    worker.cfg = {
        "vllm_cfg": {
            "prefix_writer_actor": writer_enabled,
            "prefix_write_pipeline": {
                "enabled": pipeline_enabled,
                "direct_column_encoding": not writer_enabled or pipeline_enabled,
            },
        }
    }
    worker._http_engine_client = object()
    worker._progress_reader = MagicMock()
    worker.llm = SimpleNamespace(
        renderer=SimpleNamespace(get_tokenizer=lambda: object())
    )
    client = SimpleNamespace()
    writer = SimpleNamespace(snapshot=lambda: {}, close=MagicMock())
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client", lambda *args, **kwargs: client
    )
    create = AsyncMock(return_value=writer)
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async.create_prefix_writer", create
    )
    try:
        asyncio.run(
            worker.setup_token_capture(
                {"impl": "transfer_queue", "backend": "simple"},
                "staging",
                generation_prefix_cuts_enabled=True,
                generation_cut_control_token="secret",
                generation_prefix_batch_size=256,
                generation_prefix_batch_max_tokens=4096,
            )
        )
        if writer_enabled:
            assert worker._prefix_writer is writer
            assert worker._capture_sink._prefix_store._dp_client is writer
            create.assert_awaited_once()
        else:
            assert worker._prefix_writer is None
            assert worker._capture_sink._prefix_store is None
            create.assert_not_awaited()
        assert (worker._prefix_recovery._pipeline is not None) == pipeline_enabled
        assert worker._capture_sink._store._dp_client is client
        if writer_enabled:
            with pytest.raises(RuntimeError, match="already installed"):
                asyncio.run(
                    worker.setup_token_capture(
                        {"impl": "transfer_queue", "backend": "simple"},
                        "staging",
                        generation_prefix_cuts_enabled=True,
                        generation_cut_control_token="secret",
                        generation_prefix_batch_size=256,
                        generation_prefix_batch_max_tokens=4096,
                    )
                )
    finally:
        if worker._prefix_recovery._pipeline is not None:
            worker._prefix_recovery._pipeline.close(wait=True)
        asyncio.run(worker._prefix_recovery.aclose())
        worker._prefix_recovery.shutdown()


def test_close_unblocks_pending_call_without_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, stopped = threading.Event(), threading.Event()
    actor = SimpleNamespace(put=MagicMock())
    client = RayPrefixWriterClient(actor, writer_pid=99)
    monkeypatch.setattr("ray.put", lambda batch: batch)

    def get(reference: object) -> object:
        entered.set()
        assert stopped.wait(5)
        raise RuntimeError("actor terminated")

    monkeypatch.setattr("ray.get", get)
    monkeypatch.setattr("ray.kill", lambda *args, **kwargs: stopped.set())
    errors = []

    def put() -> None:
        try:
            client.put_samples(
                sample_ids=list(_batch().sample_ids),
                partition_id="staging",
                fields=_fields(),
                tags=[{}, {}],
            )
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=put)
    thread.start()
    try:
        assert entered.wait(5)
        client.close()
    finally:
        stopped.set()
        thread.join(5)
    assert not thread.is_alive()
    assert errors == ["actor terminated"]
    assert "batches" not in client.snapshot()


def test_worker_shutdown_closes_writer_once(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _fake_worker()
    writer = SimpleNamespace(close=MagicMock())
    worker._prefix_writer = writer
    worker.server_thread = None
    worker._sparse_refit_receiver = None
    worker.llm = None
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async.shutdown_telemetry",
        lambda: None,
    )
    assert asyncio.run(worker.shutdown())
    assert asyncio.run(worker.shutdown())
    assert worker._prefix_writer is None
    writer.close.assert_called_once()
