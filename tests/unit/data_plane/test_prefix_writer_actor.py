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
import ray
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.observability import MetricsDataPlaneClient
from nemo_rl.data_plane.prefix_writer_actor import (
    PackedColumn,
    PrefixPut,
    PrefixWriterDiedError,
    PrefixWriterService,
    RayPrefixWriterClient,
    create_prefix_writer,
    pack_columns,
    unpack_columns,
)
from nemo_rl.data_plane.tq_token_sink import TQTokenSink
from nemo_rl.models.generation.prefix_recovery import PrefixRecovery
from nemo_rl.models.generation.prefix_recovery.write_pipeline import PrefixWritesConfig
from tests.unit.models.generation.test_vllm_token_capture_hosting import (
    _assert_rolled_back,
    _batch_worker_fixture,
    _fake_worker,
)

pytestmark = pytest.mark.nemo_gym


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
    assert replies == [batch.sample_ids]
    assert client.put_samples.call_args.kwargs["tags"] == list(batch.tags)


def test_client_refuses_bad_ack_and_closed_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = SimpleNamespace(put=MagicMock())
    client = RayPrefixWriterClient(actor, writer_pid=99)
    monkeypatch.setattr("ray.put", lambda batch: batch)
    monkeypatch.setattr("ray.get", lambda ref: ("wrong",))
    kill = MagicMock()
    monkeypatch.setattr("ray.kill", kill)
    with pytest.raises(RuntimeError, match="different durable keys"):
        client.put_samples(
            sample_ids=list(_batch().sample_ids),
            partition_id="staging",
            fields=_fields(),
            tags=[{}, {}],
        )
    client.close()
    client.close()
    kill.assert_called_once_with(actor, no_restart=True)
    with pytest.raises(RuntimeError, match="closed"):
        client.put_samples(
            sample_ids=["anything"], partition_id="staging", fields=_fields(), tags=[]
        )


def test_client_orders_failed_write_cleanup(
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


def test_client_marks_a_dead_writer_and_stops_calling_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = SimpleNamespace(
        put=SimpleNamespace(remote=MagicMock()),
        clear=SimpleNamespace(remote=MagicMock()),
        get_data_plane_snapshot=SimpleNamespace(remote=MagicMock()),
    )
    client = RayPrefixWriterClient(actor, writer_pid=99)
    monkeypatch.setattr("ray.put", lambda batch: batch)

    def actor_died(reference: object) -> None:
        raise ray.exceptions.RayActorError()

    monkeypatch.setattr("ray.get", actor_died)
    batch = _batch()
    with pytest.raises(PrefixWriterDiedError, match="pid 99"):
        client.put_samples(
            sample_ids=list(batch.sample_ids),
            partition_id="staging",
            fields=_fields(),
            tags=list(batch.tags),
        )
    actor.put.remote.reset_mock()
    with pytest.raises(PrefixWriterDiedError):
        client.put_samples(
            sample_ids=list(batch.sample_ids),
            partition_id="staging",
            fields=_fields(),
            tags=list(batch.tags),
        )
    with pytest.raises(PrefixWriterDiedError):
        client.clear_samples(sample_ids=list(batch.sample_ids), partition_id="staging")
    assert asyncio.run(client.get_data_plane_snapshot()) is None
    actor.put.remote.assert_not_called()
    actor.clear.remote.assert_not_called()
    actor.get_data_plane_snapshot.remote.assert_not_called()


@pytest.mark.parametrize("pipeline", [False, True])
def test_dead_writer_fails_the_whole_cut_and_discards_locally(pipeline: bool) -> None:
    """A dead writer fails the cut loudly; no call is acknowledged as cut."""
    recovery, _, states, inventory = _batch_worker_fixture([2, 3], batch_size=2)
    local = SimpleNamespace(
        put_samples=MagicMock(spec=lambda **kwargs: None),
        clear_samples=MagicMock(spec=lambda **kwargs: None),
    )
    writer = SimpleNamespace(
        put_samples=MagicMock(
            spec=lambda **kwargs: None,
            side_effect=PrefixWriterDiedError("writer died"),
        ),
        clear_samples=MagicMock(
            spec=lambda **kwargs: None,
            side_effect=PrefixWriterDiedError("writer died"),
        ),
    )
    sink = TQTokenSink(local, staging_partition="staging")
    sink.set_prefix_writer(writer)
    recovery = PrefixRecovery(
        replace(
            recovery._parts,
            sink=sink,
            prefix_writes=PrefixWritesConfig(pipeline=pipeline),
        )
    )
    try:
        with pytest.raises(PrefixWriterDiedError, match="writer died"):
            asyncio.run(recovery.run_cut(inventory))
        writer.put_samples.assert_called()
        local.put_samples.assert_not_called()
        # The partial write is discarded through the local client.
        local.clear_samples.assert_called()
        writer.clear_samples.assert_not_called()
        for state, generated in zip(states, [2, 3], strict=True):
            _assert_rolled_back(state, generated)
    finally:
        asyncio.run(recovery.aclose())
        recovery.shutdown()


@pytest.mark.parametrize(
    "failure,match",
    [
        (None, None),
        ("wrong_node", "off its generation node"),
        ("startup_error", "startup"),
        ("timeout", "needs 1 free CPU"),
    ],
)
def test_factory_pins_writer_and_cleans_failed_startup(
    failure: str | None, match: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_id = "a" * 56

    async def never_scheduled() -> None:
        await asyncio.sleep(60)

    actor = SimpleNamespace(
        ready=SimpleNamespace(
            remote=AsyncMock(
                return_value={
                    "pid": 99,
                    "node_id": "b" * 56 if failure == "wrong_node" else node_id,
                },
                side_effect={
                    "startup_error": RuntimeError("startup"),
                    "timeout": never_scheduled,
                }.get(failure),
            )
        )
    )
    monkeypatch.setattr(
        "nemo_rl.data_plane.prefix_writer_actor._WRITER_STARTUP_TIMEOUT_S", 0.01
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
        with pytest.raises(RuntimeError, match=match):
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
        assert client._writer_pid == 99
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
    prefix_writes = PrefixWritesConfig(
        pipeline=pipeline_enabled, writer_actor=writer_enabled
    )
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
                prefix_write_max_rows=256,
                prefix_write_max_tokens=4096,
                prefix_read_max_rows=1,
                prefix_writes=prefix_writes,
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
        assert worker._capture_sink._direct_prefix_columns is pipeline_enabled
        assert worker._capture_sink._store._dp_client is client
        if writer_enabled:
            with pytest.raises(RuntimeError, match="already installed"):
                asyncio.run(
                    worker.setup_token_capture(
                        {"impl": "transfer_queue", "backend": "simple"},
                        "staging",
                        generation_prefix_cuts_enabled=True,
                        generation_cut_control_token="secret",
                        prefix_write_max_rows=256,
                        prefix_write_max_tokens=4096,
                        prefix_read_max_rows=1,
                        prefix_writes=prefix_writes,
                    )
                )
    finally:
        if worker._prefix_recovery._pipeline is not None:
            worker._prefix_recovery._pipeline.close(wait=True)
        asyncio.run(worker._prefix_recovery.aclose())
        worker._prefix_recovery.shutdown()


def test_worker_setup_stops_the_pipeline_when_the_writer_fails_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _fake_worker()
    worker._http_engine_client = object()
    worker._progress_reader = MagicMock()
    worker.llm = SimpleNamespace(
        renderer=SimpleNamespace(get_tokenizer=lambda: object())
    )
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async.create_prefix_writer",
        AsyncMock(side_effect=RuntimeError("writer startup failed")),
    )
    with pytest.raises(RuntimeError, match="writer startup failed"):
        asyncio.run(
            worker.setup_token_capture(
                {"impl": "transfer_queue", "backend": "simple"},
                "staging",
                generation_prefix_cuts_enabled=True,
                generation_cut_control_token="secret",
                prefix_write_max_rows=256,
                prefix_write_max_tokens=4096,
                prefix_read_max_rows=1,
                prefix_writes=PrefixWritesConfig(pipeline=True, writer_actor=True),
            )
        )
    pipeline = worker._prefix_recovery._pipeline
    try:
        assert worker._prefix_writer is None
        pipeline._thread.join(5)
        assert not pipeline._thread.is_alive()
        with pytest.raises(RuntimeError, match="closed"):
            pipeline.submit(
                SimpleNamespace(checkpoint_id="c", inventory_digest="d"), deadline=None
            )
    finally:
        asyncio.run(worker._prefix_recovery.aclose())


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


def test_writer_reports_only_a_metrics_client_snapshot() -> None:
    assert PrefixWriterService(MagicMock(), "staging").get_data_plane_snapshot() is None
    metrics = MagicMock(spec=MetricsDataPlaneClient)
    metrics.snapshot.return_value = {"n_ops": 1}
    service = PrefixWriterService(metrics, "staging")
    assert service.get_data_plane_snapshot() == {"n_ops": 1}
    metrics.snapshot.assert_called_once_with(reset_step_window=True)


def test_worker_reports_the_writer_snapshot_beside_its_own() -> None:
    """Prefix PUTs run on the writer actor's client, so its counters must be reported."""
    worker = _fake_worker()
    worker._capture_dp_client = MagicMock(spec=MetricsDataPlaneClient)
    worker._capture_dp_client.snapshot.return_value = {"n_ops": 1}
    worker._api_server_pool = None

    async def writer_snapshot() -> dict[str, int]:
        return {"n_ops": 2}

    worker._prefix_writer = SimpleNamespace(get_data_plane_snapshot=writer_snapshot)
    assert asyncio.run(worker.get_data_plane_snapshots()) == [
        {"n_ops": 1},
        {"n_ops": 2},
    ]


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
