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
"""CPU coverage of the cut-row transport, durability barriers and cleanup routing."""

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import ray
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.generation_cut_rows import GenerationCutRow
from nemo_rl.data_plane.observability import MetricsDataPlaneClient
from nemo_rl.data_plane.prefix_writer_actor import (
    PrefixWriterActor,
    PrefixWriterDiedError,
    PrefixWriterService,
    RayPrefixWriterClient,
    create_prefix_writer,
    from_bytes,
    to_bytes,
)
from nemo_rl.data_plane.tq_token_sink import TQTokenSink
from nemo_rl.models.generation.prefix_recovery import PrefixRecovery
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


def _rows() -> list[GenerationCutRow]:
    """Two text calls' first cut rows."""
    from nemo_gym.token_id_capture.staging.capture import ActiveCall
    from nemo_gym.token_id_capture.staging.records import CaptureAdmission

    rows = []
    for index, generated in enumerate(([20, 21], [30, 31, 32])):
        admission = CaptureAdmission(
            capture_key=f"rollout-{index}:0", model_call_id=f"call-{index}", mode="text"
        )
        logprobs = [-0.5] * len(generated)
        rows.append(
            GenerationCutRow(
                call=ActiveCall(admission=admission, weight_version=1),
                prompt_token_ids=[1, 2],
                generated_token_ids=generated,
                generated_logprobs=logprobs,
                chunk_token_ids=generated,
                chunk_logprobs=logprobs,
                first_chunk=True,
                chunk_sequence=0,
            )
        )
    return rows


def _in_process_writer(
    data: Any, monkeypatch: pytest.MonkeyPatch
) -> RayPrefixWriterClient:
    """A writer client whose actor is a ``PrefixWriterService`` over ``data`` here."""
    service = PrefixWriterService(data)
    actor = SimpleNamespace(
        configure_generation_cut_sink=SimpleNamespace(
            remote=service.configure_generation_cut_sink
        ),
        stage_generation_cut_rows=SimpleNamespace(
            remote=service.stage_generation_cut_rows
        ),
    )
    monkeypatch.setattr("ray.get", lambda reference: reference)
    return RayPrefixWriterClient(actor, writer_pid=99)


def test_byte_columns_round_trip_exactly() -> None:
    """Dense, jagged, bf16, empty, strided and grad columns survive the hop."""
    fields = _fields()
    fields["strided"] = torch.arange(6, dtype=torch.int64).reshape(3, 2).t()
    bits = torch.tensor([[0, -2147483648], [2143289345, -8388608]], dtype=torch.int32)
    fields["float_bits"] = bits.view(torch.float32).requires_grad_()
    for name, tensor in fields.items():
        data, dtype, shape, offsets = to_bytes(tensor)
        assert data.dtype == np.uint8
        restored = from_bytes(data, dtype, shape, offsets)
        assert not restored.requires_grad
        for row in range(2):
            torch.testing.assert_close(
                restored[row], tensor[row], rtol=0, atol=0, equal_nan=True
            )
    restored = from_bytes(*to_bytes(fields["float_bits"]))
    assert torch.equal(restored.view(torch.int32), bits)


def test_service_returns_only_after_tq_acknowledges_the_rows() -> None:
    """The actor builds the same rows and digests the worker would, then waits for TQ."""
    entered, release = threading.Event(), threading.Event()

    def put(**kwargs: object) -> None:
        entered.set()
        assert release.wait(5)

    client = SimpleNamespace(
        put_samples=MagicMock(spec=lambda **kwargs: None, side_effect=put)
    )
    service = PrefixWriterService(client)
    service.configure_generation_cut_sink(
        staging_partition="staging", capture_media=False, media_pixel_dtype=None
    )
    rows = _rows()
    replies = []
    thread = threading.Thread(
        target=lambda: replies.append(
            service.stage_generation_cut_rows(
                rows, checkpoint_id="cut", attachments=[None, None]
            )
        )
    )
    thread.start()
    try:
        assert entered.wait(5)
        assert not replies
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    [staged] = replies
    local = TQTokenSink(
        SimpleNamespace(put_samples=MagicMock(spec=lambda **kwargs: None)),
        staging_partition="staging",
    )
    assert staged == local.stage_generation_cut_rows(rows, checkpoint_id="cut")
    call = client.put_samples.call_args.kwargs
    assert call["sample_ids"] == [row.staging_key for row in staged]
    assert call["partition_id"] == "staging"


def test_service_refuses_rows_before_its_sink_is_configured() -> None:
    with pytest.raises(RuntimeError, match="not configured"):
        PrefixWriterService(MagicMock()).stage_generation_cut_rows(
            _rows(), checkpoint_id="cut", attachments=[None, None]
        )


def test_client_refuses_a_closed_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    actor = SimpleNamespace(
        stage_generation_cut_rows=SimpleNamespace(remote=MagicMock())
    )
    client = RayPrefixWriterClient(actor, writer_pid=99)
    kill = MagicMock()
    monkeypatch.setattr("ray.kill", kill)
    client.close()
    client.close()
    kill.assert_called_once_with(actor, no_restart=True)
    with pytest.raises(RuntimeError, match="closed"):
        client.stage_generation_cut_rows(_rows(), checkpoint_id="cut")
    actor.stage_generation_cut_rows.remote.assert_not_called()


def test_client_sends_attachments_as_byte_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rows cross without their tensors; the tensors cross as NumPy buffers."""
    sent = {}

    def stage(rows: Any, *, checkpoint_id: str, attachments: Any) -> list[Any]:
        sent.update(rows=rows, attachments=attachments)
        return []

    actor = SimpleNamespace(stage_generation_cut_rows=SimpleNamespace(remote=stage))
    monkeypatch.setattr("ray.get", lambda reference: reference)
    client = RayPrefixWriterClient(actor, writer_pid=99)
    routes = torch.arange(12, dtype=torch.int16).reshape(2, 3, 2)
    first, second = _rows()
    client.stage_generation_cut_rows(
        [replace(first, attachments={"routes": routes}), second], checkpoint_id="cut"
    )
    assert [row.attachments for row in sent["rows"]] == [None, None]
    columns, none = sent["attachments"]
    assert none is None
    assert torch.equal(from_bytes(*columns["routes"]), routes)


@pytest.mark.parametrize("fail", [False, True])
def test_native_cut_uses_prefix_client_and_terminal_remains_local(
    fail: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    recovery, memory, _, inventory = _batch_worker_fixture([2, 3], batch_size=2)

    def client() -> SimpleNamespace:
        return SimpleNamespace(
            put_samples=MagicMock(spec=lambda **kwargs: None),
            clear_samples=MagicMock(spec=lambda **kwargs: None),
        )

    local, prefix = client(), client()
    sink = TQTokenSink(local, staging_partition="staging")
    sink.set_prefix_writer(_in_process_writer(prefix, monkeypatch))
    recovery = PrefixRecovery(replace(recovery._parts, sink=sink))
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
        configure_generation_cut_sink=SimpleNamespace(remote=MagicMock()),
        stage_generation_cut_rows=SimpleNamespace(remote=MagicMock()),
        get_data_plane_snapshot=SimpleNamespace(remote=MagicMock()),
    )
    client = RayPrefixWriterClient(actor, writer_pid=99)

    def actor_died(reference: object) -> None:
        raise ray.exceptions.RayActorError()

    monkeypatch.setattr("ray.get", actor_died)
    with pytest.raises(PrefixWriterDiedError, match="pid 99"):
        client.stage_generation_cut_rows(_rows(), checkpoint_id="cut")
    actor.stage_generation_cut_rows.remote.reset_mock()
    with pytest.raises(PrefixWriterDiedError):
        client.stage_generation_cut_rows(_rows(), checkpoint_id="cut")
    with pytest.raises(PrefixWriterDiedError):
        client.configure_generation_cut_sink(
            staging_partition="staging", capture_media=False, media_pixel_dtype=None
        )
    assert asyncio.run(client.get_data_plane_snapshot()) is None
    actor.stage_generation_cut_rows.remote.assert_not_called()
    actor.configure_generation_cut_sink.remote.assert_not_called()
    actor.get_data_plane_snapshot.remote.assert_not_called()


def test_dead_writer_fails_the_whole_cut_and_discards_locally() -> None:
    """A dead writer fails the cut loudly; no call is acknowledged as cut."""
    recovery, _, states, inventory = _batch_worker_fixture([2, 3], batch_size=2)
    local = SimpleNamespace(
        put_samples=MagicMock(spec=lambda **kwargs: None),
        clear_samples=MagicMock(spec=lambda **kwargs: None),
    )
    writer = SimpleNamespace(
        configure_generation_cut_sink=MagicMock(),
        stage_generation_cut_rows=MagicMock(
            side_effect=PrefixWriterDiedError("writer died")
        ),
    )
    sink = TQTokenSink(local, staging_partition="staging")
    sink.set_prefix_writer(writer)
    recovery = PrefixRecovery(replace(recovery._parts, sink=sink))
    try:
        with pytest.raises(PrefixWriterDiedError, match="writer died"):
            asyncio.run(recovery.run_cut(inventory))
        writer.stage_generation_cut_rows.assert_called()
        local.put_samples.assert_not_called()
        # The partial write is discarded through the local client.
        local.clear_samples.assert_called()
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
                create_prefix_writer({"impl": "transfer_queue", "backend": "simple"})
            )
        kill.assert_called_once_with(actor, no_restart=True)
    else:
        client = asyncio.run(
            create_prefix_writer({"impl": "transfer_queue", "backend": "simple"})
        )
        assert client._writer_pid == 99
        kill.assert_not_called()
    options = descriptor.options.call_args.kwargs
    assert options["scheduling_strategy"].node_id == node_id
    assert options["max_concurrency"] == 1
    assert options["max_restarts"] == options["max_task_retries"] == 0


@pytest.mark.parametrize(
    "dp_cfg",
    [{"impl": "local"}, {"impl": "transfer_queue", "backend": "simple"}],
    ids=["local", "transfer_queue"],
)
def test_worker_setup_installs_only_prefix_transport(
    monkeypatch: pytest.MonkeyPatch, dp_cfg: dict[str, str]
) -> None:
    """A TransferQueue data plane gets a writer actor; a local one writes itself."""
    worker = _fake_worker()
    writer_enabled = dp_cfg["impl"] == "transfer_queue"
    worker._http_engine_client = object()
    worker._progress_reader = MagicMock()
    worker.llm = SimpleNamespace(
        renderer=SimpleNamespace(get_tokenizer=lambda: object())
    )
    client = SimpleNamespace()
    writer = SimpleNamespace(
        snapshot=lambda: {},
        close=MagicMock(),
        configure_generation_cut_sink=MagicMock(),
    )
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
                dp_cfg,
                "staging",
                generation_prefix_cuts_enabled=True,
                generation_cut_control_token="secret",
                prefix_write_max_rows=256,
                prefix_write_max_tokens=4096,
                prefix_read_max_rows=1,
            )
        )
        if writer_enabled:
            assert worker._prefix_writer is writer
            assert worker._capture_sink._prefix_writer is writer
            assert (
                writer.configure_generation_cut_sink.call_args.kwargs[
                    "staging_partition"
                ]
                == "staging"
            )
            create.assert_awaited_once()
        else:
            assert worker._prefix_writer is None
            assert worker._capture_sink._prefix_writer is None
            create.assert_not_awaited()
        assert worker._prefix_recovery._pipeline is not None
        assert worker._capture_sink._store._dp_client is client
        if writer_enabled:
            with pytest.raises(RuntimeError, match="already installed"):
                asyncio.run(
                    worker.setup_token_capture(
                        dp_cfg,
                        "staging",
                        generation_prefix_cuts_enabled=True,
                        generation_cut_control_token="secret",
                        prefix_write_max_rows=256,
                        prefix_write_max_tokens=4096,
                        prefix_read_max_rows=1,
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
    actor = SimpleNamespace(
        stage_generation_cut_rows=SimpleNamespace(remote=MagicMock())
    )
    client = RayPrefixWriterClient(actor, writer_pid=99)

    def get(reference: object) -> object:
        entered.set()
        assert stopped.wait(5)
        raise RuntimeError("actor terminated")

    monkeypatch.setattr("ray.get", get)
    monkeypatch.setattr("ray.kill", lambda *args, **kwargs: stopped.set())
    errors = []

    def put() -> None:
        try:
            client.stage_generation_cut_rows(_rows(), checkpoint_id="cut")
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


def test_writer_actor_client_owns_no_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    """On mooncake_cpu a save would otherwise wait on the writer actor."""
    calls = []
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client",
        lambda *args, **kwargs: calls.append(kwargs) or MagicMock(),
    )
    monkeypatch.setattr("torch.set_num_threads", lambda threads: None)
    PrefixWriterActor({"impl": "transfer_queue", "backend": "mooncake_cpu"})
    assert calls == [{"bootstrap": False, "segment_size": 0}]


def test_writer_reports_only_a_metrics_client_snapshot() -> None:
    assert PrefixWriterService(MagicMock()).get_data_plane_snapshot() is None
    metrics = MagicMock(spec=MetricsDataPlaneClient)
    metrics.snapshot.return_value = {"n_ops": 1}
    service = PrefixWriterService(metrics)
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
