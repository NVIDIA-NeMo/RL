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

import asyncio
import io
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest
import torch

from nemo_rl.models.generation.vllm.vllm_sparse_refit import (
    VllmSparseRefitReceiver,
    _stage_sparse_payload,
)
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)
from nemo_rl.utils.weight_transfer_remote_sparse import (
    G_VLLM_REFIT_API_KEY_HEADER,
    G_VLLM_REFIT_FLUSH_PATH,
    G_VLLM_REFIT_PREPARE_PATH,
    G_VLLM_REFIT_S3_MANIFEST_PATH,
)
from nemo_rl.utils.weight_transfer_sparse_codec import (
    encode_sparse_infos,
    iter_decoded_sparse_payload,
)
from nemo_rl.utils.weight_transfer_zmq import (
    G_VLLM_REFIT_CHECKSUM_HEADER,
    G_VLLM_REFIT_PAYLOAD_HEADER,
    G_VLLM_REFIT_PRODUCER_HEADER,
    G_VLLM_REFIT_TRANSFER_HEADER,
    G_VLLM_REFIT_ZMQ_PAYLOAD_PATH,
)


@contextmanager
def _sparse_refit_receiver(
    *,
    batch_size: int = 2,
    futures: list[Future[dict[str, Any]]] | None = None,
    async_engine: bool = False,
    config: dict[str, Any] | None = None,
) -> Iterator[VllmSparseRefitReceiver]:
    owner = SimpleNamespace(
        cfg=config or {"vllm_cfg": {"async_engine": async_engine}},
        llm=MagicMock(),
    )
    owner.llm.collective_rpc.return_value = [{"ok": True}]
    receiver = VllmSparseRefitReceiver(owner)
    receiver._refit_apply_batch_size = batch_size
    receiver._refit_apply_futures = list(futures or [])
    for future in receiver._refit_apply_futures:
        future.add_done_callback(receiver._notify_refit_apply_waiters)
    try:
        yield receiver
    finally:
        receiver._refit_apply_executor.shutdown(wait=True)
        receiver._refit_partition_executor.shutdown(wait=True)


def _serialized_sparse_payload() -> bytes:
    values = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    payload = encode_sparse_infos(
        [
            (
                "weight",
                torch.empty((8,), dtype=torch.float32),
                torch.tensor([1, 3, 4, 7]),
                values,
                "overwrite",
            )
        ]
    )
    payload[2][0].update(
        verification_locations=[1, 7],
        verification_values=[1, 4],
    )
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _stage_payloads(
    receiver: VllmSparseRefitReceiver,
    staging_dir: Path,
    *payloads: bytes,
):
    return tuple(
        receiver._refit_partition_executor.submit(
            _stage_sparse_payload, payload, str(staging_dir)
        )
        for payload in payloads
    )


def test_sparse_refit_queue_batches_payloads_in_fifo_order() -> None:
    applied: list[tuple[bytes, ...]] = []

    def apply(payloads: tuple[bytes, ...]) -> dict[str, Any]:
        applied.append(payloads)
        return {
            "ok": True,
            "payloads": len(payloads),
            "receiver_total_s": float(len(payloads)),
        }

    with _sparse_refit_receiver(batch_size=3) as receiver:
        receiver.update_weights_from_serialized_sparse_payloads = apply
        responses = [
            receiver._enqueue_sparse_payload_apply(
                payload, ("transfer", 0, index), str(index)
            )
            for index, payload in enumerate((b"0", b"1", b"2", b"3", b"4"))
        ]
        response = receiver._flush_queued_sparse_payloads()
        responses.append(response)

    assert applied == [(b"0", b"1", b"2"), (b"3", b"4")]
    assert response["payloads"] == 5
    assert response["batches"] == 2
    assert sum(result.get("receiver_total_s", 0.0) for result in responses) == 5.0
    receiver._worker.llm.collective_rpc.assert_called_once_with(
        "finish_sparse_delta_refit", args=()
    )


def test_sparse_refit_queue_stages_payload_before_batch_is_full(
    tmp_path: Path,
) -> None:
    with _sparse_refit_receiver(batch_size=2) as receiver:
        receiver._refit_workers_share_node = True
        receiver._refit_batch_staging_dir = str(tmp_path)

        response = receiver._enqueue_sparse_payload_apply(
            _serialized_sparse_payload(), ("transfer", 0, 0), "checksum"
        )
        pending = receiver._refit_apply_pending_payloads[0]
        assert isinstance(pending, Future)
        staged = pending.result(timeout=1.0)

        assert response["ok"]
        assert Path(staged.path).is_file()
        assert receiver._refit_apply_futures == []
        receiver._flush_queued_sparse_payloads()

    assert not list(tmp_path.iterdir())
    assert receiver._worker.llm.collective_rpc.call_args_list[0].args[0] == (
        "update_weights_from_decoded_sparse_payload_files"
    )


def test_sparse_refit_queue_deduplicates_transactional_payloads() -> None:
    key = ("transfer", 0, 1)
    with _sparse_refit_receiver() as receiver:
        receiver.update_weights_from_serialized_sparse_payloads = MagicMock(
            return_value={"ok": True, "payloads": 1}
        )
        assert receiver._enqueue_sparse_payload_apply(b"payload", key, "checksum")["ok"]
        duplicate = receiver._enqueue_sparse_payload_apply(b"payload", key, "checksum")
        assert duplicate == {"ok": True, "payloads": 0, "duplicate": True}
        with pytest.raises(ValueError, match="reused with different data"):
            receiver._enqueue_sparse_payload_apply(b"other", key, "different")
        response = receiver._flush_queued_sparse_payloads()

    assert response["payloads"] == 1
    assert receiver._refit_seen_payloads == {}


def test_sparse_refit_queue_does_not_deduplicate_failed_enqueue() -> None:
    failed = Future()
    failed.set_exception(RuntimeError("prior apply failed"))
    with _sparse_refit_receiver(futures=[failed]) as receiver:
        with pytest.raises(RuntimeError, match="prior apply failed"):
            receiver._enqueue_sparse_payload_apply(
                b"payload", ("transfer", 0, 1), "checksum"
            )

    assert receiver._refit_seen_payloads == {}
    assert receiver._refit_apply_pending_payloads == []


def test_sparse_refit_collective_response_merges_verification_metrics() -> None:
    response = VllmSparseRefitReceiver._refit_collective_response(
        [
            {
                "receiver_total_s": 1.0,
                "verification_candidates": 4,
                "verification_samples": 2,
                "verification_exact_mismatches": 1,
                "verification_mismatches": 0,
                "verification_abs_sum": 0.25,
                "verification_max_abs": 0.25,
            },
            {
                "receiver_total_s": 2.0,
                "verification_candidates": 4,
                "verification_samples": 3,
                "verification_exact_mismatches": 2,
                "verification_mismatches": 1,
                "verification_abs_sum": 0.5,
                "verification_max_abs": 0.4,
            },
        ]
    )

    assert response == {
        "ok": True,
        "receiver_total_s": 2.0,
        "verification_candidates": 4,
        "verification_samples": 5,
        "verification_exact_mismatches": 3,
        "verification_mismatches": 1,
        "verification_abs_sum": 0.75,
        "verification_max_abs": 0.4,
    }


def test_sparse_refit_queue_releases_condition_while_backpressured() -> None:
    first: Future[dict[str, Any]] = Future()
    second: Future[dict[str, Any]] = Future()
    started = threading.Event()

    with _sparse_refit_receiver(futures=[first, second]) as receiver:
        with ThreadPoolExecutor(max_workers=1) as callers:
            pending_call = callers.submit(
                lambda: (
                    started.set(),
                    receiver._enqueue_sparse_payload_apply(
                        b"payload", ("transfer", 0, 1), "checksum"
                    ),
                )[1]
            )
            assert started.wait(timeout=1.0)
            time.sleep(0.05)
            assert receiver._refit_apply_queue_condition.acquire(timeout=1.0)
            receiver._refit_apply_queue_condition.release()
            first.set_result({"ok": True, "payloads": 1})
            assert pending_call.result(timeout=1.0)["ok"]


def test_sparse_refit_batch_decodes_once_before_collective_apply(
    tmp_path: Path,
) -> None:
    with _sparse_refit_receiver() as receiver:
        staged_locations: list[int] = []

        def collective_rpc(method: str, args: tuple[Any, ...]) -> list[Any]:
            assert method == "update_weights_from_decoded_sparse_payload_files"
            for path in args:
                staged_locations.extend(
                    int(location)
                    for _, locations, _ in iter_decoded_sparse_payload(
                        torch.load(path, weights_only=True)
                    )
                    for location in locations
                )
            return [
                {"ok": True, "receiver_total_s": 1.0},
                {"ok": True, "receiver_total_s": 1.0},
                {"ok": True, "receiver_total_s": 1.0},
            ]

        receiver._worker.llm = MagicMock(
            collective_rpc=MagicMock(side_effect=collective_rpc)
        )
        response = receiver.update_weights_from_staged_sparse_payloads(
            _stage_payloads(receiver, tmp_path, _serialized_sparse_payload())
        )
        receiver.update_weights_from_staged_sparse_payloads(
            _stage_payloads(receiver, tmp_path, _serialized_sparse_payload())
        )

        assert staged_locations == [1, 3, 4, 7] * 2
        assert not list(tmp_path.iterdir())
        assert [
            item.args[0] for item in receiver._worker.llm.collective_rpc.call_args_list
        ] == [
            "update_weights_from_decoded_sparse_payload_files",
            "update_weights_from_decoded_sparse_payload_files",
        ]
        assert response["payloads"] == 1
        assert response["receiver_worker_total_s"] == 1.0
        assert response["receiver_total_s"] >= 0.0
        assert receiver._refit_verification_candidates == 4


def test_sparse_refit_batch_drains_workers_before_error_cleanup(tmp_path: Path) -> None:
    with _sparse_refit_receiver() as receiver:
        staged_paths: tuple[str, ...] = ()

        def collective_rpc(method: str, args: tuple[Any, ...]) -> list[Any]:
            nonlocal staged_paths
            if method == "update_weights_from_decoded_sparse_payload_files":
                staged_paths = args
                raise RuntimeError("apply failed")
            assert method == "synchronize_device"
            assert all(Path(path).is_file() for path in staged_paths)
            return [True]

        receiver._worker.llm = MagicMock(
            collective_rpc=MagicMock(side_effect=collective_rpc)
        )
        with pytest.raises(RuntimeError, match="apply failed"):
            receiver.update_weights_from_staged_sparse_payloads(
                _stage_payloads(receiver, tmp_path, _serialized_sparse_payload())
            )

        assert [
            entry.args[0]
            for entry in receiver._worker.llm.collective_rpc.call_args_list
        ] == [
            "update_weights_from_decoded_sparse_payload_files",
            "synchronize_device",
        ]
        assert not list(tmp_path.iterdir())


def test_sparse_refit_batch_uses_one_collective_rpc_across_nodes() -> None:
    with _sparse_refit_receiver() as receiver:
        receiver._refit_workers_share_node = False
        receiver._worker.llm = MagicMock(
            collective_rpc=MagicMock(
                return_value=[{"ok": True, "receiver_total_s": 1.0}]
            )
        )

        response = receiver.update_weights_from_serialized_sparse_payloads(
            (_serialized_sparse_payload(),) * 3
        )

        rpc = receiver._worker.llm.collective_rpc.call_args
        assert rpc.args[0] == "update_weights_from_decoded_sparse_payload"
        assert len(rpc.kwargs["args"]) == 3
        assert all(
            len(torch.load(io.BytesIO(payload), weights_only=True)[2]) == 1
            for payload in rpc.kwargs["args"]
        )
        assert response == {"ok": True, "receiver_total_s": 1.0, "payloads": 3}


@pytest.mark.asyncio
async def test_async_sparse_refit_batch_bridges_to_async_collective(
    tmp_path: Path,
) -> None:
    with _sparse_refit_receiver(async_engine=True) as receiver:
        staged_locations: list[int] = []

        class AsyncLlm:
            async def collective_rpc(
                self, method: str, args: tuple[Any, ...]
            ) -> list[Any]:
                assert method == "update_weights_from_decoded_sparse_payload_files"
                for path in args:
                    staged_locations.extend(
                        location
                        for _, locations, _ in iter_decoded_sparse_payload(
                            torch.load(path, weights_only=True)
                        )
                        for location in locations.tolist()
                    )
                return [{"ok": True, "receiver_total_s": 1.0}]

        receiver._worker.llm = AsyncLlm()
        receiver._refit_async_loop = asyncio.get_running_loop()
        response = await asyncio.to_thread(
            receiver.update_weights_from_staged_sparse_payloads,
            _stage_payloads(receiver, tmp_path, _serialized_sparse_payload()),
        )

        assert staged_locations == [1, 3, 4, 7]
        assert response["payloads"] == 1
        assert response["receiver_worker_total_s"] == 1.0


@pytest.mark.asyncio
async def test_sparse_refit_payload_handlers_decode_and_enqueue(monkeypatch) -> None:
    with _sparse_refit_receiver() as receiver:
        enqueue = MagicMock(
            side_effect=[
                {"ok": True, "payloads": 1},
                {"ok": True, "payloads": 1},
            ]
        )
        receiver._enqueue_sparse_payload_apply = enqueue
        monkeypatch.setattr(
            "nemo_rl.models.generation.vllm.vllm_sparse_refit.download_s3_refit_payload",
            lambda _manifest: b"s3-payload",
        )
        monkeypatch.setattr(
            "nemo_rl.models.generation.vllm.vllm_sparse_refit.decode_sparse_payload",
            lambda _body, _checksum: b"zmq-payload",
        )

        s3_result = await receiver._apply_s3_manifest_payload(
            {"key": "object-key", "checksum": "checksum"}
        )
        assert s3_result["ok"]

        invalid_request = SimpleNamespace(headers={}, body=AsyncMock())
        with pytest.raises(ValueError, match="Missing or invalid"):
            await receiver._apply_zmq_payload(invalid_request)

        request = SimpleNamespace(
            headers={
                G_VLLM_REFIT_TRANSFER_HEADER: "transfer",
                G_VLLM_REFIT_PRODUCER_HEADER: "2",
                G_VLLM_REFIT_PAYLOAD_HEADER: "3",
                G_VLLM_REFIT_CHECKSUM_HEADER: "checksum",
            },
            body=AsyncMock(return_value=b"compressed"),
        )
        zmq_result = await receiver._apply_zmq_payload(request)
        assert zmq_result["ok"]
        assert enqueue.call_args_list == [
            call(b"s3-payload", ("object-key", -1, -1), "checksum"),
            call(b"zmq-payload", ("transfer", 2, 3), "checksum"),
        ]


def test_sparse_refit_api_auth_dispatch_and_error_mapping(monkeypatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setenv("NRL_TEST_REFIT_KEY", "secret")
    config = {
        "vllm_cfg": {
            "async_engine": True,
            "http_refit_api_key_env_var": "NRL_TEST_REFIT_KEY",
        }
    }
    with _sparse_refit_receiver(async_engine=True, config=config) as receiver:
        receiver._apply_s3_manifest_payload = AsyncMock(
            return_value={"ok": True, "payloads": 1}
        )
        receiver._apply_zmq_payload = AsyncMock(
            side_effect=RuntimeError("apply failed")
        )
        receiver._flush_queued_sparse_payloads = MagicMock(
            return_value={"ok": True, "payloads": 2}
        )
        receiver._refit_collective_rpc = MagicMock(return_value=[])
        app = FastAPI()
        receiver.setup_api_server(app)
        headers = {G_VLLM_REFIT_API_KEY_HEADER: "secret"}

        with TestClient(app) as client:
            unauthorized = client.post(G_VLLM_REFIT_S3_MANIFEST_PATH, json={})
            s3_response = client.post(
                G_VLLM_REFIT_S3_MANIFEST_PATH,
                json={"key": "key"},
                headers=headers,
            )
            flush_response = client.post(G_VLLM_REFIT_FLUSH_PATH, headers=headers)
            prepare_response = client.post(
                G_VLLM_REFIT_PREPARE_PATH,
                json={"tensors": {"weight": [[2, 3], "bfloat16"]}},
                headers=headers,
            )
            zmq_response = client.post(
                G_VLLM_REFIT_ZMQ_PAYLOAD_PATH,
                content=b"payload",
                headers=headers,
            )

        assert unauthorized.status_code == 403
        assert s3_response.status_code == 200
        assert flush_response.status_code == 200
        assert prepare_response.status_code == 200
        assert zmq_response.status_code == 500
        assert receiver._refit_async_loop is not None
        receiver._apply_s3_manifest_payload.assert_awaited_once_with({"key": "key"})
        receiver._apply_zmq_payload.assert_awaited_once()
        receiver._flush_queued_sparse_payloads.assert_called_once_with()
        receiver._refit_collective_rpc.assert_called_once_with(
            "prepare_sparse_delta_refit_info",
            ({"weight": ((2, 3), torch.bfloat16)},),
        )


def test_sync_sparse_refit_server_shutdown_cleans_transport_resources(
    monkeypatch,
) -> None:
    import uvicorn

    from nemo_rl.models.generation.vllm import vllm_sparse_refit as refit_module

    configs: list[Any] = []
    servers: list[Any] = []

    def make_config(app: Any, **kwargs: Any) -> Any:
        config = SimpleNamespace(app=app, **kwargs)
        configs.append(config)
        return config

    class Server:
        def __init__(self, config: Any) -> None:
            self.config = config
            self.should_exit = False
            self.ran = threading.Event()
            servers.append(self)

        def run(self) -> None:
            self.ran.set()

    monkeypatch.setattr(uvicorn, "Config", make_config)
    monkeypatch.setattr(uvicorn, "Server", Server)
    monkeypatch.setattr(refit_module, "_get_free_port_local", lambda *_args: 12345)
    monkeypatch.setattr(refit_module, "_get_node_ip_local", lambda: "10.0.0.1")
    config = {
        "vllm_cfg": {"async_engine": False, "http_refit_server_port": None},
        "port_range_low": 10000,
        "port_range_high": 11000,
    }

    with _sparse_refit_receiver(config=config) as receiver:
        receiver._worker.base_url = "http://10.0.0.2:8000/v1"
        assert receiver.report_refit_server_base_url() == "http://10.0.0.2:8000"
        receiver.setup_api_server = MagicMock()
        receiver._setup_vllm_refit_server()
        assert configs[0].host == "0.0.0.0"
        assert configs[0].port == 12345
        assert servers[0].ran.wait(timeout=1.0)
        assert receiver.report_refit_server_base_url() == "http://10.0.0.1:12345"

        relay = MagicMock()
        receiver._zmq_refit_server = (relay, "tcp://relay")
        receiver._flush_queued_sparse_payloads = MagicMock()
        receiver._refit_apply_executor = MagicMock()
        receiver.shutdown()

        relay.close.assert_called_once_with()
        receiver._flush_queued_sparse_payloads.assert_called_once_with()
        receiver._refit_apply_executor.shutdown.assert_called_once_with(wait=True)
        assert servers[0].should_exit is True
        assert receiver._refit_http_server is None


@pytest.mark.asyncio
async def test_async_sparse_refit_post_init_records_worker_locality() -> None:
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker._sparse_refit_receiver = MagicMock()
    worker._mtp_load_from_disk = False
    worker.report_device_id_async = AsyncMock(return_value=["0"])
    worker.llm = MagicMock()
    worker.llm.collective_rpc = AsyncMock(return_value=["node-0", "node-0"])

    await worker.post_init_async()

    assert worker.vllm_device_ids == ["0"]
    worker._sparse_refit_receiver.set_worker_hostnames.assert_called_once_with(
        ["node-0", "node-0"]
    )
    assert worker.llm.collective_rpc.await_args_list == [
        call("bind_numa", args=()),
        call("report_node_hostname", args=()),
    ]


def test_async_sparse_refit_exposes_zmq_relay(monkeypatch) -> None:
    from nemo_rl.models.generation.vllm import vllm_sparse_refit as refit_module

    server = MagicMock()
    server_type = MagicMock(return_value=server)
    monkeypatch.setattr(refit_module, "ZmqSparseRefitServer", server_type)
    monkeypatch.setattr(refit_module, "_get_free_port_local", lambda *_args: 12345)
    monkeypatch.setattr(refit_module, "_get_node_ip_local", lambda: "10.0.0.1")
    config = {
        "vllm_cfg": {
            "async_engine": True,
            "zmq_refit_server_port": None,
            "http_refit_api_key_env_var": None,
        }
    }

    with _sparse_refit_receiver(async_engine=True, config=config) as receiver:
        assert receiver.start_zmq_sparse_refit_relay(["http://receiver"]) == (
            "tcp://10.0.0.1:12345"
        )
        server.start.assert_called_once_with()

        receiver.stop_zmq_sparse_refit_relay()
        server.close.assert_called_once_with()
        assert receiver._zmq_refit_server is None
