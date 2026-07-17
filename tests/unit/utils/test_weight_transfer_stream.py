# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
import requests
import torch
import zstandard

from nemo_rl.models.generation.vllm.config import VllmRefitConfig
from nemo_rl.utils import (
    weight_transfer_http,
    weight_transfer_stream,
    weight_transfer_zmq,
)
from nemo_rl.utils.weight_transfer_sparse_codec import (
    DeltaCompressionTracker,
    encode_sparse_infos,
    sparse_locations_for_item,
)
from nemo_rl.utils.weight_transfer_stream import (
    SparseRefitTransport,
    download_s3_refit_payload,
    sparse_export_chunk_size,
    sparse_payload_checksum,
)
from nemo_rl.utils.weight_transfer_zmq import (
    ZmqSparseRefitClient,
    ZmqSparseRefitServer,
)


def _refit_config(
    *,
    encoding: str = "overwrite",
    bucket_bytes: int = 1024,
    verify_samples: int = 0,
    s3_bucket: str | None = None,
    s3_region: str = "us-east-1",
    s3_prefix: str = "nemo-rl-refit",
    s3_export_bytes: int = 64 * 1024**2,
    zmq_export_bytes: int = 256 * 1024**2,
    s3_encode_workers: int = 8,
    zmq_encode_workers: int = 8,
    s3_transfer_workers: int = 32,
    zmq_transfer_workers: int = 4,
    zmq_retries: int = 3,
) -> VllmRefitConfig:
    return VllmRefitConfig(
        delta_compression={
            "encoding": encoding,
            "sparse_bucket_size_bytes": bucket_bytes,
            "export_chunk_bytes": {
                "s3": s3_export_bytes,
                "zmq": zmq_export_bytes,
            },
        },
        storage={
            "s3_bucket": s3_bucket,
            "s3_region": s3_region,
            "s3_prefix": s3_prefix,
        },
        baseline={"in_memory": True},
        tuning={
            "encode_workers": {
                "s3": s3_encode_workers,
                "zmq": zmq_encode_workers,
            },
            "transfer_workers": {
                "s3": s3_transfer_workers,
                "zmq": zmq_transfer_workers,
            },
            "zmq_retries": zmq_retries,
        },
        verify_samples_per_payload=verify_samples,
    )


class _SparsePipelineTracker:
    sparse_bucket_size_bytes = 1
    refit_config = _refit_config(
        bucket_bytes=1,
        zmq_export_bytes=1,
        zmq_encode_workers=1,
    )

    @staticmethod
    def prepare_sparse_delta_payload(chunk):
        count = sum(tensor.numel() for _, tensor in chunk)
        payload = encode_sparse_infos(
            (
                (
                    name,
                    tensor,
                    torch.arange(tensor.numel()),
                    tensor.reshape(-1),
                    "overwrite",
                )
                for name, tensor in chunk
            )
        )
        return payload, count, count


class _BaselineNamesTracker:
    sparse_bucket_size_bytes = 4
    refit_config = _refit_config(bucket_bytes=4, zmq_export_bytes=4)

    def __init__(self) -> None:
        self.names = []

    def snapshot_baseline(self, chunk) -> None:
        self.names.extend(name for name, _tensor in chunk)


def _stream_sparse_test_payloads(tensors, send_payload):
    cleaned = []
    transport = SparseRefitTransport(
        name="zmq",
        transfer_workers=1,
        send=lambda body, payload_id, _candidates: send_payload(body, payload_id),
        cleanup=lambda: cleaned.append(True),
    )
    try:
        return weight_transfer_stream.stream_sparse_delta_payloads(
            tensors,
            delta_tracker=_SparsePipelineTracker(),
            transport=transport,
            shard_rank=0,
            shard_count=1,
        )
    finally:
        assert cleaned


def _delta_tracker(
    encoding: str = "overwrite", **config: Any
) -> DeltaCompressionTracker:
    return DeltaCompressionTracker(_refit_config(encoding=encoding, **config))


def _baseline_names(tensors, *, rank: int):
    tracker = _BaselineNamesTracker()
    weight_transfer_stream.init_sparse_delta_baseline_from_iterator(
        tensors,
        delta_tracker=tracker,
        shard_rank=rank,
        shard_count=2,
        transport="zmq",
    )
    return tracker.names


def test_delta_tracker_commits_only_successful_syncs() -> None:
    tracker = _delta_tracker()
    tensor = torch.tensor([1.0, 2.0, 3.0])
    tracker.snapshot_baseline([("weight", tensor)])
    tensor[1] += 4

    assert tracker.prepare_sparse_delta_payload([("weight", tensor)])[0][2]
    tracker.on_sync_failed()
    assert tracker.prepare_sparse_delta_payload([("weight", tensor)])[0][2]
    tracker.on_sync_succeeded()
    assert not tracker.prepare_sparse_delta_payload([("weight", tensor)])[0][2]


def test_delta_tracker_emits_bounded_verification_budget() -> None:
    tracker = _delta_tracker(verify_samples=2)
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    tracker.snapshot_baseline([("weight", tensor)])
    tensor[[1, 3]] += 1

    (_, _, metadata), changed, total = tracker.prepare_sparse_delta_payload(
        [("weight", tensor)]
    )

    assert metadata[0]["verification_samples"] == 2
    assert metadata[0]["operation"] == "overwrite"
    assert (changed, total) == (2, 4)


def test_delta_tracker_commits_exact_source_baseline() -> None:
    tracker = _delta_tracker()
    tensor = torch.tensor([1.0])
    tracker.snapshot_baseline([("weight", tensor)])
    tensor.add_(0.001)

    (_, value_groups, _), _, _ = tracker.prepare_sparse_delta_payload(
        [("weight", tensor)]
    )
    assert torch.equal(value_groups[0], tensor.view(torch.int32))
    tracker.on_sync_succeeded()
    tracker.prepare_sparse_delta_payload([("weight", tensor)])

    assert torch.equal(tracker.baseline["weight"], tensor)


def test_delta_tracker_xor_encodes_against_baseline() -> None:
    tracker = _delta_tracker("xor")
    tensor = torch.tensor([1.0, 2.0, 3.0])
    baseline = tensor.clone()
    tracker.snapshot_baseline([("weight", tensor)])
    tensor[[0, 2]] = torch.tensor([4.0, 5.0])

    (_, value_groups, metadata), changed, total = tracker.prepare_sparse_delta_payload(
        [("weight", tensor)]
    )
    locations = torch.tensor([0, 2])
    expected = tensor.view(torch.int32)[locations].bitwise_xor(
        baseline.view(torch.int32)[locations]
    )

    assert torch.equal(value_groups[0], expected)
    assert metadata[0]["operation"] == "xor"
    assert (changed, total) == (2, 3)
    tracker.on_sync_succeeded()
    tracker.prepare_sparse_delta_payload([("weight", tensor)])
    assert torch.equal(tracker.baseline["weight"], tensor)


def test_delta_tracker_uses_overwrite_for_receiver_incompatible_weights() -> None:
    tracker = _delta_tracker("xor")
    weight = torch.tensor([1.0])
    scale = torch.tensor([2.0])
    tracker.snapshot_baseline([("weight", weight), ("scale", scale)])
    tracker.overwrite_names = frozenset({"scale"})
    weight.add_(1)
    scale.add_(1)

    (_, _, metadata), _, _ = tracker.prepare_sparse_delta_payload(
        [("weight", weight), ("scale", scale)]
    )

    assert [item["operation"] for item in metadata] == ["xor", "overwrite"]


def test_sparse_index_encoding_preserves_uint64_locations() -> None:
    locations = torch.tensor([0, 2**32 + 5])
    packed, _, metadata = encode_sparse_infos(
        [
            (
                "weight",
                torch.empty(2),
                locations,
                torch.ones(2, dtype=torch.int32),
                "overwrite",
            )
        ],
    )

    decoded = sparse_locations_for_item(metadata[0], packed, device="cpu")
    assert torch.equal(decoded, locations)


def test_sparse_index_encoding_preserves_uint32_locations() -> None:
    locations = torch.tensor([0, 2**16 + 1])
    packed, _, metadata = encode_sparse_infos(
        [
            (
                "weight",
                torch.empty(2),
                locations,
                torch.ones(2, dtype=torch.int32),
                "overwrite",
            )
        ],
    )

    assert metadata[0]["index_encoding"] == "deltas"
    assert packed.numel() == 2 * 4
    decoded = sparse_locations_for_item(metadata[0], packed, device="cpu")
    assert torch.equal(decoded, locations)


@pytest.mark.parametrize("encoding", ["xor", "overwrite"])
def test_delta_tracker_encodes_fp8_weight_and_scale_bits(encoding: str) -> None:
    tracker = _delta_tracker(encoding, verify_samples=2)
    weight = torch.tensor([0x38, 0x40, 0x48], dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    scale = torch.tensor([1.0, 2.0], dtype=torch.float32)
    tracker.snapshot_baseline([("weight", weight), ("weight_scale_inv", scale)])
    weight.view(torch.uint8)[1] = 0x41
    scale[0] = 1.5

    (locations, value_groups, metadata), changed, total = (
        tracker.prepare_sparse_delta_payload(
            [("weight", weight), ("weight_scale_inv", scale)]
        )
    )

    assert (changed, total) == (2, 5)
    assert len(value_groups) == 2
    assert [item["operation"] for item in metadata] == [encoding, encoding]
    assert [item["dtype"] for item in metadata] == ["float8_e4m3fn", "float32"]
    assert [item["verification_samples"] for item in metadata] == [1, 1]
    assert [
        sparse_locations_for_item(item, locations, device="cpu").tolist()
        for item in metadata
    ] == [
        [1],
        [0],
    ]

    tracker.on_sync_succeeded()
    assert not tracker.prepare_sparse_delta_payload(
        [("weight", weight), ("weight_scale_inv", scale)]
    )[0][2]


def test_refit_config_rejects_arithmetic_encoding() -> None:
    with pytest.raises(ValueError, match="Input should be 'xor' or 'overwrite'"):
        _delta_tracker("add")


def test_s3_download_verifies_checksum(monkeypatch) -> None:
    compressed = zstandard.ZstdCompressor().compress(b"payload")
    monkeypatch.setattr(
        weight_transfer_stream,
        "_get_manifest_s3_store",
        lambda *_args: SimpleNamespace(get=lambda _key: bytearray(compressed)),
    )
    manifest = {
        "bucket": "bucket",
        "region": "region",
        "key": "key",
        "checksum": sparse_payload_checksum(compressed),
    }

    assert download_s3_refit_payload(manifest) == b"payload"
    manifest["checksum"] = "0" * 32
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_s3_refit_payload(manifest)


def test_refit_http_session_does_not_retry_application_errors() -> None:
    retry = weight_transfer_http.refit_http_session().get_adapter("http://").max_retries

    assert 500 not in retry.status_forcelist
    assert {502, 503, 504} <= set(retry.status_forcelist)


def test_refit_http_sessions_share_connection_pool_across_threads() -> None:
    barrier = threading.Barrier(4)

    def adapter(_):
        barrier.wait()
        return weight_transfer_http.refit_http_session().get_adapter("http://")

    with ThreadPoolExecutor(max_workers=4) as executor:
        adapters = list(executor.map(adapter, range(4)))

    assert all(adapter is adapters[0] for adapter in adapters)


def test_refit_http_error_preserves_non_json_status_and_body(monkeypatch) -> None:
    response = requests.Response()
    response.status_code = 500
    response._content = b"<html>gateway failure</html>"
    session = SimpleNamespace(post=lambda *_args, **_kwargs: response)
    monkeypatch.setattr(weight_transfer_http, "refit_http_session", lambda: session)

    with pytest.raises(RuntimeError, match="HTTP 500: <html>gateway failure"):
        weight_transfer_http.post_vllm_refit_endpoints(
            ["http://receiver/refit"], {}, api_key=None, timeout_s=1.0
        )


def test_sparse_export_finishes_before_blocked_transfers() -> None:
    exported = threading.Event()
    release_transfers = threading.Event()
    result = []

    def tensors():
        for index in range(4):
            yield f"weight-{index}", torch.ones(1)
        exported.set()

    def send_payload(_body, _payload_index):
        assert release_transfers.wait(timeout=5.0)
        return {"receiver": {}}

    def run():
        result.append(_stream_sparse_test_payloads(tensors(), send_payload))

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert exported.wait(timeout=2.0)
    finally:
        release_transfers.set()
        thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert result == [{"payloads": 4, "changed_elements": 4, "total_elements": 4}]


def test_sparse_export_finishes_before_transfer_error() -> None:
    exported = []

    def tensors():
        for index in range(4):
            exported.append(index)
            yield f"weight-{index}", torch.ones(1)

    def fail_transfer(_body, _payload_index):
        raise RuntimeError("transfer failed")

    with pytest.raises(RuntimeError, match="transfer failed"):
        _stream_sparse_test_payloads(tensors(), fail_transfer)

    assert exported == list(range(4))


def test_sparse_transport_cleanup_runs_on_transfer_workers() -> None:
    class Transport:
        name = "zmq"
        transfer_workers = 2

        def __init__(self) -> None:
            self.barrier = threading.Barrier(2)
            self.send_threads = set()
            self.cleanup_threads = set()

        def send(self, _body, _payload_id, _verification_candidates):
            self.send_threads.add(threading.get_ident())
            self.barrier.wait(timeout=5.0)
            return {"receiver": {}}

        def cleanup(self) -> None:
            self.cleanup_threads.add(threading.get_ident())

    transport = Transport()
    result = weight_transfer_stream.stream_sparse_delta_payloads(
        [(f"weight-{index}", torch.ones(1)) for index in range(2)],
        delta_tracker=_SparsePipelineTracker(),
        transport=transport,
        shard_rank=0,
        shard_count=1,
    )

    assert result["payloads"] == 2
    assert transport.send_threads == transport.cleanup_threads


def test_sparse_stream_coalesces_export_chunks() -> None:
    tracker = _delta_tracker(
        bucket_bytes=8,
        zmq_export_bytes=4,
        zmq_encode_workers=2,
    )
    tensors = [(f"weight-{index}", torch.zeros(1)) for index in range(4)]
    tracker.snapshot_baseline(tensors)
    for _, tensor in tensors:
        tensor.fill_(1)
    payloads = {}

    def send(body, payload_index):
        raw = zstandard.ZstdDecompressor().decompress(body)
        payloads[payload_index] = torch.load(
            io.BytesIO(raw), map_location="cpu", weights_only=True
        )
        return {"receiver": {}}

    cleaned = []
    transport = SparseRefitTransport(
        "zmq",
        1,
        lambda body, payload_id, _candidates: send(body, payload_id),
        lambda: cleaned.append(True),
    )
    result = weight_transfer_stream.stream_sparse_delta_payloads(
        tensors,
        delta_tracker=tracker,
        transport=transport,
        shard_rank=0,
        shard_count=1,
    )

    assert cleaned
    assert result == {"payloads": 2, "changed_elements": 4, "total_elements": 4}
    payload_names = [
        [item["name"] for item in payloads[index][2]] for index in sorted(payloads)
    ]
    assert all(len(names) == 2 for names in payload_names)
    assert sorted(name for names in payload_names for name in names) == [
        f"weight-{index}" for index in range(4)
    ]
    for locations, value_groups, metadata in payloads.values():
        for item in metadata:
            assert sparse_locations_for_item(
                item, locations, device="cpu"
            ).tolist() == [0]
            assert value_groups[item["value_group"]][
                item["value_start"] : item["value_end"]
            ].tolist() == [1065353216]


def test_sparse_export_chunk_defaults_are_transport_specific() -> None:
    tracker = _delta_tracker(bucket_bytes=1024**3)

    assert sparse_export_chunk_size(tracker, "s3") == 64 * 1024**2
    assert sparse_export_chunk_size(tracker, "zmq") == 256 * 1024**2


def test_sparse_baseline_snapshots_only_owned_export_chunks(capsys) -> None:
    tensors = [(f"weight-{index}", torch.tensor([float(index)])) for index in range(4)]

    assert _baseline_names(tensors, rank=1) == ["weight-1", "weight-3"]
    assert "chunks=4" in capsys.readouterr().out


def test_sparse_stream_sends_only_owned_export_chunks() -> None:
    sent = []
    transport = SparseRefitTransport(
        "zmq",
        1,
        lambda _body, payload_id, _candidates: (
            sent.append(payload_id) or {"receiver": {}}
        ),
        lambda: None,
    )

    result = weight_transfer_stream.stream_sparse_delta_payloads(
        [(f"weight-{index}", torch.ones(1)) for index in range(4)],
        delta_tracker=_SparsePipelineTracker(),
        transport=transport,
        shard_rank=1,
        shard_count=2,
    )

    assert result == {"payloads": 2, "changed_elements": 2, "total_elements": 2}
    assert sorted(sent) == [0, 1]


def test_s3_manifest_transport_validates_configuration(monkeypatch) -> None:
    tracker = SimpleNamespace(refit_config=_refit_config(s3_bucket="bucket"))
    kwargs = {
        "iterator": (),
        "delta_tracker": tracker,
        "transfer_id": "transfer",
        "api_key_env_var": None,
        "timeout_s": 1.0,
        "shard_rank": 0,
        "shard_count": 1,
    }
    with pytest.raises(ValueError, match="URL is required"):
        weight_transfer_stream.stream_sparse_delta_payloads_via_s3_manifest(
            refit_targets=[], **kwargs
        )

    tracker.refit_config = _refit_config()
    with pytest.raises(RuntimeError, match="refit_cfg.storage.s3_bucket"):
        weight_transfer_stream.stream_sparse_delta_payloads_via_s3_manifest(
            refit_targets=["http://receiver"], **kwargs
        )


def test_s3_manifest_transport_uploads_notifies_and_deletes(monkeypatch) -> None:
    operations = []
    posts = []

    class Store:
        bucket = "bucket"
        region = "us-west-2"

        def put(self, key, body) -> None:
            operations.append(("put", key, body))

        def delete(self, key) -> None:
            operations.append(("delete", key))

    store = Store()
    monkeypatch.setenv("NRL_TEST_REFIT_KEY", "secret")
    monkeypatch.setattr(
        weight_transfer_stream,
        "_get_manifest_s3_store",
        lambda *_args: store,
    )

    def post(endpoints, manifest, **kwargs):
        posts.append((endpoints, manifest, kwargs))
        return [
            {"ok": True, "receiver_total_s": 1.0},
            {"ok": True, "receiver_total_s": 2.0},
        ]

    def stream(iterator, **kwargs):
        assert list(iterator) == [("weight", torch.tensor([1.0]))]
        transport = kwargs["transport"]
        assert transport.name == "s3"
        assert transport.transfer_workers == 3
        response = transport.send(b"payload", 3, 4)
        assert response["receiver"] == {"receiver_total_s": 2.0}
        transport.cleanup()
        return {"payloads": 1, "changed_elements": 1, "total_elements": 1}

    monkeypatch.setattr(weight_transfer_stream, "post_vllm_refit_endpoints", post)
    monkeypatch.setattr(weight_transfer_stream, "stream_sparse_delta_payloads", stream)

    result = weight_transfer_stream.stream_sparse_delta_payloads_via_s3_manifest(
        [("weight", torch.tensor([1.0]))],
        delta_tracker=SimpleNamespace(
            refit_config=_refit_config(
                s3_bucket=store.bucket,
                s3_region=store.region,
                s3_prefix="/prefix/",
                s3_transfer_workers=3,
            )
        ),
        refit_targets=[" http://receiver-a/ ", "http://receiver-b"],
        transfer_id="transfer",
        api_key_env_var="NRL_TEST_REFIT_KEY",
        timeout_s=7.0,
        shard_rank=2,
        shard_count=4,
    )

    key = "prefix/transfer/000002/000003.pt"
    assert result == {"payloads": 1, "changed_elements": 1, "total_elements": 1}
    assert operations == [("put", key, b"payload"), ("delete", key)]
    assert posts == [
        (
            [
                "http://receiver-a/nemo-rl/refit/s3-manifest",
                "http://receiver-b/nemo-rl/refit/s3-manifest",
            ],
            {
                "bucket": store.bucket,
                "region": store.region,
                "key": key,
                "checksum": sparse_payload_checksum(b"payload"),
                "verification_candidates": 4,
            },
            {"api_key": "secret", "timeout_s": 7.0},
        )
    ]


def test_zmq_stream_routes_shards_and_closes_clients(monkeypatch) -> None:
    created = []
    sent = []
    closed = []
    monkeypatch.setenv("NRL_TEST_REFIT_KEY", "secret")

    class Client:
        def __init__(self, address, **kwargs) -> None:
            created.append((address, kwargs))

        def send_payload(self, **kwargs):
            sent.append(kwargs)
            return {"ok": True, "receiver_total_s": 0.5}

        def close(self) -> None:
            closed.append(True)

    def stream(_iterator, **kwargs):
        transport = kwargs["transport"]
        assert transport.name == "zmq"
        assert transport.transfer_workers == 2
        for payload_id in range(2):
            response = transport.send(
                f"body-{payload_id}".encode(), payload_id, payload_id + 1
            )
            assert response["receiver"]["ok"]
        transport.cleanup()
        return {"payloads": 2, "changed_elements": 2, "total_elements": 2}

    monkeypatch.setattr(weight_transfer_zmq, "ZmqSparseRefitClient", Client)
    monkeypatch.setattr(weight_transfer_zmq, "stream_sparse_delta_payloads", stream)

    with pytest.raises(ValueError, match="address is required"):
        weight_transfer_zmq.stream_sparse_delta_payloads_via_zmq(
            (),
            delta_tracker=SimpleNamespace(),
            refit_targets=[],
            transfer_id="transfer",
            api_key_env_var=None,
            timeout_s=1.0,
            shard_rank=0,
            shard_count=1,
        )

    kwargs = {
        "iterator": (),
        "delta_tracker": SimpleNamespace(
            refit_config=_refit_config(zmq_transfer_workers=2)
        ),
        "refit_targets": ["tcp://receiver-a", " tcp://receiver-b "],
        "transfer_id": "transfer",
        "api_key_env_var": "NRL_TEST_REFIT_KEY",
        "timeout_s": 7.0,
        "shard_rank": 3,
        "shard_count": 4,
    }
    assert (
        weight_transfer_zmq.stream_sparse_delta_payloads_via_zmq(**kwargs)["payloads"]
        == 2
    )
    assert (
        weight_transfer_zmq.stream_sparse_delta_payloads_via_zmq(**kwargs)["payloads"]
        == 2
    )

    assert created == 2 * [
        (
            "tcp://receiver-b",
            {
                "timeout_s": 7.0,
                "producer_id": 3,
                "retries": 3,
                "api_key": "secret",
            },
        )
    ]
    assert closed == [True, True]
    assert [item["payload_id"] for item in sent] == [0, 1, 0, 1]
    assert [item["verification_candidates"] for item in sent] == [1, 2, 1, 2]
    assert all(
        item["checksum"] == sparse_payload_checksum(item["body"]) for item in sent
    )


def test_zmq_client_filters_replies_rejects_nack_and_retries(monkeypatch) -> None:
    class Socket:
        def __init__(self, replies=(), send_failures=0) -> None:
            self.replies = list(replies)
            self.send_failures = send_failures
            self.sent = []

        def send_multipart(self, frames, **_kwargs) -> None:
            self.sent.append(frames)
            if self.send_failures:
                self.send_failures -= 1
                raise weight_transfer_zmq.zmq.Again()

        def poll(self, *_args) -> bool:
            return bool(self.replies)

        def recv_multipart(self):
            return self.replies.pop(0)

    def client(socket) -> ZmqSparseRefitClient:
        result = ZmqSparseRefitClient.__new__(ZmqSparseRefitClient)
        result._address = "tcp://receiver"
        result._timeout_ms = 1000
        result._producer_id = 4
        result._retries = 1
        result._api_key = "secret"
        result._socket = socket
        return result

    success = {
        "ok": True,
        "transfer_id": "transfer-a",
        "producer_id": 4,
        "payload_id": 7,
    }
    socket = Socket(
        [
            [b"malformed"],
            [
                b"ACK",
                json.dumps({**success, "transfer_id": "stale"}).encode(),
            ],
            [b"ACK", json.dumps(success).encode()],
        ]
    )
    assert _send_zmq_payload(client(socket), 7, b"body") == success
    assert json.loads(socket.sent[0][1])["api_key"] == "secret"

    denied = Socket([[b"NACK", json.dumps({"ok": False, "error": "denied"}).encode()]])
    with pytest.raises(RuntimeError, match="denied"):
        _send_zmq_payload(client(denied), 7, b"body")

    with pytest.raises(TimeoutError, match="payload 7"):
        _send_zmq_payload(client(Socket(send_failures=2)), 7, b"body")


def test_zmq_server_rejects_malformed_messages() -> None:
    server = ZmqSparseRefitServer.__new__(ZmqSparseRefitServer)
    server._token = "secret"
    body = b"body"
    metadata = {
        "protocol": "nemo-rl-sparse-zmq-v1",
        "api_key": "secret",
        "transfer_id": "transfer",
        "producer_id": 0,
        "payload_id": 1,
        "checksum": sparse_payload_checksum(body),
        "verification_candidates": 2,
    }

    def frames(kind: bytes = b"DATA", **updates: object) -> list[bytes]:
        return [b"id", kind, json.dumps({**metadata, **updates}).encode(), body]

    for message, error, match in (
        ([], ValueError, "Expected 4"),
        (frames(b"OTHER"), ValueError, "Unsupported ZeroMQ sparse refit message"),
        (frames(protocol="other"), ValueError, "protocol"),
        (frames(api_key="wrong"), PermissionError, "authentication"),
        (frames(transfer_id=""), ValueError, "identity"),
        (frames(checksum=""), ValueError, "identity"),
        (frames(verification_candidates=-1), ValueError, "identity"),
    ):
        with pytest.raises(error, match=match):
            server._parse_data_message(message)

    assert server._parse_data_message(frames())[1] == ("transfer", 0, 1)


def _send_zmq_payload(
    client: ZmqSparseRefitClient,
    payload_id: int,
    body: bytes,
    checksum: str | None = None,
    transfer_id: str = "transfer-a",
) -> dict[str, object]:
    return client.send_payload(
        transfer_id=transfer_id,
        payload_id=payload_id,
        checksum=checksum or sparse_payload_checksum(body),
        verification_candidates=2,
        body=body,
    )


def test_zmq_sparse_refit_relay_fans_out_and_rejects_corruption(monkeypatch) -> None:
    monkeypatch.setenv("NRL_TEST_REFIT_KEY", "secret")
    received = [[] for _ in range(4)]

    def apply(items):
        def apply_payload(body, metadata):
            if metadata["checksum"] != sparse_payload_checksum(body):
                raise ValueError("checksum mismatch")
            items.append((dict(metadata), body))
            return {"ok": True, "receiver_total_s": 0.25}

        return apply_payload

    relays = [
        ZmqSparseRefitServer(
            apply(items),
            bind_address="tcp://127.0.0.1:*",
            api_key_env_var="NRL_TEST_REFIT_KEY",
            timeout_s=5.0,
            tuning=_refit_config().tuning,
        )
        for items in received
    ]
    addresses = [relay.start() for relay in relays]
    for relay, address in zip(relays, addresses, strict=True):
        relay.configure_tree(addresses, own_address=address)
    unauthenticated_client = ZmqSparseRefitClient(
        addresses[0],
        timeout_s=5.0,
        producer_id=2,
        retries=3,
    )
    client = ZmqSparseRefitClient(
        addresses[0],
        timeout_s=5.0,
        producer_id=3,
        retries=3,
        api_key="secret",
    )
    body = b"compressed sparse payload"
    checksum = sparse_payload_checksum(body)
    try:
        with pytest.raises(RuntimeError, match="authentication failed"):
            _send_zmq_payload(unauthenticated_client, 6, body)
        first = _send_zmq_payload(client, 7, body)
        duplicate = _send_zmq_payload(client, 7, body)
        assert first["ok"] and first["staged"]
        assert duplicate["ok"] and duplicate["staged"]
        for relay in relays:
            flushed = relay.flush("transfer-a", expected_payloads=1)
            assert flushed["payloads"] == 1
            assert flushed["receiver_total_s"] == 0.25
        assert [len(items) for items in received] == [1] * 4
        for items in received:
            metadata, posted_body = items[0]
            assert posted_body == body
            assert metadata["transfer_id"] == "transfer-a"
            assert metadata["producer_id"] == 3
            assert metadata["payload_id"] == 7
            assert metadata["checksum"] == checksum
            assert metadata["verification_candidates"] == 2
            assert metadata["api_key"] == "secret"

        with pytest.raises(RuntimeError, match="already flushed"):
            _send_zmq_payload(client, 8, body)
        assert _send_zmq_payload(client, 8, body, "0" * 32, "transfer-b")["staged"]
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            relays[0].flush("transfer-b")
    finally:
        unauthenticated_client.close()
        client.close()
        for relay in relays:
            relay.close()
