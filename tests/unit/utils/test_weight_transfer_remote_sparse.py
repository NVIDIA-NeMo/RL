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

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import torch
import zstandard

from nemo_rl.utils import weight_transfer_remote_sparse, weight_transfer_zmq
from nemo_rl.utils.weight_transfer_remote_sparse import download_s3_refit_payload
from nemo_rl.utils.weight_transfer_sparse_codec import (
    DeltaCompressionTracker,
    _bytewise_diff_mask,
    encode_sparse_infos,
    sparse_locations_for_item,
)
from nemo_rl.utils.weight_transfer_zmq import (
    G_VLLM_REFIT_CHECKSUM_HEADER,
    G_VLLM_REFIT_PAYLOAD_HEADER,
    G_VLLM_REFIT_PRODUCER_HEADER,
    G_VLLM_REFIT_TRANSFER_HEADER,
    ZmqSparseRefitClient,
    ZmqSparseRefitServer,
    sparse_payload_checksum,
)


class _SparsePipelineTracker:
    sparse_bucket_size_bytes = 1

    @staticmethod
    def prepare_sparse_delta_payload(chunk):
        return (chunk, torch.ones(1), [1]), 1, 1


def _stream_sparse_test_payloads(tensors, send_payload):
    return weight_transfer_remote_sparse.stream_sparse_delta_payloads(
        tensors,
        delta_tracker=_SparsePipelineTracker(),
        transport="zmq",
        send_payload=send_payload,
        transfer_workers=1,
        shard_rank=0,
        shard_count=1,
    )


def _delta_tracker(encoding: str = "overwrite") -> DeltaCompressionTracker:
    return DeltaCompressionTracker(
        {"encoding": encoding, "sparse_bucket_size_bytes": 1024}
    )


def test_delta_tracker_commits_only_successful_syncs(monkeypatch) -> None:
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    tracker = _delta_tracker()
    tensor = torch.tensor([1.0, 2.0, 3.0])
    tracker.snapshot_baseline([("weight", tensor)])
    tensor[1] += 4

    assert tracker.prepare_sparse_delta_payload([("weight", tensor)])[0][2]
    tracker.on_sync_failed()
    assert tracker.prepare_sparse_delta_payload([("weight", tensor)])[0][2]
    tracker.on_sync_succeeded()
    assert not tracker.prepare_sparse_delta_payload([("weight", tensor)])[0][2]


def test_delta_tracker_emits_bounded_delta_samples(monkeypatch) -> None:
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    monkeypatch.setenv("NRL_REFIT_VERIFY_SAMPLES_PER_PAYLOAD", "2")
    tracker = _delta_tracker()
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    tracker.snapshot_baseline([("weight", tensor)])
    tensor[[1, 3]] += 1

    (_, _, metadata), changed, total = tracker.prepare_sparse_delta_payload(
        [("weight", tensor)]
    )

    assert metadata[0]["verification_locations"] == [1, 3]
    assert metadata[0]["verification_values"] == [
        int(tensor.view(torch.int32)[1]),
        int(tensor.view(torch.int32)[3]),
    ]
    assert metadata[0]["operation"] == "overwrite"
    assert (changed, total) == (2, 4)


def test_delta_tracker_commits_exact_source_baseline(monkeypatch) -> None:
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
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


def test_delta_tracker_xor_encodes_against_baseline(monkeypatch) -> None:
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
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


def test_bytewise_diff_mask_supports_float8() -> None:
    baseline = torch.tensor([0x38, 0x7F, 0x00], dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    current = baseline.clone()
    current.view(torch.uint8)[1] = 0x7E

    assert _bytewise_diff_mask(current, baseline).tolist() == [False, True, False]


@pytest.mark.parametrize("encoding", ["xor", "overwrite"])
def test_delta_tracker_encodes_fp8_weight_and_scale_bits(
    monkeypatch, encoding: str
) -> None:
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    monkeypatch.setenv("NRL_REFIT_VERIFY_SAMPLES_PER_PAYLOAD", "2")
    tracker = _delta_tracker(encoding)
    weight = torch.tensor([0x38, 0x40, 0x48], dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    scale = torch.tensor([1.0, 2.0], dtype=torch.float32)
    baseline_weight = weight.clone()
    baseline_scale = scale.clone()
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
    expected_weight = int(weight.view(torch.uint8)[1])
    expected_scale = int(scale.view(torch.int32)[0])
    if encoding == "xor":
        expected_weight ^= int(baseline_weight.view(torch.uint8)[1])
        expected_scale ^= int(baseline_scale.view(torch.int32)[0])
    assert [item["verification_values"] for item in metadata] == [
        [expected_weight],
        [expected_scale],
    ]
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


def test_delta_tracker_rejects_arithmetic_encoding() -> None:
    with pytest.raises(ValueError, match="Unsupported sparse-refit operation"):
        _delta_tracker("add")


def test_s3_download_verifies_checksum(monkeypatch) -> None:
    compressed = zstandard.ZstdCompressor().compress(b"payload")
    monkeypatch.setattr(
        weight_transfer_remote_sparse,
        "_get_manifest_s3_store",
        lambda *_args: SimpleNamespace(get=lambda _key: compressed),
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
    retry = (
        weight_transfer_remote_sparse.refit_http_session()
        .get_adapter("http://")
        .max_retries
    )

    assert 500 not in retry.status_forcelist
    assert {502, 503, 504} <= set(retry.status_forcelist)


def test_sparse_export_finishes_before_blocked_transfers(monkeypatch) -> None:
    monkeypatch.setenv("NRL_REFIT_ZMQ_ENCODE_WORKERS", "1")
    monkeypatch.setenv("NRL_REFIT_ZMQ_EXPORT_CHUNK_BYTES", "1")
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


def test_sparse_export_finishes_before_transfer_error(monkeypatch) -> None:
    monkeypatch.setenv("NRL_REFIT_ZMQ_ENCODE_WORKERS", "1")
    monkeypatch.setenv("NRL_REFIT_ZMQ_EXPORT_CHUNK_BYTES", "1")
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


def test_sparse_baseline_snapshots_only_owned_export_chunks(
    monkeypatch, capsys
) -> None:
    monkeypatch.setenv("NRL_REFIT_ZMQ_EXPORT_CHUNK_BYTES", "4")

    class Tracker:
        sparse_bucket_size_bytes = 4

        def __init__(self) -> None:
            self.names = []

        def snapshot_baseline(self, chunk) -> None:
            self.names.extend(name for name, _tensor in chunk)

    tracker = Tracker()
    weight_transfer_remote_sparse.init_sparse_delta_baseline_from_iterator(
        [(f"weight-{index}", torch.tensor([float(index)])) for index in range(4)],
        delta_tracker=tracker,
        shard_rank=1,
        shard_count=2,
        transport="zmq",
    )

    assert tracker.names == ["weight-1", "weight-3"]
    assert "chunks=4" in capsys.readouterr().out


def test_s3_manifest_transport_validates_configuration(monkeypatch) -> None:
    kwargs = {
        "iterator": (),
        "delta_tracker": SimpleNamespace(),
        "transfer_id": "transfer",
        "api_key_env_var": None,
        "timeout_s": 1.0,
        "shard_rank": 0,
        "shard_count": 1,
    }
    monkeypatch.setenv("NRL_REFIT_S3_BUCKET", "bucket")
    with pytest.raises(ValueError, match="URL is required"):
        weight_transfer_remote_sparse.stream_sparse_delta_payloads_via_s3_manifest(
            refit_targets=[], **kwargs
        )

    monkeypatch.delenv("NRL_REFIT_S3_BUCKET")
    with pytest.raises(RuntimeError, match="NRL_REFIT_S3_BUCKET"):
        weight_transfer_remote_sparse.stream_sparse_delta_payloads_via_s3_manifest(
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
    monkeypatch.setenv("NRL_REFIT_S3_BUCKET", store.bucket)
    monkeypatch.setenv("NRL_REFIT_S3_REGION", store.region)
    monkeypatch.setenv("NRL_REFIT_S3_PREFIX", "/prefix/")
    monkeypatch.setenv("NRL_REFIT_S3_UPLOAD_WORKERS", "3")
    monkeypatch.setenv("NRL_TEST_REFIT_KEY", "secret")
    monkeypatch.setattr(
        weight_transfer_remote_sparse,
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
        assert kwargs["transport"] == "s3"
        assert kwargs["transfer_workers"] == 3
        response = kwargs["send_payload"](b"payload", 3)
        assert response["receiver"] == {"receiver_total_s": 2.0}
        return {"payloads": 1, "changed_elements": 1, "total_elements": 1}

    monkeypatch.setattr(
        weight_transfer_remote_sparse, "post_vllm_refit_endpoints", post
    )
    monkeypatch.setattr(
        weight_transfer_remote_sparse, "stream_sparse_delta_payloads", stream
    )

    result = weight_transfer_remote_sparse.stream_sparse_delta_payloads_via_s3_manifest(
        [("weight", torch.tensor([1.0]))],
        delta_tracker=SimpleNamespace(),
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
            },
            {"api_key": "secret", "timeout_s": 7.0},
        )
    ]


def test_zmq_stream_routes_shards_and_reuses_clients(monkeypatch) -> None:
    created = []
    sent = []
    monkeypatch.setenv("NRL_TEST_REFIT_KEY", "secret")
    monkeypatch.setenv("NRL_REFIT_ZMQ_SEND_WORKERS", "2")
    monkeypatch.delattr(weight_transfer_zmq._ZMQ_LOCAL, "clients", raising=False)

    class Client:
        def __init__(self, address, **kwargs) -> None:
            created.append((address, kwargs))

        def send_payload(self, **kwargs):
            sent.append(kwargs)
            return {"ok": True, "receiver_total_s": 0.5}

    def stream(_iterator, **kwargs):
        assert kwargs["transport"] == "zmq"
        assert kwargs["transfer_workers"] == 2
        for payload_id in range(2):
            response = kwargs["send_payload"](f"body-{payload_id}".encode(), payload_id)
            assert response["receiver"]["ok"]
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
        "delta_tracker": SimpleNamespace(),
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

    assert created == [
        (
            "tcp://receiver-b",
            {"timeout_s": 7.0, "producer_id": 3, "api_key": "secret"},
        )
    ]
    assert [item["payload_id"] for item in sent] == [0, 1, 0, 1]
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

    monkeypatch.setenv("NRL_REFIT_ZMQ_RETRIES", "1")
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
    }

    def frames(kind: bytes = b"DATA", **updates: object) -> list[bytes]:
        return [b"id", kind, json.dumps({**metadata, **updates}).encode(), body]

    for message, error, match in (
        ([], ValueError, "Expected 4"),
        (frames(b"OTHER"), ValueError, "Unsupported ZeroMQ sparse refit message"),
        (frames(protocol="other"), ValueError, "protocol"),
        (frames(api_key="wrong"), PermissionError, "authentication"),
        (frames(transfer_id=""), ValueError, "identity"),
        (frames(checksum="wrong"), ValueError, "checksum mismatch"),
    ):
        with pytest.raises(error, match=match):
            server._parse_data_message(message)

    assert server._parse_data_message(frames())[1] == ("transfer", 0, 1)


def _receiver_server(received):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            received.append(
                ({key.lower(): value for key, value in self.headers.items()}, body)
            )
            response = json.dumps({"ok": True, "receiver_total_s": 0.25}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _send_zmq_payload(
    client: ZmqSparseRefitClient,
    payload_id: int,
    body: bytes,
    checksum: str | None = None,
) -> dict[str, object]:
    return client.send_payload(
        transfer_id="transfer-a",
        payload_id=payload_id,
        checksum=checksum or sparse_payload_checksum(body),
        body=body,
    )


def test_zmq_sparse_refit_relay_fans_out_and_rejects_corruption(monkeypatch) -> None:
    monkeypatch.setenv("NRL_TEST_REFIT_KEY", "secret")
    received = [[], []]
    receivers = [_receiver_server(items) for items in received]
    urls = [f"http://127.0.0.1:{server.server_port}" for server, _ in receivers]
    relay = ZmqSparseRefitServer(
        urls,
        bind_address="tcp://127.0.0.1:*",
        api_key_env_var="NRL_TEST_REFIT_KEY",
        timeout_s=5.0,
    )
    address = relay.start()
    unauthenticated_client = ZmqSparseRefitClient(
        address,
        timeout_s=5.0,
        producer_id=2,
    )
    client = ZmqSparseRefitClient(
        address,
        timeout_s=5.0,
        producer_id=3,
        api_key="secret",
    )
    body = b"compressed sparse payload"
    checksum = sparse_payload_checksum(body)
    try:
        with pytest.raises(RuntimeError, match="authentication failed"):
            _send_zmq_payload(unauthenticated_client, 6, body)
        first = _send_zmq_payload(client, 7, body)
        assert first["ok"]
        assert first["receiver_total_s"] == 0.25
        assert [len(items) for items in received] == [1, 1]
        for items in received:
            headers, posted_body = items[0]
            assert posted_body == body
            assert headers[G_VLLM_REFIT_TRANSFER_HEADER] == "transfer-a"
            assert headers[G_VLLM_REFIT_PRODUCER_HEADER] == "3"
            assert headers[G_VLLM_REFIT_PAYLOAD_HEADER] == "7"
            assert headers[G_VLLM_REFIT_CHECKSUM_HEADER] == checksum
            assert headers["x-nemo-rl-refit-key"] == "secret"

        with pytest.raises(RuntimeError, match="checksum mismatch"):
            _send_zmq_payload(client, 8, body, "0" * 32)
    finally:
        unauthenticated_client.close()
        client.close()
        relay.close()
        for server, thread in receivers:
            server.shutdown()
            thread.join(timeout=5.0)
            server.server_close()
