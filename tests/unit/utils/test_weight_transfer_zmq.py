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

import pytest

from nemo_rl.utils.weight_transfer_zmq import (
    G_VLLM_REFIT_CHECKSUM_HEADER,
    G_VLLM_REFIT_PAYLOAD_HEADER,
    G_VLLM_REFIT_PRODUCER_HEADER,
    G_VLLM_REFIT_TRANSFER_HEADER,
    ZmqSparseRefitClient,
    ZmqSparseRefitServer,
    sparse_payload_checksum,
)


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
    unauthenticated_client = ZmqSparseRefitClient(
        relay.start(),
        timeout_s=5.0,
        producer_id=2,
    )
    client = ZmqSparseRefitClient(
        relay.start(),
        timeout_s=5.0,
        producer_id=3,
        api_key="secret",
    )
    body = b"compressed sparse payload"
    checksum = sparse_payload_checksum(body)
    try:
        with pytest.raises(RuntimeError, match="authentication failed"):
            unauthenticated_client.send_payload(
                transfer_id="transfer-a",
                payload_id=6,
                checksum=checksum,
                body=body,
            )
        first = client.send_payload(
            transfer_id="transfer-a",
            payload_id=7,
            checksum=checksum,
            body=body,
        )
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
            client.send_payload(
                transfer_id="transfer-a",
                payload_id=8,
                checksum="0" * 32,
                body=body,
            )
    finally:
        unauthenticated_client.close()
        client.close()
        relay.close()
        for server, thread in receivers:
            server.shutdown()
            thread.join(timeout=5.0)
            server.server_close()
