# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""IPC weight streaming when one parameter exceeds the staging buffer.

The staging buffers are sized from *free memory*
(``NRL_REFIT_BUFFER_MEMORY_RATIO``, default 0.3, halved again for ping-pong)
with no floor at the largest parameter, so a large embedding can be bigger than
a single buffer and cannot be packed at all. That used to abort the refit with
``AssertionError: Parameter ... too large for buffer``; DeepSeek-V3 hit it with
``model.embed_tokens.weight`` at 1.73 GiB against a 1.65 GiB buffer.

These tests run on CPU tensors with a stub socket -- no GPU, no Ray -- because
the packing/hand-off logic is plain Python around a byte buffer.
"""

import torch

from nemo_rl.models.policy import utils

# Small enough to keep the test instant; the arithmetic is scale-free.
BUFFER_BYTES = 4096
ALIGNMENT = 512


class FakeSocket:
    """Records the payloads a policy worker streams, and ACKs each one."""

    def __init__(self):
        self.payloads = []
        self.completed = False
        self.pending_acks = 0

    def send_pyobj(self, payload):
        # The receiver ACKs the end-of-stream marker too, not just data groups.
        if payload is utils.IPCProtocol.COMPLETE:
            self.completed = True
            self.pending_acks += 1
            return
        _handle, param_names, used_bytes = payload
        self.payloads.append((list(param_names), used_bytes))
        self.pending_acks += 1

    def recv(self):
        # A receiver only ACKs a group it was actually sent. If the streamer
        # waits for an ACK it is not owed, that is a protocol bug worth failing
        # on rather than deadlocking.
        assert self.pending_acks > 0, "streamer waited for an ACK it was not owed"
        self.pending_acks -= 1
        return b""

    def getsockopt(self, _opt):
        return 0


def _stream(monkeypatch, params, buffer_size_bytes=BUFFER_BYTES):
    """Run the streamer over ``params`` with CUDA calls stubbed out."""
    monkeypatch.setattr(utils, "get_handle_from_tensor", lambda tensor: ("handle",))
    # send_buffer_group_overlap synchronizes the current CUDA stream; on a
    # CPU-only box there is none.
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *a, **k: _NullStream(), raising=False
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    socket = FakeSocket()
    utils.stream_weights_via_ipc_zmq_impl(
        params_generator=iter(params),
        buffer_size_bytes=buffer_size_bytes,
        zmq_socket=socket,
        rank=0,
        worker_name="test-worker",
    )
    return socket


class _NullStream:
    def synchronize(self):
        pass


def _param(name, nbytes):
    return name, torch.zeros(nbytes, dtype=torch.uint8)


def test_oversized_parameter_is_streamed_instead_of_aborting(monkeypatch):
    """The parameter that used to trip the assertion is delivered on its own."""
    # buffer_size_bytes is halved for ping-pong, so the usable buffer is 2048.
    # 3072 exceeds it, exactly like embed_tokens.weight vs its staging buffer.
    params = [
        _param("model.layers.0.weight", 512),
        _param("model.embed_tokens.weight", 3072),
        _param("model.layers.1.weight", 512),
    ]

    socket = _stream(monkeypatch, params)

    streamed = [name for names, _ in socket.payloads for name in names]
    assert streamed == [p[0] for p in params], (
        "every parameter must be delivered exactly once, in order"
    )
    assert socket.completed, "the stream must still be terminated with COMPLETE"
    assert socket.pending_acks == 0, "every sent group must be ACKed"

    # The oversized parameter cannot share a group with anything else.
    oversized_group = [
        names for names, _ in socket.payloads if "model.embed_tokens.weight" in names
    ]
    assert oversized_group == [["model.embed_tokens.weight"]]


def test_oversized_parameter_alone(monkeypatch):
    """A single oversized parameter needs no preceding group to flush."""
    socket = _stream(monkeypatch, [_param("model.embed_tokens.weight", 3072)])

    assert socket.payloads == [(["model.embed_tokens.weight"], 3072)]
    assert socket.completed
    assert socket.pending_acks == 0


def test_consecutive_oversized_parameters(monkeypatch):
    """Back-to-back oversized parameters each get their own buffer and ACK."""
    params = [_param("a.weight", 3072), _param("b.weight", 4096)]

    socket = _stream(monkeypatch, params)

    assert [names for names, _ in socket.payloads] == [["a.weight"], ["b.weight"]]
    assert socket.pending_acks == 0


def test_parameters_that_fit_are_still_batched(monkeypatch):
    """Regression guard: the common path must keep packing many params per group.

    This is the behaviour every currently-passing refit relies on, so it must
    not change -- the oversized branch is only reachable where the old code
    raised.
    """
    params = [_param(f"layer.{i}.weight", 512) for i in range(8)]

    socket = _stream(monkeypatch, params)

    assert len(socket.payloads) < len(params), (
        "parameters that fit must be batched, not sent one per group"
    )
    streamed = [name for names, _ in socket.payloads for name in names]
    assert streamed == [p[0] for p in params]
    assert socket.pending_acks == 0


def test_alignment_is_what_decides_oversized(monkeypatch):
    """A parameter is oversized by its *aligned* size, not its raw size."""
    # 2048 raw fits the 2048-byte usable buffer exactly; 2049 aligns up to 2560.
    assert utils.calculate_aligned_size(2048, ALIGNMENT) == 2048
    assert utils.calculate_aligned_size(2049, ALIGNMENT) == 2560

    socket = _stream(monkeypatch, [_param("exact.weight", 2048)])
    assert socket.payloads == [(["exact.weight"], 2048)]
