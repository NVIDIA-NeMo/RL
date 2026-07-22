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

# trtllm_backend imports tensorrt_llm eagerly, so import it only inside tests.

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytestmark = pytest.mark.trtllm


def _extension(backend):
    extension = backend.NcclExtension.__new__(backend.NcclExtension)
    module = MagicMock()
    module._weights_removed = False
    model = MagicMock()
    model.modules.return_value = [module]
    model_loader = MagicMock()
    model_engine = SimpleNamespace(model=model, model_loader=model_loader)
    engine = MagicMock()
    engine.model_engine = model_engine
    engine.control_action.side_effect = lambda **_: contextlib.nullcontext()
    extension.engine = engine
    extension.device_id = 0
    extension.model_update_group = object()
    extension.state_dict_info = {"model.weight": (torch.Size([1]), torch.float32)}
    return extension, module, model, model_loader, engine


def _ipc_extension(backend):
    """Reuse the collective-path fixture, then add IPC-ZMQ-only state."""
    extension, _, _, model_loader, engine = _extension(backend)
    # Pre-set zmq_socket so maybe_init_zmq() short-circuits (no real bind).
    extension.zmq_socket = MagicMock()
    # Two fp32 weights: 8 B and 12 B, each 512-B aligned -> offsets 512, 1024.
    extension.state_dict_info = {
        "a": (torch.Size([2]), torch.float32),
        "b": (torch.Size([3]), torch.float32),
    }
    return extension, model_loader, engine


@pytest.mark.parametrize(
    ("drain", "recompute_kv"),
    [(True, False), (False, False), (False, True)],
)
def test_collective_refit_runs_at_async_engine_boundary(
    monkeypatch, drain, recompute_kv
):
    from nemo_rl.models.generation.trtllm import trtllm_backend as backend

    extension, module, model, model_loader, engine = _extension(backend)
    call_order = []

    def packed_consumer(*, iterator, group, src, post_unpack_func):
        call_order.append("broadcast")
        assert list(iterator) == list(extension.state_dict_info.items())
        assert group is extension.model_update_group
        assert src == 0
        post_unpack_func([("model.weight", torch.tensor([1.0]))])

    module.pre_reload_weights.side_effect = lambda: call_order.append("pre")
    module.process_weights_after_loading.side_effect = lambda: call_order.append(
        "process"
    )
    module.post_load_weights.side_effect = lambda: call_order.append("post")
    model_loader.begin_update_weights.side_effect = lambda: call_order.append("begin")
    model_loader.reload.side_effect = lambda *_, **__: call_order.append("reload")
    model_loader.finalize_update_weights.side_effect = lambda: call_order.append(
        "finalize"
    )
    monkeypatch.setattr(backend, "packed_broadcast_consumer", packed_consumer)
    monkeypatch.setattr(
        backend.torch.cuda, "synchronize", lambda: call_order.append("cuda_sync")
    )
    stream = MagicMock()
    stream.synchronize.side_effect = lambda: call_order.append("stream_sync")
    monkeypatch.setattr(backend.torch.cuda, "current_stream", lambda: stream)

    assert (
        extension.update_weights_from_collective(
            drain=drain,
            recompute_kv=recompute_kv,
        )
        is True
    )

    engine.control_action.assert_called_once_with(drain=drain)
    model_loader.reload.assert_called_once_with(
        model,
        {"model.weight": torch.tensor([1.0])},
        allow_partial_loading=True,
    )
    assert call_order == [
        "cuda_sync",
        "begin",
        "pre",
        "broadcast",
        "reload",
        "finalize",
        "process",
        "post",
        "stream_sync",
    ]
    model_loader.abort_update_weights.assert_not_called()
    engine.reset_prefix_cache.assert_called_once_with()


def test_collective_refit_requires_metadata():
    from nemo_rl.models.generation.trtllm import trtllm_backend as backend

    extension, *_ = _extension(backend)
    del extension.state_dict_info

    with pytest.raises(AssertionError, match="prepare_refit_info first"):
        extension.update_weights_from_collective()


def test_collective_refit_returns_false_when_reload_fails(monkeypatch):
    from nemo_rl.models.generation.trtllm import trtllm_backend as backend

    extension, _, _, model_loader, engine = _extension(backend)

    def packed_consumer(*, post_unpack_func, **_):
        post_unpack_func([("model.weight", torch.tensor([1.0]))])

    model_loader.reload.side_effect = RuntimeError("reload failed")
    monkeypatch.setattr(backend, "packed_broadcast_consumer", packed_consumer)
    monkeypatch.setattr(backend.torch.cuda, "synchronize", lambda: None)

    assert extension.update_weights_from_collective() is False
    model_loader.begin_update_weights.assert_called_once_with()
    model_loader.finalize_update_weights.assert_not_called()
    model_loader.abort_update_weights.assert_called_once_with()
    engine.reset_prefix_cache.assert_not_called()


def test_ipc_zmq_streams_chunk_and_reloads_with_aligned_offsets(monkeypatch):
    from nemo_rl.models.generation.trtllm import trtllm_backend as backend
    from nemo_rl.models.policy.utils import IPCProtocol

    extension, model_loader, engine = _ipc_extension(backend)

    buffer = torch.zeros(1024, dtype=torch.uint8)
    monkeypatch.setattr(backend, "rebuild_cuda_tensor_from_ipc", lambda h, d: buffer)
    monkeypatch.setattr(backend.torch.cuda, "current_stream", lambda: MagicMock())
    monkeypatch.setattr(backend.torch.cuda, "empty_cache", lambda: None)

    # One weight chunk carrying keys a,b (used_bytes == computed offset), then COMPLETE.
    extension.zmq_socket.recv_pyobj.side_effect = [
        ("ipc_handle", ["a", "b"], 1024),
        IPCProtocol.COMPLETE,
    ]

    assert extension.update_weights_via_ipc_zmq() is True

    model_loader.reload.assert_called_once()
    args, kwargs = model_loader.reload.call_args
    weights = args[1]
    assert set(weights) == {"a", "b"}
    assert weights["a"].shape == torch.Size([2])
    assert weights["b"].shape == torch.Size([3])
    assert kwargs["allow_partial_loading"] is True
    model_loader.begin_update_weights.assert_called_once_with()
    model_loader.finalize_update_weights.assert_called_once_with()
    model_loader.abort_update_weights.assert_not_called()
    engine.reset_prefix_cache.assert_called_once_with()
    # COMPLETE is ACKed after the final chunk.
    assert extension.zmq_socket.send.call_count == 2


def test_ipc_zmq_offset_mismatch_returns_false_without_reload(monkeypatch):
    from nemo_rl.models.generation.trtllm import trtllm_backend as backend
    from nemo_rl.models.policy.utils import IPCProtocol

    extension, model_loader, engine = _ipc_extension(backend)

    buffer = torch.zeros(1024, dtype=torch.uint8)
    monkeypatch.setattr(backend, "rebuild_cuda_tensor_from_ipc", lambda h, d: buffer)
    monkeypatch.setattr(backend.torch.cuda, "current_stream", lambda: MagicMock())

    # used_bytes (999) != computed offset (1024) -> assertion -> caught -> False.
    extension.zmq_socket.recv_pyobj.side_effect = [
        ("ipc_handle", ["a", "b"], 999),
        IPCProtocol.COMPLETE,
    ]

    assert extension.update_weights_via_ipc_zmq() is False
    model_loader.reload.assert_not_called()
    model_loader.begin_update_weights.assert_called_once_with()
    model_loader.finalize_update_weights.assert_not_called()
    model_loader.abort_update_weights.assert_called_once_with()
    engine.reset_prefix_cache.assert_not_called()


def test_zmq_address_and_cleanup_are_per_gpu_and_idempotent():
    from nemo_rl.models.generation.trtllm import trtllm_backend as backend

    extension = backend.NcclExtension.__new__(backend.NcclExtension)
    extension.report_device_id = MagicMock(return_value="GPU-abc")
    extension.zmq_socket = MagicMock()
    extension.zmq_context = MagicMock()
    socket = extension.zmq_socket
    context = extension.zmq_context

    assert extension.get_zmq_address() == "ipc:///tmp/GPU-abc.sock"
    extension.cleanup_zmq()
    extension.cleanup_zmq()

    socket.close.assert_called_once()
    context.destroy.assert_called_once()
