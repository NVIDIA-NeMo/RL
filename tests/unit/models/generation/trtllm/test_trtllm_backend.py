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


@pytest.mark.parametrize(
    ("drain", "recompute_kv", "should_reset"),
    [(True, False, False), (False, False, False), (False, True, True)],
)
def test_collective_refit_runs_at_async_engine_boundary(
    monkeypatch, drain, recompute_kv, should_reset
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
    model_loader.reload.side_effect = lambda *_, **__: call_order.append("reload")
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
        "pre",
        "broadcast",
        "reload",
        "process",
        "post",
        "stream_sync",
    ]
    assert engine.reset_prefix_cache.called is should_reset


def test_collective_refit_requires_metadata():
    from nemo_rl.models.generation.trtllm import trtllm_backend as backend

    extension, *_ = _extension(backend)
    del extension.state_dict_info

    with pytest.raises(AssertionError, match="prepare_refit_info first"):
        extension.update_weights_from_collective()


def test_collective_refit_returns_false_when_reload_fails(monkeypatch):
    from nemo_rl.models.generation.trtllm import trtllm_backend as backend

    extension, _, _, model_loader, _ = _extension(backend)

    def packed_consumer(*, post_unpack_func, **_):
        post_unpack_func([("model.weight", torch.tensor([1.0]))])

    model_loader.reload.side_effect = RuntimeError("reload failed")
    monkeypatch.setattr(backend, "packed_broadcast_consumer", packed_consumer)
    monkeypatch.setattr(backend.torch.cuda, "synchronize", lambda: None)

    assert extension.update_weights_from_collective() is False


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
