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

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo_rl.utils.refit_debug import RefitDebugStats


@pytest.fixture
def vllm_backend_module(monkeypatch, request):
    backend_name = "nemo_rl.models.generation.vllm.vllm_backend"
    original_backend = sys.modules.pop(backend_name, None)

    vllm_stub = types.ModuleType("vllm")
    zmq_stub = types.ModuleType("zmq")
    zmq_stub.REP = object()

    policy_utils_stub = types.ModuleType("nemo_rl.models.policy.utils")
    policy_utils_stub.IPCProtocol = SimpleNamespace(COMPLETE="complete", ACK="ack")
    policy_utils_stub.calculate_aligned_size = lambda size: size
    policy_utils_stub.rebuild_cuda_tensor_from_ipc = lambda *_args: None

    nsys_stub = types.ModuleType("nemo_rl.utils.nsys")
    nsys_stub.wrap_with_nvtx_name = lambda _name: lambda fn: fn

    packed_stub = types.ModuleType("nemo_rl.utils.packed_tensor")
    packed_stub.packed_broadcast_consumer = lambda **_kwargs: None

    fp8_stub = types.ModuleType(
        "nemo_rl.models.generation.vllm.quantization.fp8"
    )
    fp8_stub.is_fp8_model = lambda _config: False
    quantization_stub = types.ModuleType(
        "nemo_rl.models.generation.vllm.quantization"
    )
    quantization_stub.__path__ = []
    quantization_stub.fp8 = fp8_stub

    monkeypatch.setitem(sys.modules, "vllm", vllm_stub)
    monkeypatch.setitem(sys.modules, "zmq", zmq_stub)
    monkeypatch.setitem(
        sys.modules, "nemo_rl.models.policy.utils", policy_utils_stub
    )
    monkeypatch.setitem(sys.modules, "nemo_rl.utils.nsys", nsys_stub)
    monkeypatch.setitem(sys.modules, "nemo_rl.utils.packed_tensor", packed_stub)
    monkeypatch.setitem(
        sys.modules,
        "nemo_rl.models.generation.vllm.quantization",
        quantization_stub,
    )
    monkeypatch.setitem(
        sys.modules,
        "nemo_rl.models.generation.vllm.quantization.fp8",
        fp8_stub,
    )

    backend_path = (
        Path(__file__).parents[4]
        / "nemo_rl/models/generation/vllm/vllm_backend.py"
    )
    spec = importlib.util.spec_from_file_location(backend_name, backend_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[backend_name] = module
    spec.loader.exec_module(module)

    def restore_backend():
        sys.modules.pop(backend_name, None)
        if original_backend is not None:
            sys.modules[backend_name] = original_backend

    request.addfinalizer(restore_backend)
    return module


class _FakeVllmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.gate = nn.Linear(2, 2, bias=False)

    def load_weights(self, weights):
        parameters = dict(self.named_parameters())
        loaded = set()
        with torch.no_grad():
            for name, tensor in weights:
                parameters[name].copy_(tensor)
                loaded.add(name)
        return loaded


def test_load_weights_records_incoming_and_returns_loaded_names(
    monkeypatch, capsys, vllm_backend_module
):
    monkeypatch.setenv("NRL_REFIT_DEBUG", "1")
    name = "model.layers.0.mlp.gate.weight"
    source = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    model = _FakeVllmModel()
    extension = object.__new__(vllm_backend_module.VllmInternalWorkerExtension)
    extension.model_runner = SimpleNamespace(
        model=model,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(architectures=[])
        ),
    )
    extension._refit_debug_names = {"router_gate": name}
    extension._refit_debug_stats = RefitDebugStats()

    result = extension._load_weights([(name, source)])

    assert result == {name}
    assert extension._refit_debug_stats.loaded_count == 1
    assert torch.equal(dict(model.named_parameters())[name], source)
    captured = capsys.readouterr().out
    assert "[REFIT_DEBUG] phase=vllm_incoming" in captured
    assert "category=router_gate" in captured
    assert "dtype=torch.float32" in captured
