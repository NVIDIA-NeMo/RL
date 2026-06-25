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


def _install_fake_vllm_modules(monkeypatch, process_weights_after_loading):
    for module_name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.model_loader",
    ):
        monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))

    utils_module = types.ModuleType("vllm.model_executor.model_loader.utils")
    utils_module.process_weights_after_loading = process_weights_after_loading
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.utils",
        utils_module,
    )


def _load_vllm_backend(monkeypatch, process_weights_after_loading):
    _install_fake_vllm_modules(monkeypatch, process_weights_after_loading)

    module_name = f"_test_vllm_backend_{id(process_weights_after_loading)}"
    module_path = (
        Path(__file__).parents[4]
        / "nemo_rl/models/generation/vllm/vllm_backend.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _make_worker(backend):
    worker = backend.VllmInternalWorkerExtension.__new__(
        backend.VllmInternalWorkerExtension
    )
    state_info = object()
    worker.state_dict_info = {
        "model.weight": state_info,
    }
    worker.model_update_group = object()
    worker.model_runner = SimpleNamespace(model=object())
    worker.model_config = object()
    worker.device = object()
    return worker, state_info


def test_update_weights_from_collective_processes_weights_after_loading(monkeypatch):
    call_order = []
    process_calls = []

    def process_weights_after_loading(model, model_config, device):
        call_order.append("process")
        process_calls.append((model, model_config, device))

    backend = _load_vllm_backend(monkeypatch, process_weights_after_loading)
    worker, expected_state_info = _make_worker(backend)

    def load_weights(weights):
        call_order.append("load")
        assert weights == [("model.weight", "weight-value")]

    def packed_broadcast_consumer(iterator, group, src, post_unpack_func):
        call_order.append("broadcast")
        assert list(iterator) == [("model.weight", expected_state_info)]
        assert group is worker.model_update_group
        assert src == 0
        post_unpack_func([("model.weight", "weight-value")])

    worker._load_weights = load_weights
    worker._maybe_process_fp8_kv_cache = lambda: call_order.append("kv")
    monkeypatch.setattr(backend, "packed_broadcast_consumer", packed_broadcast_consumer)
    monkeypatch.setattr(backend.gc, "collect", lambda: call_order.append("gc"))
    monkeypatch.setattr(
        backend.torch.cuda,
        "empty_cache",
        lambda: call_order.append("empty_cache"),
    )

    assert worker.update_weights_from_collective() is True

    assert process_calls == [
        (worker.model_runner.model, worker.model_config, worker.device)
    ]
    assert call_order == ["broadcast", "load", "process", "kv", "gc", "empty_cache"]


def test_update_weights_from_collective_returns_false_on_post_load_failure(
    monkeypatch,
):
    call_order = []

    def process_weights_after_loading(_model, _model_config, _device):
        call_order.append("process")
        raise RuntimeError("post-load failed")

    backend = _load_vllm_backend(monkeypatch, process_weights_after_loading)
    worker, _ = _make_worker(backend)

    def packed_broadcast_consumer(_iterator, _group, _src, post_unpack_func):
        call_order.append("broadcast")
        post_unpack_func([("model.weight", "weight-value")])

    worker._load_weights = lambda _weights: call_order.append("load")
    worker._maybe_process_fp8_kv_cache = lambda: call_order.append("kv")
    monkeypatch.setattr(backend, "packed_broadcast_consumer", packed_broadcast_consumer)
    monkeypatch.setattr(backend.gc, "collect", lambda: call_order.append("gc"))
    monkeypatch.setattr(
        backend.torch.cuda,
        "empty_cache",
        lambda: call_order.append("empty_cache"),
    )

    assert worker.update_weights_from_collective() is False

    assert call_order == ["broadcast", "load", "process"]
