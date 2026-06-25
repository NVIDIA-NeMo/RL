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

import importlib
import os
import sys
import types

import pytest
import torch

from nemo_rl.modelopt.models.generation.vllm_modelopt_patch import (
    _canonicalize_nvfp4_weight_scale,
    _convert_nvfp4_linear_kernel_format,
    _modelopt_dense_apply,
    _modelopt_dense_process_weights,
    apply_modelopt_nvfp4_patches,
    modelopt_process_weights_after_loading,
    prepare_modelopt_for_weight_reload,
)
from nemo_rl.modelopt.utils import (
    build_vllm_modelopt_nvfp4_config,
    iter_quant_ignore_name_candidates,
    matches_quant_ignore_pattern,
    resolve_quant_cfg,
)


def _import_vllm_quant_backend(monkeypatch):
    """Import the NeMo-RL backend without requiring the vLLM C extension."""
    monkeypatch.delenv("VLLM_MODELOPT_REAL_QUANT", raising=False)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    _install_fake_modelopt_tensor_quantizer(monkeypatch)
    sys.modules.pop("nemo_rl.modelopt.models.generation.vllm_quant_backend", None)
    sys.modules.pop("nemo_rl.models.generation.vllm.vllm_backend", None)
    try:
        return importlib.import_module(
            "nemo_rl.modelopt.models.generation.vllm_quant_backend"
        )
    except ImportError as exc:
        pytest.skip(f"could not import vLLM quant backend: {exc}")


def _install_fake_modelopt_tensor_quantizer(monkeypatch):
    """Install the minimal ModelOpt module hierarchy needed by vLLM backend import."""
    module_names = [
        "modelopt",
        "modelopt.torch",
        "modelopt.torch.quantization",
        "modelopt.torch.quantization.nn",
        "modelopt.torch.quantization.nn.modules",
    ]
    modules = {}
    for module_name in module_names:
        module = types.ModuleType(module_name)
        module.__path__ = []
        modules[module_name] = module
        monkeypatch.setitem(sys.modules, module_name, module)

    tensor_quantizer_module = types.ModuleType(
        "modelopt.torch.quantization.nn.modules.tensor_quantizer"
    )

    class FakeTensorQuantizer(torch.nn.Module):
        pass

    tensor_quantizer_module.TensorQuantizer = FakeTensorQuantizer
    monkeypatch.setitem(
        sys.modules,
        "modelopt.torch.quantization.nn.modules.tensor_quantizer",
        tensor_quantizer_module,
    )
    modules["modelopt"].torch = modules["modelopt.torch"]
    modules["modelopt.torch"].quantization = modules["modelopt.torch.quantization"]
    modules["modelopt.torch.quantization"].nn = modules[
        "modelopt.torch.quantization.nn"
    ]
    modules["modelopt.torch.quantization.nn"].modules = modules[
        "modelopt.torch.quantization.nn.modules"
    ]
    modules[
        "modelopt.torch.quantization.nn.modules"
    ].tensor_quantizer = tensor_quantizer_module


def test_w4a16_real_quant_config_keeps_weight_only_default():
    cfg = build_vllm_modelopt_nvfp4_config()

    group = cfg["config_groups"]["group_0"]
    assert cfg["quant_method"] == "modelopt"
    assert cfg["quant_algo"] == "NVFP4"
    assert cfg["quant_mode"] == "w4a16_nvfp4"
    assert cfg["weight_only"] is True
    assert cfg["group_size"] == 16
    assert group["input_activations"] is None
    assert group["weights"] == {
        "dynamic": False,
        "num_bits": 4,
        "type": "float",
        "group_size": 16,
    }
    assert cfg["ignore"] == [
        "lm_head",
        "*output_layer*",
        "*mlp.gate",
        "*router*",
        "*block_sparse_moe.gate*",
        "*self_attention*",
        "*self_attn*",
    ]


def test_real_quant_config_allows_explicit_ignore_override():
    cfg = build_vllm_modelopt_nvfp4_config(ignore=["lm_head"])

    assert cfg["ignore"] == ["lm_head"]


def test_default_ignore_patterns_match_expected_layers():
    ignore_patterns = build_vllm_modelopt_nvfp4_config()["ignore"]

    assert matches_quant_ignore_pattern(
        "model.layers.0.self_attn.o_proj.weight", ignore_patterns
    )
    assert matches_quant_ignore_pattern(
        "layers.0.self_attn.o_proj.weight", ignore_patterns
    )
    assert matches_quant_ignore_pattern(
        "model.layers.0.mlp.gate.weight", ignore_patterns
    )
    assert matches_quant_ignore_pattern("model.layers.0.router.weight", ignore_patterns)
    assert matches_quant_ignore_pattern("lm_head.weight", ignore_patterns)
    assert matches_quant_ignore_pattern(
        "model.layers.0.mlp.gate.weight_scale", ignore_patterns
    )
    assert not matches_quant_ignore_pattern(
        "model.layers.0.mlp.experts.0.w1.weight", ignore_patterns
    )


def test_quant_ignore_name_candidates_include_model_prefix_and_base_names():
    assert list(
        iter_quant_ignore_name_candidates("layers.0.self_attn.q_proj.weight")
    ) == [
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj",
    ]
    assert list(iter_quant_ignore_name_candidates("model.lm_head.weight_scale")) == [
        "model.lm_head.weight_scale",
        "model.lm_head",
        "lm_head.weight_scale",
        "lm_head",
    ]


def test_configure_quant_engine_kwargs_for_fake_quant(monkeypatch):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)
    monkeypatch.delenv("VLLM_MODELOPT_REAL_QUANT", raising=False)

    llm_kwargs = {}
    worker_mod._configure_quant_engine_kwargs(
        {"quant_cfg": "examples/modelopt/quant_configs/nvfp4_w4a8_fp8.yaml"},
        llm_kwargs,
    )

    assert llm_kwargs["worker_cls"] == (
        "nemo_rl.modelopt.models.generation.vllm_quant_patch.FakeQuantWorker"
    )
    assert llm_kwargs["worker_extension_cls"] == (
        "nemo_rl.modelopt.models.generation.vllm_quant_backend.VllmQuantInternalWorkerExtension"
    )
    assert os.environ["VLLM_QUANT_CFG"] == (
        "examples/modelopt/quant_configs/nvfp4_w4a8_fp8.yaml"
    )
    assert "quantization" not in llm_kwargs


def test_configure_quant_engine_kwargs_for_real_quant(monkeypatch):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    patch_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_modelopt_patch"
    )
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)
    monkeypatch.delenv("VLLM_MODELOPT_REAL_QUANT", raising=False)
    patch_calls = []
    monkeypatch.setattr(
        patch_mod, "apply_modelopt_nvfp4_patches", lambda: patch_calls.append(True)
    )

    llm_kwargs = {}
    worker_mod._configure_quant_engine_kwargs(
        {
            "quant_cfg": "examples/modelopt/quant_configs/nvfp4_a16_mlp_only.yaml",
            "real_quant": True,
            "real_quant_ignore": ["lm_head"],
        },
        llm_kwargs,
    )

    assert patch_calls == [True]
    assert os.environ["VLLM_MODELOPT_REAL_QUANT"] == "1"
    assert "VLLM_QUANT_CFG" not in os.environ
    assert "worker_cls" not in llm_kwargs
    assert llm_kwargs["quantization"] == "modelopt"
    assert llm_kwargs["hf_overrides"]["quantization_config"] == (
        build_vllm_modelopt_nvfp4_config(ignore=["lm_head"])
    )


def test_configure_quant_engine_kwargs_preserves_hf_overrides(monkeypatch):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    patch_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_modelopt_patch"
    )
    monkeypatch.delenv("VLLM_MODELOPT_REAL_QUANT", raising=False)
    monkeypatch.setattr(patch_mod, "apply_modelopt_nvfp4_patches", lambda: None)

    llm_kwargs = {"hf_overrides": {"trust_remote_code": True}}
    worker_mod._configure_quant_engine_kwargs(
        {"quant_cfg": "NVFP4_DEFAULT_CFG", "real_quant": True},
        llm_kwargs,
    )

    assert llm_kwargs["hf_overrides"]["trust_remote_code"] is True
    assert (
        llm_kwargs["hf_overrides"]["quantization_config"]["quant_method"] == "modelopt"
    )


def test_configure_quant_engine_kwargs_for_fake_quant_without_quant_cfg(monkeypatch):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)
    monkeypatch.delenv("VLLM_MODELOPT_REAL_QUANT", raising=False)

    llm_kwargs = {}
    worker_mod._configure_quant_engine_kwargs({"quant_cfg": None}, llm_kwargs)

    assert "VLLM_QUANT_CFG" not in os.environ
    assert llm_kwargs["worker_cls"] == (
        "nemo_rl.modelopt.models.generation.vllm_quant_patch.FakeQuantWorker"
    )


def test_quant_generation_worker_create_engine_configures_quant(monkeypatch):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    worker_cls = worker_mod.VllmQuantGenerationWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker.cfg = {"quant_cfg": None}
    calls = []

    def fake_configure(cfg, llm_kwargs):
        calls.append(("configure", cfg, llm_kwargs))
        llm_kwargs["configured"] = True

    def fake_base_create_engine(self, llm_kwargs):
        calls.append(("base", dict(llm_kwargs)))

    monkeypatch.setattr(worker_mod, "_configure_quant_engine_kwargs", fake_configure)
    monkeypatch.setattr(
        worker_mod.VllmGenerationWorkerImpl,
        "_create_engine",
        fake_base_create_engine,
    )

    llm_kwargs = {}
    worker._create_engine(llm_kwargs)

    assert calls == [
        ("configure", worker.cfg, {"configured": True}),
        ("base", {"configured": True}),
    ]


def test_quant_generation_worker_collective_rpc_accessors():
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    worker_cls = worker_mod.VllmQuantGenerationWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    calls = []

    class FakeLLM:
        def collective_rpc(self, name, args):
            calls.append((name, args))
            return [{"name": name, "args": args}]

    worker.llm = FakeLLM()

    assert worker.get_quantizer_stats() == {
        "name": "get_quantizer_stats",
        "args": tuple(),
    }
    assert worker.get_weight_snapshot("weight") == {
        "name": "get_weight_snapshot",
        "args": ("weight",),
    }
    assert calls == [
        ("get_quantizer_stats", tuple()),
        ("get_weight_snapshot", ("weight",)),
    ]


@pytest.mark.asyncio
async def test_async_quant_generation_worker_collective_rpc_accessors():
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    worker_cls = (
        worker_mod.VllmQuantAsyncGenerationWorker.__ray_metadata__.modified_class
    )
    worker = object.__new__(worker_cls)
    calls = []

    class FakeLLM:
        async def collective_rpc(self, name, args):
            calls.append((name, args))
            return [{"name": name, "args": args}]

    worker.llm = FakeLLM()

    assert await worker.get_quantizer_stats() == {
        "name": "get_quantizer_stats",
        "args": tuple(),
    }
    assert await worker.get_weight_snapshot("weight") == {
        "name": "get_weight_snapshot",
        "args": ("weight",),
    }
    assert calls == [
        ("get_quantizer_stats", tuple()),
        ("get_weight_snapshot", ("weight",)),
    ]


def test_vllm_modelopt_backend_imports_without_gpt_oss_helper(monkeypatch):
    _import_vllm_quant_backend(monkeypatch)


def test_vllm_modelopt_backend_applies_real_quant_patch_on_import(monkeypatch):
    patch_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_modelopt_patch"
    )
    calls = []

    monkeypatch.setenv("VLLM_MODELOPT_REAL_QUANT", "1")
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    _install_fake_modelopt_tensor_quantizer(monkeypatch)
    monkeypatch.setattr(
        patch_mod,
        "apply_modelopt_nvfp4_patches",
        lambda: calls.append("patched"),
    )
    sys.modules.pop("nemo_rl.modelopt.models.generation.vllm_quant_backend", None)

    importlib.import_module("nemo_rl.modelopt.models.generation.vllm_quant_backend")

    assert calls == ["patched"]


def test_real_quant_load_weights_copies_ignored_float_weights(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(2, 2, bias=False)
            self.keep = torch.nn.Linear(2, 2, bias=False)

    model = TinyModel()
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(
                    quantization_config={"ignore": ["lm_head"]}
                )
            )
        ),
    )

    forwarded = []

    def fake_base_load_weights(self, weights):
        forwarded.extend(weights)
        return "loaded"

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        fake_base_load_weights,
    )

    ignored_weight = torch.full_like(model.lm_head.weight, 7.0)
    kept_weight = torch.full_like(model.keep.weight, 3.0)

    assert (
        extension._load_weights(
            [
                ("lm_head.weight", ignored_weight),
                ("lm_head.weight_scale", torch.ones(1)),
                ("keep.weight", kept_weight),
            ]
        )
        == "loaded"
    )

    torch.testing.assert_close(model.lm_head.weight, ignored_weight)
    assert [name for name, _ in forwarded] == ["keep.weight"]
    torch.testing.assert_close(forwarded[0][1], kept_weight)


def test_real_quant_load_weights_returns_when_only_ignored_weights(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(2, 2, bias=False)
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(
                    quantization_config={"ignore": ["lm_head"]}
                )
            )
        ),
    )
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )

    assert (
        extension._load_weights(
            [
                ("lm_head.weight", torch.full_like(model.lm_head.weight, 1.5)),
                ("lm_head.weight_scale", torch.ones(1)),
                ("lm_head.weight_scale_2", torch.ones(1)),
            ]
        )
        is None
    )
    torch.testing.assert_close(
        model.lm_head.weight,
        torch.full_like(model.lm_head.weight, 1.5),
    )


def test_real_quant_load_weights_forwards_ignored_shape_mismatch(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(2, 2, bias=False)
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(
                    quantization_config={"ignore": ["lm_head"]}
                )
            )
        ),
    )
    forwarded = []

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        lambda self, weights: forwarded.extend(weights) or "loaded",
    )

    mismatched = torch.ones(1, dtype=model.lm_head.weight.dtype)

    assert extension._load_weights([("lm_head.weight", mismatched)]) == "loaded"
    assert forwarded == [("lm_head.weight", mismatched)]


def test_fake_quant_load_weights_exposes_input_quantizer_buffers(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    child = torch.nn.Module()
    child.weight = torch.nn.Parameter(torch.ones(1))
    child.register_buffer("input_quantizer_amax", torch.tensor([1.0]))
    child.register_buffer("weight_quantizer_amax", torch.tensor([2.0]))
    model = torch.nn.Module()
    model.child = child
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(model=model)
    seen_names = []

    def fake_base_load_weights(self, weights):
        params = dict(child.named_parameters())
        seen_names.extend(params)
        params["input_quantizer_amax"].weight_loader(
            params["input_quantizer_amax"],
            torch.tensor([3.0]),
        )
        return "loaded"

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: False,
    )
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        fake_base_load_weights,
    )

    assert extension._load_weights([("unused", torch.ones(1))]) == "loaded"

    assert "weight" in seen_names
    assert "input_quantizer_amax" in seen_names
    assert "weight_quantizer_amax" not in seen_names
    assert not hasattr(child.input_quantizer_amax, "weight_loader")
    torch.testing.assert_close(child.input_quantizer_amax, torch.tensor([3.0]))


def test_real_quant_collective_reload_runs_modelopt_hooks(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    patch_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_modelopt_patch"
    )

    model = torch.nn.Linear(1, 1)
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(model=model)
    calls = []

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    monkeypatch.setattr(
        patch_mod,
        "prepare_modelopt_for_weight_reload",
        lambda model_arg, device: calls.append(("prepare", model_arg, device)),
    )
    monkeypatch.setattr(
        patch_mod,
        "modelopt_process_weights_after_loading",
        lambda model_arg: calls.append(("process", model_arg)),
    )
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "update_weights_from_collective",
        lambda self: True,
    )
    extension.device = torch.device("cpu")

    assert extension.update_weights_from_collective() is True
    assert calls == [
        ("prepare", model, torch.device("cpu")),
        ("process", model),
    ]


def test_real_quant_collective_reload_skips_processing_when_base_fails(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    patch_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_modelopt_patch"
    )

    model = torch.nn.Linear(1, 1)
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(model=model)
    extension.device = torch.device("cpu")
    calls = []

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    monkeypatch.setattr(
        patch_mod,
        "prepare_modelopt_for_weight_reload",
        lambda model_arg, device: calls.append(("prepare", model_arg, device)),
    )
    monkeypatch.setattr(
        patch_mod,
        "modelopt_process_weights_after_loading",
        lambda model_arg: calls.append(("process", model_arg)),
    )
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "update_weights_from_collective",
        lambda self: False,
    )

    assert extension.update_weights_from_collective() is False
    assert calls == [("prepare", model, torch.device("cpu"))]


def test_non_real_quant_collective_reload_delegates(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: False,
    )
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "update_weights_from_collective",
        lambda self: "delegated",
    )

    assert extension.update_weights_from_collective() == "delegated"


def test_real_quant_ipc_complete_processes_modelopt_and_acks(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    patch_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_modelopt_patch"
    )
    from nemo_rl.models.policy.utils import IPCProtocol

    class FakeSocket:
        def __init__(self):
            self.sent = []

        def recv_pyobj(self):
            return IPCProtocol.COMPLETE

        def send(self, payload):
            self.sent.append(payload)

    model = torch.nn.Linear(1, 1)
    socket = FakeSocket()
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(model=model)
    extension.device = torch.device("cpu")
    extension.zmq_socket = socket
    extension.maybe_init_zmq = lambda: None
    extension._maybe_process_fp8_kv_cache = lambda: None
    calls = []

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    monkeypatch.setattr(
        patch_mod,
        "prepare_modelopt_for_weight_reload",
        lambda model_arg, device: calls.append(("prepare", model_arg, device)),
    )
    monkeypatch.setattr(
        patch_mod,
        "modelopt_process_weights_after_loading",
        lambda model_arg: calls.append(("process", model_arg)),
    )
    monkeypatch.setattr(backend.torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(
        backend.torch.cuda, "empty_cache", lambda: calls.append("empty")
    )

    assert extension.update_weights_via_ipc_zmq() is True
    assert calls == [
        ("prepare", model, torch.device("cpu")),
        ("process", model),
        "sync",
        "empty",
    ]
    assert socket.sent == [IPCProtocol.ACK.value.encode()]


def test_real_quant_ipc_payload_loads_weights_and_handles_gpt_oss(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    patch_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_modelopt_patch"
    )
    from nemo_rl.models.policy.utils import IPCProtocol

    payload_weight = torch.tensor([1.0, 2.0], dtype=torch.float32)
    payload_buffer = payload_weight.view(torch.uint8)
    used_bytes = backend.calculate_aligned_size(payload_weight.nbytes)
    loaded = []
    calls = []

    class FakeSocket:
        def __init__(self):
            self.payloads = iter(
                [
                    ("ipc-handle", ["decoder.weight"], used_bytes),
                    IPCProtocol.COMPLETE,
                ]
            )
            self.sent = []

        def recv_pyobj(self):
            return next(self.payloads)

        def send(self, payload):
            self.sent.append(payload)

    model = torch.nn.Linear(1, 1)
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(architectures=["GptOssForCausalLM"])
        ),
    )
    extension.device = torch.device("cuda:0")
    extension.zmq_socket = FakeSocket()
    extension.state_dict_info = {"decoder.weight": ([2], torch.float32)}
    extension.maybe_init_zmq = lambda: None
    extension._maybe_process_fp8_kv_cache = lambda: calls.append("kv")
    extension._load_weights = lambda weights: loaded.extend(weights)

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    monkeypatch.setattr(
        patch_mod,
        "prepare_modelopt_for_weight_reload",
        lambda model_arg, device: calls.append(("prepare", model_arg, device)),
    )
    monkeypatch.setattr(
        patch_mod,
        "modelopt_process_weights_after_loading",
        lambda model_arg: calls.append(("process", model_arg)),
    )
    monkeypatch.setattr(
        backend,
        "rebuild_cuda_tensor_from_ipc",
        lambda ipc_handle, device_index: payload_buffer,
    )
    monkeypatch.setattr(backend.torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(
        backend.torch.cuda, "empty_cache", lambda: calls.append("empty")
    )
    monkeypatch.setattr(backend.gc, "collect", lambda: calls.append("gc"))

    assert extension.update_weights_via_ipc_zmq() is True

    assert extension.zmq_socket.sent == [
        IPCProtocol.ACK.value.encode(),
        IPCProtocol.ACK.value.encode(),
    ]
    assert loaded[0][0] == "decoder.weight"
    torch.testing.assert_close(loaded[0][1], payload_weight)
    assert calls == [
        ("prepare", model, torch.device("cuda:0")),
        "sync",
        ("process", model),
        "sync",
        "kv",
        "gc",
        "empty",
    ]


def test_non_real_quant_ipc_delegates(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: False,
    )
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "update_weights_via_ipc_zmq",
        lambda self: "delegated",
    )

    assert extension.update_weights_via_ipc_zmq() == "delegated"


def test_weight_snapshot_returns_cpu_clone_and_missing_name_raises(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    model = torch.nn.Linear(2, 1, bias=False)
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(model=model)

    snapshot = extension.get_weight_snapshot("weight")
    model.weight.data.add_(1.0)

    assert snapshot.device.type == "cpu"
    assert not torch.equal(snapshot, model.weight.detach().cpu())
    with pytest.raises(KeyError, match="missing"):
        extension.get_weight_snapshot("missing")


def test_get_quantizer_stats_counts_enabled_positive_amax(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    class FakeQuantizer(torch.nn.Module):
        def __init__(self, enabled, amax):
            super().__init__()
            self.is_enabled = enabled
            self.amax = amax

    model = torch.nn.Module()
    model.q_enabled_positive = FakeQuantizer(True, torch.tensor([1.0]))
    model.q_enabled_missing = FakeQuantizer(True, None)
    model.q_disabled_positive = FakeQuantizer(False, torch.tensor([2.0]))
    model.q_enabled_zero = FakeQuantizer(True, torch.tensor([0.0]))
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(model=model)
    monkeypatch.setattr(backend, "TensorQuantizer", FakeQuantizer)

    assert extension.get_quantizer_stats() == {
        "total": 4,
        "enabled": 3,
        "with_amax": 2,
        "positive_amax": 1,
    }


def test_resolve_quant_cfg_passes_relative_names_to_modelopt(monkeypatch):
    modelopt_recipe = pytest.importorskip("modelopt.recipe")
    captured = {}

    def fake_load_config(config_file):
        captured["config_file"] = config_file
        return {"quant_cfg": [{"name": "mock"}], "algorithm": "max"}

    monkeypatch.setattr(modelopt_recipe, "load_config", fake_load_config)

    assert resolve_quant_cfg("examples/modelopt/quant_configs/nvfp4_a16.yaml") == {
        "quant_cfg": [{"name": "mock"}],
        "algorithm": "max",
    }

    assert captured["config_file"] == "examples/modelopt/quant_configs/nvfp4_a16.yaml"


def test_resolve_quant_cfg_accepts_builtin_modelopt_constant(monkeypatch):
    mtq = pytest.importorskip("modelopt.torch.quantization")
    sentinel = {"quant_cfg": [{"name": "builtin"}], "algorithm": "max"}
    monkeypatch.setattr(mtq, "UNIT_TEST_CFG", sentinel, raising=False)

    assert resolve_quant_cfg("UNIT_TEST_CFG") is sentinel


def test_resolve_quant_cfg_extracts_nested_quantize_section(monkeypatch):
    modelopt_recipe = pytest.importorskip("modelopt.recipe")

    monkeypatch.setattr(
        modelopt_recipe,
        "load_config",
        lambda config_name: {
            "quantize": {
                "quant_cfg": [{"name": config_name}],
                "algorithm": "max",
            }
        },
    )

    assert resolve_quant_cfg("unit-test-recipe") == {
        "quant_cfg": [{"name": "unit-test-recipe"}],
        "algorithm": "max",
    }


def test_resolve_quant_cfg_rejects_unknown_config(monkeypatch):
    modelopt_recipe = pytest.importorskip("modelopt.recipe")

    def fake_load_config(config_name):
        raise FileNotFoundError(config_name)

    monkeypatch.setattr(modelopt_recipe, "load_config", fake_load_config)

    with pytest.raises(ValueError, match="Unknown quant_cfg"):
        resolve_quant_cfg("does-not-exist")


def test_resolve_quant_cfg_rejects_recipe_without_quant_cfg(monkeypatch):
    modelopt_recipe = pytest.importorskip("modelopt.recipe")
    monkeypatch.setattr(modelopt_recipe, "load_config", lambda config_name: {})

    with pytest.raises(ValueError, match="must contain a 'quant_cfg'"):
        resolve_quant_cfg("missing-quant-cfg")


def test_vllm_reload_canonicalizes_nvfp4_scales_before_kernel_conversion():
    layer = torch.nn.Module()
    layer.weight_scale = torch.nn.Parameter(
        torch.tensor([[1.0, -2.0], [-0.5, 4.0]]),
        requires_grad=False,
    )

    _canonicalize_nvfp4_weight_scale(layer)

    torch.testing.assert_close(
        layer.weight_scale,
        torch.tensor([[1.0, 2.0], [0.5, 4.0]]),
    )


def test_prepare_modelopt_for_weight_reload_restores_deleted_dense_params():
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.ones(2, 1), requires_grad=False)
    layer._nrl_modelopt_param_meta = {
        "weight": {
            "shape": (2, 2),
            "dtype": torch.float32,
            "device": "cpu",
            "param_class": torch.nn.Parameter,
        },
        "weight_scale_2": {
            "shape": (1,),
            "dtype": torch.float32,
            "device": "cpu",
            "param_class": torch.nn.Parameter,
        },
    }
    layer._nrl_modelopt_weight_loaders = {}
    model = torch.nn.Module()
    model.layer = layer

    prepare_modelopt_for_weight_reload(model, device=torch.device("cpu"))

    assert hasattr(layer, "weight_scale_2")
    assert tuple(layer.weight_scale_2.shape) == (1,)
    assert layer.weight_scale_2.dtype == torch.float32
    assert layer.weight_scale_2.device.type == "cpu"


def test_prepare_modelopt_for_weight_reload_restores_loader_class_when_shape_matches():
    class FakeModelWeightParameter(torch.nn.Parameter):
        def __new__(cls, data, **kwargs):
            return super().__new__(cls, data=data, requires_grad=False)

        def __init__(self, data, weight_loader, input_dim=1, output_dim=0):
            self.weight_loader = weight_loader
            self._input_dim = input_dim
            self._output_dim = output_dim

    def fake_merged_loader(param, loaded_weight, shard_id):
        shard_size = loaded_weight.shape[0]
        shard_offset = shard_id * shard_size
        param.data.narrow(0, shard_offset, shard_size).copy_(loaded_weight)

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4, 2), requires_grad=False)
    layer._nrl_modelopt_param_meta = {
        "weight": {
            "shape": (4, 2),
            "dtype": torch.float32,
            "device": "cpu",
            "param_class": FakeModelWeightParameter,
            "input_dim": 1,
            "output_dim": 0,
        },
    }
    layer._nrl_modelopt_weight_loaders = {"weight": fake_merged_loader}
    model = torch.nn.Module()
    model.layer = layer

    prepare_modelopt_for_weight_reload(model, device=torch.device("cpu"))

    assert isinstance(layer.weight, FakeModelWeightParameter)
    assert layer.weight.weight_loader is fake_merged_loader
    layer.weight.data.zero_()
    layer.weight.weight_loader(layer.weight, torch.ones(2, 2), 1)
    torch.testing.assert_close(layer.weight[:2], torch.zeros(2, 2))
    torch.testing.assert_close(layer.weight[2:], torch.ones(2, 2))


def test_prepare_modelopt_for_weight_reload_restores_plain_parameter_loader():
    def fake_loader(param, loaded_weight):
        param.data.copy_(loaded_weight)

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
    layer._nrl_modelopt_param_meta = {
        "weight": {
            "shape": (2, 2),
            "dtype": torch.float32,
            "device": "cpu",
            "param_class": torch.nn.Parameter,
        },
    }
    layer._nrl_modelopt_weight_loaders = {"weight": fake_loader}
    model = torch.nn.Module()
    model.layer = layer

    prepare_modelopt_for_weight_reload(model, device=torch.device("cpu"))

    assert isinstance(layer.weight, torch.nn.Parameter)
    assert layer.weight.weight_loader is fake_loader
    layer.weight.weight_loader(layer.weight, torch.ones(2, 2))
    torch.testing.assert_close(layer.weight, torch.ones(2, 2))


def test_modelopt_process_weights_after_loading_runs_dense_quant_method():
    calls = []

    class ModelOptNvFp4LinearMethod:
        def process_weights_after_loading(self, layer):
            calls.append(layer)

    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.quant_method = ModelOptNvFp4LinearMethod()

    modelopt_process_weights_after_loading(model)

    assert calls == [model.layer]


def test_apply_modelopt_nvfp4_patches_updates_vllm_method(monkeypatch):
    patch_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_modelopt_patch"
    )
    module_names = [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.quantization",
    ]
    for module_name in module_names:
        monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))
    modelopt_module = types.ModuleType(
        "vllm.model_executor.layers.quantization.modelopt"
    )

    class FakeModelOptNvFp4Config:
        @classmethod
        def _from_config(cls, **kwargs):
            return types.SimpleNamespace()

    def fake_linear_apply(self, layer, x, bias=None):
        pass

    class FakeModelOptNvFp4LinearMethod:
        apply = fake_linear_apply
        process_weights_after_loading = None

    modelopt_module.ModelOptNvFp4Config = FakeModelOptNvFp4Config
    modelopt_module.ModelOptNvFp4LinearMethod = FakeModelOptNvFp4LinearMethod
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.modelopt",
        modelopt_module,
    )
    monkeypatch.setattr(patch_mod, "_patched", False)

    apply_modelopt_nvfp4_patches()
    apply_modelopt_nvfp4_patches()

    cfg = FakeModelOptNvFp4Config._from_config(
        quant_method="NVFP4",
        kv_cache_quant_method=None,
        exclude_modules=[],
        original_config=build_vllm_modelopt_nvfp4_config(),
        group_size=16,
    )

    assert getattr(cfg, "_nrl_weight_only_w4a16") is True
    assert FakeModelOptNvFp4LinearMethod._nrl_original_apply is fake_linear_apply
    assert (
        FakeModelOptNvFp4LinearMethod.process_weights_after_loading
        is _modelopt_dense_process_weights
    )
    assert FakeModelOptNvFp4LinearMethod.apply is _modelopt_dense_apply
    assert patch_mod._patched is True


def test_convert_nvfp4_linear_kernel_format_uses_vllm_fallback(monkeypatch):
    calls = []
    module_names = [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.quantization",
        "vllm.model_executor.layers.quantization.utils",
    ]
    for module_name in module_names:
        monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))
    nvfp4_utils = types.ModuleType(
        "vllm.model_executor.layers.quantization.utils.nvfp4_utils"
    )

    def fake_convert(backend, layer):
        calls.append((backend, layer))

    nvfp4_utils.convert_to_nvfp4_linear_kernel_format = fake_convert
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.utils.nvfp4_utils",
        nvfp4_utils,
    )
    layer = torch.nn.Module()
    quant_method = types.SimpleNamespace(backend="backend")

    _convert_nvfp4_linear_kernel_format(quant_method, layer)

    assert calls == [("backend", layer)]


def test_modelopt_dense_process_uses_vllm_kernel_api():
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones(2, 1), requires_grad=False)
    layer.weight._input_dim = 1
    layer.weight._output_dim = 0
    layer.weight.weight_loader = lambda param, loaded_weight: None
    layer.weight_scale = torch.nn.Parameter(
        torch.tensor([[1.0, -2.0], [-0.5, 4.0]]),
        requires_grad=False,
    )
    layer.weight_scale_2 = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)

    calls = []

    class FakeKernel:
        def process_weights_after_loading(self, processed_layer):
            calls.append(processed_layer)
            torch.testing.assert_close(
                processed_layer.weight_scale,
                torch.tensor([[1.0, 2.0], [0.5, 4.0]]),
            )

    quant_method = types.SimpleNamespace(kernel=FakeKernel())

    _modelopt_dense_process_weights(quant_method, layer)

    assert calls == [layer]
    assert not hasattr(layer, "weight_scale_2")
    torch.testing.assert_close(layer.input_global_scale, torch.tensor(1.0))
    torch.testing.assert_close(layer.weight_global_scale, torch.tensor(2.0))
    torch.testing.assert_close(layer.alpha, torch.tensor(2.0))
    torch.testing.assert_close(layer.input_global_scale_inv, torch.tensor(1.0))
    assert set(layer._nrl_modelopt_param_meta) == {
        "weight",
        "weight_scale",
        "weight_scale_2",
    }
    assert layer._nrl_modelopt_param_meta["weight"]["input_dim"] == 1
    assert layer._nrl_modelopt_param_meta["weight"]["output_dim"] == 0
    assert "weight" in layer._nrl_modelopt_weight_loaders


def test_modelopt_dense_process_w4a16_uses_marlin_weight_only(monkeypatch):
    module_names = [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.quantization",
        "vllm.model_executor.layers.quantization.utils",
    ]
    for module_name in module_names:
        monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))

    marlin_utils = types.ModuleType(
        "vllm.model_executor.layers.quantization.utils.marlin_utils_fp4"
    )
    calls = []

    def fake_prepare(layer):
        calls.append(("prepare", layer))
        layer.workspace = torch.empty(1)

    def fake_apply(**kwargs):
        calls.append(("apply", kwargs))
        return "out"

    marlin_utils.prepare_fp4_layer_for_marlin = fake_prepare
    marlin_utils.apply_fp4_marlin_linear = fake_apply
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.utils.marlin_utils_fp4",
        marlin_utils,
    )

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones(2, 1), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.tensor([[1.0, -2.0], [-0.5, 4.0]]),
        requires_grad=False,
    )
    layer.weight_scale_2 = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
    layer.input_scale = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)
    layer.input_global_scale = torch.nn.Parameter(
        torch.tensor([4.0]),
        requires_grad=False,
    )
    layer.alpha = torch.nn.Parameter(torch.tensor([5.0]), requires_grad=False)
    layer.input_global_scale_inv = torch.nn.Parameter(
        torch.tensor([6.0]),
        requires_grad=False,
    )
    layer.output_size_per_partition = 2
    layer.input_size_per_partition = 2
    quant_method = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(_nrl_weight_only_w4a16=True)
    )

    _modelopt_dense_process_weights(quant_method, layer)
    result = _modelopt_dense_apply(quant_method, layer, torch.ones(1, 2))

    assert result == "out"
    assert calls[0] == ("prepare", layer)
    assert calls[1][0] == "apply"
    assert not hasattr(layer, "input_scale")
    assert not hasattr(layer, "input_global_scale")
    assert not hasattr(layer, "alpha")
    assert not hasattr(layer, "input_global_scale_inv")
    assert not hasattr(layer, "weight_scale_2")
    torch.testing.assert_close(layer.weight_global_scale, torch.tensor(2.0))
    torch.testing.assert_close(
        layer.weight_scale,
        torch.tensor([[1.0, 2.0], [0.5, 4.0]]),
    )
    assert calls[1][1]["weight_global_scale"] is layer.weight_global_scale
