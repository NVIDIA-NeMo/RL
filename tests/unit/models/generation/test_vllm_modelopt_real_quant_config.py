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
import weakref
from contextlib import nullcontext

import pytest
import torch

import nemo_rl.modelopt.models.generation.vllm_modelopt as vllm_modelopt
import nemo_rl.modelopt.utils as modelopt_utils
from nemo_rl.modelopt.models.generation.vllm_modelopt import (
    NEMO_MODELOPT_W4A4,
    NEMO_MODELOPT_W4A16,
    _pad_nvfp4_moe_for_marlin,
    quantization_method_for_mode,
    register_nemo_modelopt_nvfp4,
)
from nemo_rl.modelopt.utils import (
    build_vllm_modelopt_nvfp4_config,
    iter_quant_ignore_name_candidates,
    matches_quant_ignore_pattern,
    resolve_nvfp4_real_quant_mode,
    resolve_quant_cfg,
)


@pytest.fixture(autouse=True)
def _install_optional_modelopt_config_api(monkeypatch):
    """Provide ModelOpt's config APIs when the optional dependency is absent."""
    try:
        import modelopt.torch.export.convert_hf_config  # noqa: F401
        import modelopt.torch.quantization.config  # noqa: F401

        return
    except ImportError:
        pass

    module_names = (
        "modelopt",
        "modelopt.recipe",
        "modelopt.torch",
        "modelopt.torch.export",
        "modelopt.torch.quantization",
    )
    for module_name in module_names:
        module = types.ModuleType(module_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)

    def missing_recipe(config_name):
        raise FileNotFoundError(config_name)

    sys.modules["modelopt.recipe"].load_config = missing_recipe

    convert_module = types.ModuleType("modelopt.torch.export.convert_hf_config")

    def convert_hf_quant_config_format(config):
        quantization = config["quantization"]
        algo = quantization["quant_algo"]
        group = {
            "weights": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": quantization["group_size"],
            },
            "targets": ["Linear"],
        }
        if algo == "NVFP4":
            group["input_activations"] = dict(group["weights"])
        return {
            "config_groups": {"group_0": group},
            "ignore": quantization["exclude_modules"],
            "quant_algo": algo,
            "producer": config["producer"],
            "quant_method": "modelopt",
        }

    convert_module.convert_hf_quant_config_format = convert_hf_quant_config_format
    monkeypatch.setitem(
        sys.modules,
        "modelopt.torch.export.convert_hf_config",
        convert_module,
    )

    config_module = types.ModuleType("modelopt.torch.quantization.config")

    class QuantizerCfgEntry:
        def __init__(self, entry):
            self.entry = {"enable": True, **entry}

        def model_dump(self, **kwargs):
            del kwargs
            return {
                key: value for key, value in self.entry.items() if value is not None
            }

    class QuantizeConfig:
        def __init__(self, quant_cfg, **kwargs):
            del kwargs
            self.quant_cfg = [QuantizerCfgEntry(entry) for entry in quant_cfg]

    config_module.QuantizeConfig = QuantizeConfig
    monkeypatch.setitem(
        sys.modules,
        "modelopt.torch.quantization.config",
        config_module,
    )


def _import_vllm_quant_backend(monkeypatch):
    """Import the NeMo-RL backend without requiring the vLLM C extension."""
    monkeypatch.delenv("VLLM_MODELOPT_REAL_QUANT", raising=False)
    vllm_module = types.ModuleType("vllm")
    vllm_module.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    _install_fake_vllm_reload(monkeypatch)
    _install_fake_modelopt_tensor_quantizer(monkeypatch)
    sys.modules.pop("nemo_rl.modelopt.models.generation.vllm_quant_backend", None)
    sys.modules.pop("nemo_rl.models.generation.vllm.vllm_backend", None)
    try:
        return importlib.import_module(
            "nemo_rl.modelopt.models.generation.vllm_quant_backend"
        )
    except ImportError as exc:
        pytest.skip(f"could not import vLLM quant backend: {exc}")


def _base_vllm_backend():
    return sys.modules["nemo_rl.models.generation.vllm.vllm_backend"]


def _install_fake_vllm_reload(monkeypatch):
    """Install the public vLLM layerwise-reload API used by real-quant refits."""
    module_names = (
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.quantization",
        "vllm.model_executor.model_loader",
    )
    for module_name in module_names:
        module = types.ModuleType(module_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)

    reload_module = types.ModuleType("vllm.model_executor.model_loader.reload")
    reload_module.__path__ = []
    reload_module.initialize_layerwise_reload = lambda model: None
    reload_module.finalize_layerwise_reload = lambda model, model_config: None
    layerwise_module = types.ModuleType(
        "vllm.model_executor.model_loader.reload.layerwise"
    )
    layerwise_module.get_layerwise_info = lambda module: types.SimpleNamespace(
        loaded_weights=[],
        load_numel=0,
        load_numel_total=None,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.reload",
        reload_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.reload.layerwise",
        layerwise_module,
    )
    modelopt_module = types.ModuleType(
        "vllm.model_executor.layers.quantization.modelopt"
    )
    modelopt_module.ModelOptNvFp4FusedMoE = type("ModelOptNvFp4FusedMoE", (), {})
    modelopt_module.ModelOptNvFp4LinearMethod = type(
        "ModelOptNvFp4LinearMethod", (), {}
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.modelopt",
        modelopt_module,
    )
    attention_module = types.ModuleType("vllm.model_executor.layers.attention")
    attention_module.Attention = type("Attention", (torch.nn.Module,), {})
    attention_module.MLAAttention = type("MLAAttention", (torch.nn.Module,), {})
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.attention",
        attention_module,
    )
    kv_cache_module = types.ModuleType(
        "vllm.model_executor.layers.quantization.kv_cache"
    )
    kv_cache_module.BaseKVCacheMethod = type("BaseKVCacheMethod", (), {})
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.kv_cache",
        kv_cache_module,
    )
    return reload_module


def _install_fake_registered_vllm_modelopt(monkeypatch):
    """Install the public vLLM registration surface used by vllm_modelopt."""
    module_names = (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.kernels",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.fused_moe.oracle",
        "vllm.model_executor.layers.quantization",
        "vllm.model_executor.layers.quantization.utils",
    )
    for module_name in module_names:
        module = types.ModuleType(module_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)

    registry = {}
    events = []

    quantization_module = sys.modules["vllm.model_executor.layers.quantization"]

    def register_quantization_config(name):
        def register(config_cls):
            registry[name] = config_cls
            return config_cls

        return register

    quantization_module.register_quantization_config = register_quantization_config

    weight_loader_v2_supported = []
    linear_module = types.ModuleType("vllm.model_executor.layers.linear")

    def register_weight_loader_v2_supported_method(method_cls: type) -> type:
        weight_loader_v2_supported.append(method_cls.__name__)
        return method_cls

    linear_module.register_weight_loader_v2_supported_method = (
        register_weight_loader_v2_supported_method
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.linear",
        linear_module,
    )

    class FakeModelOptNvFp4LinearMethod:
        def __init__(self, quant_config):
            self.quant_config = quant_config

        def create_weights(self, layer, *args, **kwargs):
            del args, kwargs
            if not hasattr(layer, "input_scale"):
                layer.input_scale = torch.nn.Parameter(torch.ones(1))

    class FakeModelOptNvFp4FusedMoE:
        def __init__(self, quant_config, moe_config):
            self.quant_config = quant_config
            self.moe = moe_config
            self.moe_kernel = None
            self.moe_quant_config = None

        def create_weights(self, layer, *args, **kwargs):
            events.append(("native_create_weights", layer, args, kwargs))
            num_experts = args[0] if args else 1
            if not hasattr(layer, "w13_input_scale"):
                layer.w13_input_scale = torch.nn.Parameter(torch.zeros(num_experts, 2))
            if not hasattr(layer, "w2_input_scale"):
                layer.w2_input_scale = torch.nn.Parameter(torch.zeros(num_experts))

        def get_fused_moe_quant_config(self, layer):
            del layer
            return object()

        def process_weights_after_loading(self, layer):
            events.append(
                (
                    "native_process_moe",
                    getattr(
                        getattr(layer, "moe_config", None),
                        "intermediate_size_per_partition",
                        None,
                    ),
                )
            )
            self.moe_kernel = types.SimpleNamespace(source="native", layer=layer)
            w13_input_scale = getattr(layer, "w13_input_scale", None)
            w2_input_scale = getattr(layer, "w2_input_scale", None)
            if isinstance(w13_input_scale, torch.Tensor) and isinstance(
                w2_input_scale, torch.Tensor
            ):
                self.moe_quant_config = types.SimpleNamespace(
                    source="native",
                    a1_gscale=1.0 / w13_input_scale,
                    a2_gscale=1.0 / w2_input_scale,
                )
            else:
                self.moe_quant_config = types.SimpleNamespace(source="native")

    class FakeModelOptNvFp4Config:
        LinearMethodCls = FakeModelOptNvFp4LinearMethod
        FusedMoEMethodCls = FakeModelOptNvFp4FusedMoE

        def __init__(self, group_size=16):
            self.group_size = group_size

        @classmethod
        def from_config(cls, config):
            target = config.get("quantization", config)
            instance = cls(group_size=target.get("group_size", 16))
            instance.parsed_config = config
            return instance

        @classmethod
        def _extract_modelopt_quant_algo(cls, config):
            del cls
            target = config.get("quantization", config)
            return str(target.get("quant_algo", "")).upper()

    modelopt_module = types.ModuleType(
        "vllm.model_executor.layers.quantization.modelopt"
    )
    modelopt_module.ModelOptNvFp4Config = FakeModelOptNvFp4Config
    modelopt_module.ModelOptNvFp4LinearMethod = FakeModelOptNvFp4LinearMethod
    modelopt_module.ModelOptNvFp4FusedMoE = FakeModelOptNvFp4FusedMoE
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.modelopt",
        modelopt_module,
    )

    class FakeFusedMoEMethodBase:
        def __init__(self, moe_config):
            self.moe = moe_config
            self.moe_kernel = None
            self.moe_quant_config = None

    fused_method_module = types.ModuleType(
        "vllm.model_executor.layers.fused_moe.fused_moe_method_base"
    )
    fused_method_module.FusedMoEMethodBase = FakeFusedMoEMethodBase
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.fused_moe_method_base",
        fused_method_module,
    )

    class FakeNvFp4LinearLayerConfig:
        pass

    class FakeMarlinNvFp4LinearKernel:
        def __init__(self, config):
            self.config = config

        def process_weights_after_loading(self, layer):
            events.append(("process_marlin_kernel", layer))

        def apply_weights(self, **kwargs):
            events.append(("apply_marlin_kernel", kwargs))
            return "output"

    linear_kernel_module = types.ModuleType("vllm.model_executor.kernels.linear")
    linear_kernel_module.MarlinNvFp4LinearKernel = FakeMarlinNvFp4LinearKernel
    linear_kernel_module.NvFp4LinearLayerConfig = FakeNvFp4LinearLayerConfig
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.kernels.linear",
        linear_kernel_module,
    )

    class FakeMarlinExperts:
        pass

    oracle_module = types.ModuleType(
        "vllm.model_executor.layers.fused_moe.oracle.nvfp4"
    )
    oracle_module.NvFp4MoeBackend = types.SimpleNamespace(MARLIN="marlin")

    def convert_to_nvfp4_moe_kernel_format(**kwargs):
        events.append(
            (
                "convert_moe",
                kwargs["layer"].moe_config.intermediate_size_per_partition,
            )
        )
        return tuple(
            kwargs[name]
            for name in (
                "w13",
                "w13_scale",
                "w13_scale_2",
                "a13_scale",
                "w2",
                "w2_scale",
                "w2_scale_2",
                "a2_scale",
            )
        )

    oracle_module.convert_to_nvfp4_moe_kernel_format = (
        convert_to_nvfp4_moe_kernel_format
    )
    oracle_module.is_global_sf_supported_for_nvfp4_backend = lambda backend: False
    oracle_module.select_nvfp4_moe_backend = lambda **kwargs: (
        oracle_module.NvFp4MoeBackend.MARLIN,
        FakeMarlinExperts,
    )
    oracle_module.make_nvfp4_moe_kernel = lambda **kwargs: events.append(
        ("make_moe_kernel", kwargs)
    ) or types.SimpleNamespace(
        fused_experts=types.SimpleNamespace(
            process_weights_after_loading=lambda layer: events.append(
                ("process_moe", layer)
            )
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.oracle.nvfp4",
        oracle_module,
    )

    quant_utils_module = types.ModuleType(
        "vllm.model_executor.layers.quantization.utils.quant_utils"
    )
    quant_utils_module.kNvfp4Static = object()
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        quant_utils_module,
    )

    utils_module = types.ModuleType("vllm.model_executor.utils")

    def replace_parameter(layer, name, value):
        if name in layer._parameters:
            del layer._parameters[name]
        setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))

    utils_module.replace_parameter = replace_parameter
    monkeypatch.setitem(sys.modules, "vllm.model_executor.utils", utils_module)
    return types.SimpleNamespace(
        registry=registry,
        events=events,
        weight_loader_v2_supported=weight_loader_v2_supported,
    )


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


def _make_real_quant_extension(backend, model, ignore):
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.device = torch.device("cpu")
    extension._nrl_w13_num_shards_by_prefix = {}
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(enable_expert_parallel=False),
            model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(quantization_config={"ignore": ignore})
            ),
        ),
    )
    return extension


def _patch_real_quant_load(monkeypatch, backend, forwarded=None):
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    if forwarded is not None:
        monkeypatch.setattr(
            backend.VllmInternalWorkerExtension,
            "_load_weights",
            lambda self, weights: forwarded.extend(weights) or "loaded",
        )


def _mark_as_modelopt_layer(model):
    modelopt_module = sys.modules["vllm.model_executor.layers.quantization.modelopt"]
    model.quant_method = modelopt_module.ModelOptNvFp4LinearMethod()
    return model


def test_base_ipc_data_ack_fence_synchronizes_current_stream_once(monkeypatch):
    _import_vllm_quant_backend(monkeypatch)
    backend = _base_vllm_backend()
    extension = object.__new__(backend.VllmInternalWorkerExtension)
    calls = []
    stream = types.SimpleNamespace(synchronize=lambda: calls.append("sync"))
    monkeypatch.setattr(
        backend.torch.cuda,
        "current_stream",
        lambda: calls.append("current_stream") or stream,
    )

    extension._synchronize_before_ipc_data_ack()

    assert calls == ["current_stream", "sync"]


def test_w4a16_real_quant_config_is_weight_only():
    cfg = build_vllm_modelopt_nvfp4_config(mode="w4a16")

    group = cfg["config_groups"]["group_0"]
    assert cfg["quant_method"] == "modelopt"
    assert cfg["quant_algo"] == "W4A16_NVFP4"
    assert "input_activations" not in group
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


def test_w4a4_real_quant_config_has_static_input_activations():
    cfg = build_vllm_modelopt_nvfp4_config(mode="w4a4")

    group = cfg["config_groups"]["group_0"]
    assert cfg["quant_method"] == "modelopt"
    assert cfg["quant_algo"] == "NVFP4"
    assert group["input_activations"] == {
        "dynamic": False,
        "num_bits": 4,
        "type": "float",
        "group_size": 16,
    }


def test_real_quant_config_rejects_unsupported_mode():
    with pytest.raises(ValueError, match="expected 'w4a4' or 'w4a16'"):
        build_vllm_modelopt_nvfp4_config(mode="w4a8")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mode", "method"),
    [("w4a4", NEMO_MODELOPT_W4A4), ("w4a16", NEMO_MODELOPT_W4A16)],
)
def test_quantization_method_for_mode_uses_registered_names(mode, method):
    assert quantization_method_for_mode(mode) == method


def test_quantization_method_for_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported ModelOpt NVFP4 rollout mode"):
        quantization_method_for_mode("w4a8")


def test_real_quant_config_allows_explicit_ignore_override():
    ignore = ["lm_head", "*.mixer.in_proj*"]
    cfg = build_vllm_modelopt_nvfp4_config(mode="w4a16", ignore=ignore)

    assert cfg["ignore"] == ignore
    assert matches_quant_ignore_pattern(
        "model.layers.0.mixer.in_proj.weight",
        cfg["ignore"],
    )


def test_default_ignore_patterns_match_expected_layers():
    ignore_patterns = build_vllm_modelopt_nvfp4_config(mode="w4a16")["ignore"]

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
    assert matches_quant_ignore_pattern(
        "model.layers.0.mlp.gate.input_scale", ignore_patterns
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
    assert list(iter_quant_ignore_name_candidates("model.lm_head.input_scale")) == [
        "model.lm_head.input_scale",
        "model.lm_head",
        "lm_head.input_scale",
        "lm_head",
    ]


def test_configure_quant_engine_kwargs_for_fake_quant(monkeypatch, tmp_path):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)
    monkeypatch.delenv("VLLM_MODELOPT_REAL_QUANT", raising=False)

    quant_cfg = "quant.yaml"
    (tmp_path / quant_cfg).touch()
    monkeypatch.chdir(tmp_path)

    llm_kwargs = {}
    worker_mod._configure_quant_engine_kwargs(
        {"quant_cfg": quant_cfg},
        llm_kwargs,
    )

    assert llm_kwargs["worker_cls"] == (
        "nemo_rl.modelopt.models.generation.vllm_quant_patch.FakeQuantWorker"
    )
    assert llm_kwargs["worker_extension_cls"] == (
        "nemo_rl.modelopt.models.generation.vllm_quant_backend.VllmQuantInternalWorkerExtension"
    )
    assert os.environ["VLLM_QUANT_CFG"] == os.path.abspath(quant_cfg)
    assert "quantization" not in llm_kwargs


def test_quant_worker_forwards_snapshot_pythonpath_to_inner_vllm_workers():
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )

    assert "PYTHONPATH" in worker_mod._EXTRA_ENV_VARS


def test_configure_quant_engine_kwargs_for_real_quant(monkeypatch):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)
    monkeypatch.delenv("VLLM_MODELOPT_REAL_QUANT", raising=False)
    registration_calls = []
    monkeypatch.setattr(
        vllm_modelopt,
        "register_nemo_modelopt_nvfp4",
        lambda: registration_calls.append(True),
    )
    monkeypatch.setattr(
        modelopt_utils, "resolve_nvfp4_real_quant_mode", lambda _: "w4a16"
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

    assert registration_calls == [True]
    assert os.environ["VLLM_MODELOPT_REAL_QUANT"] == "1"
    assert "VLLM_QUANT_CFG" not in os.environ
    assert "worker_cls" not in llm_kwargs
    assert llm_kwargs["quantization"] == NEMO_MODELOPT_W4A16
    assert llm_kwargs["hf_overrides"]["quantization_config"] == (
        build_vllm_modelopt_nvfp4_config(mode="w4a16", ignore=["lm_head"])
    )


@pytest.mark.parametrize("mode", ["w4a4", "w4a16"])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_configure_real_quant_preserves_kv_cache_dtype(
    monkeypatch,
    mode,
    kv_cache_dtype,
):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.setattr(vllm_modelopt, "register_nemo_modelopt_nvfp4", lambda: None)
    monkeypatch.setattr(
        modelopt_utils,
        "resolve_nvfp4_real_quant_mode",
        lambda _: mode,
    )

    llm_kwargs = {"kv_cache_dtype": kv_cache_dtype}
    worker_mod._configure_quant_engine_kwargs(
        {"quant_cfg": "NVFP4_EXPERTS_ONLY_CFG", "real_quant": True},
        llm_kwargs,
    )

    assert llm_kwargs["kv_cache_dtype"] == kv_cache_dtype
    assert llm_kwargs["quantization"] == quantization_method_for_mode(mode)
    assert "kv_cache" not in llm_kwargs["hf_overrides"]["quantization_config"]


def test_configure_quant_engine_kwargs_preserves_hf_overrides(monkeypatch):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_MODELOPT_REAL_QUANT", raising=False)
    monkeypatch.setattr(vllm_modelopt, "register_nemo_modelopt_nvfp4", lambda: None)
    monkeypatch.setattr(
        modelopt_utils, "resolve_nvfp4_real_quant_mode", lambda _: "w4a16"
    )

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


def test_real_quant_backend_uses_modelopt_refit_timeout(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    events = []

    class FakeSocket:
        def setsockopt(self, option, value):
            events.append(("setsockopt", option, value))

        def connect(self, address):
            events.append(("connect", address))

    class FakeContext:
        def socket(self, socket_type):
            events.append(("socket", socket_type))
            return FakeSocket()

    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.get_zmq_address = lambda: "ipc:///tmp/modelopt-test.sock"
    monkeypatch.setattr(backend.zmq, "Context", FakeContext)
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda _self: True,
    )

    extension.maybe_init_zmq()

    assert events[0] == ("socket", backend.zmq.REP)
    assert ("setsockopt", backend.zmq.LINGER, 0) in events
    assert ("connect", "ipc:///tmp/modelopt-test.sock") in events
    assert events[-2:] == [
        (
            "setsockopt",
            backend.zmq.SNDTIMEO,
            modelopt_utils.MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS,
        ),
        (
            "setsockopt",
            backend.zmq.RCVTIMEO,
            modelopt_utils.MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS,
        ),
    ]


def test_vllm_modelopt_backend_registers_real_quant_configs_on_import(monkeypatch):
    calls = []

    monkeypatch.setenv("VLLM_MODELOPT_REAL_QUANT", "1")
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    _install_fake_modelopt_tensor_quantizer(monkeypatch)
    monkeypatch.setattr(
        vllm_modelopt,
        "register_nemo_modelopt_nvfp4",
        lambda: calls.append("registered"),
    )
    sys.modules.pop("nemo_rl.modelopt.models.generation.vllm_quant_backend", None)

    importlib.import_module("nemo_rl.modelopt.models.generation.vllm_quant_backend")

    assert calls == ["registered"]


def test_modelopt_moe_manifest_requires_complete_w4a4_family(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    prefix = "model.layers.0.mixer"
    state_dict_info = {
        f"{prefix}.experts.w13_weight": ((2, 4, 3), torch.uint8),
        f"{prefix}.experts.w13_weight_scale": ((2, 4, 1), torch.uint8),
        f"{prefix}.experts.w13_weight_scale_2": ((2, 2), torch.float32),
        f"{prefix}.experts.w13_input_scale": ((2, 2), torch.float32),
        f"{prefix}.experts.w2_weight": ((2, 3, 4), torch.uint8),
        f"{prefix}.experts.w2_weight_scale": ((2, 1, 4), torch.uint8),
        f"{prefix}.experts.w2_weight_scale_2": ((2,), torch.float32),
        f"{prefix}.experts.w2_input_scale": ((2,), torch.float32),
    }

    assert backend._w13_num_shards_from_state_dict_info(
        state_dict_info, require_input_scales=True
    ) == {prefix: 2}

    legacy_state_dict_info = dict(state_dict_info)
    legacy_state_dict_info[f"{prefix}.experts.w13_weight_scale_2"] = (
        (2,),
        torch.float32,
    )
    legacy_state_dict_info[f"{prefix}.experts.w13_input_scale"] = (
        (2,),
        torch.float32,
    )
    assert backend._w13_num_shards_from_state_dict_info(
        legacy_state_dict_info, require_input_scales=True
    ) == {prefix: 1}

    mismatched_state_dict_info = dict(state_dict_info)
    mismatched_state_dict_info[f"{prefix}.experts.w13_input_scale"] = (
        (2, 1),
        torch.float32,
    )
    with pytest.raises(RuntimeError, match="input/global scale layouts disagree"):
        backend._w13_num_shards_from_state_dict_info(
            mismatched_state_dict_info, require_input_scales=True
        )

    del state_dict_info[f"{prefix}.experts.w2_input_scale"]
    with pytest.raises(RuntimeError, match="missing.*w2_input_scale"):
        backend._w13_num_shards_from_state_dict_info(
            state_dict_info, require_input_scales=True
        )


def test_real_quant_load_weights_batches_full_experts_and_expands_global_scales(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)

    class ModelOptNvFp4FusedMoE:
        quant_config = types.SimpleNamespace(get_name=lambda: NEMO_MODELOPT_W4A16)

    def make_model(expert_map):
        model = torch.nn.Module()
        model.moe = torch.nn.Module()
        model.moe.quant_method = ModelOptNvFp4FusedMoE()
        model.moe._expert_map = expert_map
        model.moe.local_num_experts = 2 if expert_map is None else 1
        model.moe.global_num_experts = 2
        # ModelOpt assigns the same quant config to attention's FP8 KV method;
        # this must not be mistaken for expert parallelism.
        model.attention = torch.nn.Module()
        model.attention.quant_method = ModelOptNvFp4FusedMoE()
        return model

    prefix = "model.layers.0.mlp"
    w13_weight = torch.arange(24).reshape(2, 4, 3)
    w13_scale_2 = torch.tensor([[1.0], [2.0]])
    state_dict_info = {
        f"{prefix}.experts.w13_weight": ((2, 4, 3), torch.uint8),
        f"{prefix}.experts.w13_weight_scale": ((2, 4, 1), torch.uint8),
        f"{prefix}.experts.w13_weight_scale_2": ((2, 1), torch.float32),
        f"{prefix}.experts.w2_weight": ((2, 3, 2), torch.uint8),
        f"{prefix}.experts.w2_weight_scale": ((2, 3, 1), torch.uint8),
        f"{prefix}.experts.w2_weight_scale_2": ((2,), torch.float32),
    }

    batched_forwarded = []
    extension = _make_real_quant_extension(backend, make_model(None), [])
    extension.prepare_refit_info(state_dict_info)
    extension._nrl_w13_num_shards_by_prefix = {prefix: 1}
    _patch_real_quant_load(monkeypatch, backend, batched_forwarded)
    assert (
        extension._load_weights(
            [
                (f"{prefix}.experts.w13_weight", w13_weight),
                (f"{prefix}.experts.w13_weight_scale_2", w13_scale_2),
            ]
        )
        == "loaded"
    )
    assert [name for name, _ in batched_forwarded] == [
        f"{prefix}.experts.0.up_proj.weight",
        f"{prefix}.experts.0.up_proj.weight_scale_2",
        f"{prefix}.experts.1.up_proj.weight_scale_2",
    ]
    assert batched_forwarded[0][1] is w13_weight
    torch.testing.assert_close(batched_forwarded[1][1], w13_scale_2[0, 0])
    torch.testing.assert_close(batched_forwarded[2][1], w13_scale_2[1, 0])

    extension = _make_real_quant_extension(
        backend,
        make_model(torch.tensor([0, -1])),
        [],
    )
    extension.model_runner.vllm_config.parallel_config.enable_expert_parallel = True
    with pytest.raises(RuntimeError, match="all experts local"):
        extension.prepare_refit_info(state_dict_info)


def test_real_quant_load_weights_forwards_ignored_float_weights(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(2, 2, bias=False)
            self.keep = torch.nn.Linear(2, 2, bias=False)

    model = TinyModel()
    forwarded = []
    extension = _make_real_quant_extension(backend, model, ["lm_head"])
    _patch_real_quant_load(monkeypatch, backend, forwarded)

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

    assert [name for name, _ in forwarded] == ["lm_head.weight", "keep.weight"]
    torch.testing.assert_close(forwarded[0][1], ignored_weight)
    torch.testing.assert_close(forwarded[1][1], kept_weight)


def test_real_quant_load_weights_returns_when_only_ignored_scales(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(2, 2, bias=False)
    extension = _make_real_quant_extension(backend, model, ["lm_head"])
    _patch_real_quant_load(monkeypatch, backend)

    assert (
        extension._load_weights(
            [
                ("lm_head.weight_scale", torch.ones(1)),
                ("lm_head.weight_scale_2", torch.ones(1)),
            ]
        )
        is None
    )


def test_real_quant_load_weights_forwards_ignored_weights_to_vllm_loader(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(2, 2, bias=False)
    forwarded = []
    extension = _make_real_quant_extension(backend, model, ["lm_head"])
    _patch_real_quant_load(monkeypatch, backend, forwarded)

    mismatched = torch.ones(1, dtype=model.lm_head.weight.dtype)

    assert extension._load_weights([("lm_head.weight", mismatched)]) == "loaded"
    assert forwarded == [("lm_head.weight", mismatched)]


def test_real_quant_load_weights_detaches_pending_layerwise_views(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    layerwise_mod = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    model = torch.nn.Module()
    model.reload_root = torch.nn.Linear(2, 2, bias=False)
    model.unrelated = torch.nn.Linear(2, 2, bias=False)
    extension = _make_real_quant_extension(backend, model, [])
    extension._nrl_modelopt_reload_roots = (model.reload_root,)
    _patch_real_quant_load(monkeypatch, backend, [])

    source = torch.arange(4, dtype=torch.float32)
    incoming = source.view(2, 2)
    bound_arguments = types.SimpleNamespace(arguments={"loaded_weight": incoming})
    layerwise_info = types.SimpleNamespace(loaded_weights=[("weight", bound_arguments)])
    inspected = []

    def get_layerwise_info(module):
        inspected.append(module)
        if module is model.reload_root:
            return layerwise_info
        return types.SimpleNamespace(loaded_weights=[])

    monkeypatch.setattr(
        layerwise_mod,
        "get_layerwise_info",
        get_layerwise_info,
    )

    assert extension._load_weights([("reload_root.weight", incoming)]) == "loaded"

    detached = bound_arguments.arguments["loaded_weight"]
    assert detached.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    torch.testing.assert_close(detached, incoming)
    source.zero_()
    torch.testing.assert_close(detached, torch.arange(4).view(2, 2).float())
    assert inspected == [model.reload_root]


def test_real_quant_pre_ack_fence_is_device_wide_and_load_does_not_fence(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Linear(1, 1)
    extension = _make_real_quant_extension(backend, model, [])
    extension._nrl_modelopt_reload_roots = (model,)
    extension.device = types.SimpleNamespace(type="cuda")
    events = []

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda _self: True,
    )
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        lambda _self, _weights: events.append("load") or "loaded",
    )
    monkeypatch.setattr(
        backend,
        "_detach_pending_layerwise_weights",
        lambda _roots, _storage_ptrs: events.append("detach"),
    )
    monkeypatch.setattr(backend.torch, "device", lambda _device: nullcontext())
    monkeypatch.setattr(
        backend.torch.accelerator,
        "synchronize",
        lambda: events.append("sync"),
    )
    monkeypatch.setattr(
        backend.torch.cuda,
        "current_stream",
        lambda: pytest.fail("real quant must use one device-wide IPC ACK fence"),
    )

    assert extension._load_weights([("weight", torch.ones(1))]) == "loaded"
    assert events == ["load", "detach"]

    extension._synchronize_before_ipc_data_ack()
    assert events == ["load", "detach", "sync"]


@pytest.mark.parametrize("load_numel", [0, 10])
def test_real_quant_rejects_incomplete_modelopt_layerwise_reload(
    monkeypatch, load_numel
):
    backend = _import_vllm_quant_backend(monkeypatch)
    layerwise_mod = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]

    modelopt_module = types.ModuleType(
        "vllm.model_executor.layers.quantization.modelopt"
    )
    modelopt_base = type("ModelOptNvFp4FusedMoE", (), {})
    modelopt_module.ModelOptNvFp4FusedMoE = modelopt_base
    modelopt_module.ModelOptNvFp4LinearMethod = type(
        "ModelOptNvFp4LinearMethod", (), {}
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.modelopt",
        modelopt_module,
    )
    experts = torch.nn.Module()
    experts.quant_method = type("NemoModelOptNvFp4FusedMoE", (modelopt_base,), {})()
    model = torch.nn.Module()
    model.experts = experts
    info = types.SimpleNamespace(
        load_numel=load_numel,
        load_numel_total=12,
        loaded_weights=[("w13_weight", object())] if load_numel else [],
    )
    monkeypatch.setattr(layerwise_mod, "get_layerwise_info", lambda _module: info)

    with pytest.raises(
        RuntimeError,
        match=rf"experts: {load_numel}/12 elements",
    ):
        backend._require_complete_modelopt_layerwise_reload(model)


def test_real_quant_accepts_processed_modelopt_layerwise_reload(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    layerwise_mod = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]

    modelopt_module = types.ModuleType(
        "vllm.model_executor.layers.quantization.modelopt"
    )
    modelopt_module.ModelOptNvFp4FusedMoE = type("ModelOptNvFp4FusedMoE", (), {})
    modelopt_base = type("ModelOptNvFp4LinearMethod", (), {})
    modelopt_module.ModelOptNvFp4LinearMethod = modelopt_base
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.modelopt",
        modelopt_module,
    )
    linear = torch.nn.Module()
    linear.quant_method = type("NemoModelOptW4A16LinearMethod", (modelopt_base,), {})()
    model = torch.nn.Module()
    model.linear = linear
    info = types.SimpleNamespace(
        load_numel=0,
        load_numel_total=None,
        loaded_weights=[],
    )
    monkeypatch.setattr(layerwise_mod, "get_layerwise_info", lambda _module: info)

    backend._require_complete_modelopt_layerwise_reload(model)


def test_real_quant_scopes_native_reload_away_from_mamba_alias_buffers(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    modelopt_module = sys.modules["vllm.model_executor.layers.quantization.modelopt"]
    attention_module = sys.modules["vllm.model_executor.layers.attention"]
    kv_cache_module = sys.modules["vllm.model_executor.layers.quantization.kv_cache"]

    class MambaMixer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1d = torch.nn.Linear(3, 2, bias=False)
            self.register_buffer(
                "conv_weights",
                self.conv1d.weight.detach().view(-1),
                persistent=False,
            )

    class KVAttention(attention_module.Attention):
        def __init__(self):
            super().__init__()
            self.quant_method = kv_cache_module.BaseKVCacheMethod()
            self.kv_cache_dtype = "fp8"
            self.projection = _mark_as_modelopt_layer(torch.nn.Linear(1, 1))

    model = torch.nn.Module()
    model.mamba = MambaMixer()
    model.experts = _mark_as_modelopt_layer(torch.nn.Linear(1, 1))
    model.attention = KVAttention()

    assert backend._modelopt_layerwise_reload_roots(
        model,
        include_fp8_kv_cache=False,
    ) == [model.experts, model.attention.projection]
    assert backend._modelopt_layerwise_reload_roots(
        model,
        include_fp8_kv_cache=True,
    ) == [model.experts, model.attention]

    model.attention.kv_cache_dtype = "auto"
    assert backend._modelopt_layerwise_reload_roots(
        model,
        include_fp8_kv_cache=True,
    ) == [model.experts, model.attention.projection]
    model.attention.kv_cache_dtype = "fp8"

    model.shared = torch.nn.Module()
    model.shared.experts = model.experts
    assert backend._modelopt_layerwise_reload_roots(
        model,
        include_fp8_kv_cache=True,
    ) == [model.experts, model.attention]

    for roots in (
        backend._modelopt_layerwise_reload_roots(model, include_fp8_kv_cache=False),
        backend._modelopt_layerwise_reload_roots(model, include_fp8_kv_cache=True),
    ):
        assert model.mamba not in roots
        assert model.mamba.conv1d not in roots


def test_real_quant_caches_scoped_reload_roots(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Linear(1, 1)
    extension = _make_real_quant_extension(backend, model, [])
    extension._nrl_modelopt_reload_roots = None
    selected_roots = [model]
    calls = []

    def select_modelopt_roots(model_arg, *, include_fp8_kv_cache):
        calls.append((model_arg, include_fp8_kv_cache))
        return selected_roots

    monkeypatch.setattr(
        backend,
        "_modelopt_layerwise_reload_roots",
        select_modelopt_roots,
    )

    first = extension._get_modelopt_reload_roots()
    second = extension._get_modelopt_reload_roots()

    assert first is second
    assert first == (model,)
    assert calls == [(model, False)]


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


def test_real_quant_collective_reload_uses_vllm_layerwise_lifecycle(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    base_backend = _base_vllm_backend()
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]

    model = _mark_as_modelopt_layer(torch.nn.Linear(1, 1))
    model_config = object()
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(model=model)
    extension.model_config = model_config
    extension.device = torch.device("cpu")
    extension.state_dict_info = {}
    extension.model_update_group = object()
    calls = []

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    monkeypatch.setattr(
        reload_mod,
        "initialize_layerwise_reload",
        lambda model_arg: calls.append(("initialize", model_arg)),
    )
    monkeypatch.setattr(
        base_backend,
        "packed_broadcast_consumer",
        lambda **kwargs: calls.append(("consume", kwargs["post_unpack_func"].__name__)),
    )
    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        lambda model_arg, config_arg: calls.append(("finalize", model_arg, config_arg)),
    )
    monkeypatch.setattr(
        backend.torch.accelerator,
        "synchronize",
        lambda: calls.append("sync"),
    )

    assert extension.update_weights_from_collective() is True
    assert calls == [
        ("initialize", model),
        ("consume", "_load_weights"),
        ("finalize", model, model_config),
        "sync",
    ]


def test_real_quant_collective_reload_raises_on_failure(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    base_backend = _base_vllm_backend()
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]

    model = _mark_as_modelopt_layer(torch.nn.Linear(1, 1))
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(model=model)
    extension.model_config = object()
    extension.device = torch.device("cpu")
    extension.state_dict_info = {}
    extension.model_update_group = object()
    calls = []

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    monkeypatch.setattr(
        reload_mod,
        "initialize_layerwise_reload",
        lambda model_arg: calls.append(("initialize", model_arg)),
    )

    def _raise_consume(**kwargs):
        raise ValueError("broadcast boom")

    monkeypatch.setattr(base_backend, "packed_broadcast_consumer", _raise_consume)
    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        lambda _model, _model_config: pytest.fail(
            "a failed transfer must not be finalized"
        ),
    )

    with pytest.raises(RuntimeError, match="collective refit failed"):
        extension.update_weights_from_collective()
    assert calls == [("initialize", model)]


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


def test_real_quant_ipc_complete_finalizes_vllm_layerwise_reload_and_acks(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]
    from nemo_rl.models.policy.utils import IPCProtocol

    class FakeSocket:
        def __init__(self):
            self.sent = []

        def recv_pyobj(self):
            return IPCProtocol.COMPLETE

        def send(self, payload):
            self.sent.append(payload)

    model = _mark_as_modelopt_layer(torch.nn.Linear(1, 1))
    model_config = object()
    socket = FakeSocket()
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(model=model)
    extension.model_config = model_config
    extension.device = torch.device("cpu")
    extension.zmq_socket = socket
    extension.state_dict_info = {}
    extension.maybe_init_zmq = lambda: None
    calls = []

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    monkeypatch.setattr(
        reload_mod,
        "initialize_layerwise_reload",
        lambda model_arg: calls.append(("initialize", model_arg)),
    )
    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        lambda model_arg, config_arg: calls.append(("finalize", model_arg, config_arg)),
    )
    monkeypatch.setattr(
        backend.torch.accelerator,
        "synchronize",
        lambda: calls.append("sync"),
    )
    monkeypatch.setattr(
        backend.torch.cuda, "empty_cache", lambda: calls.append("empty")
    )

    assert extension.update_weights_via_ipc_zmq() is True
    assert calls == [
        ("initialize", model),
        ("finalize", model, model_config),
        "sync",
        "empty",
    ]
    assert socket.sent == [IPCProtocol.ACK.value.encode()]


def test_real_quant_ipc_finalize_failure_acks_complete(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]
    from nemo_rl.models.policy.utils import IPCProtocol

    socket = types.SimpleNamespace(
        recv_pyobj=lambda: IPCProtocol.COMPLETE,
        sent=[],
    )
    socket.send = socket.sent.append
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=_mark_as_modelopt_layer(torch.nn.Linear(1, 1))
    )
    extension.model_config = object()
    extension.device = torch.device("cpu")
    extension.zmq_socket = socket
    extension.state_dict_info = {}
    extension.maybe_init_zmq = lambda: None
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda _self: True,
    )

    def fail_finalize(_model, _model_config):
        raise RuntimeError("bad scales")

    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        fail_finalize,
    )

    with pytest.raises(
        RuntimeError, match="ModelOpt real-quant refit post-processing failed"
    ):
        extension.update_weights_via_ipc_zmq()
    assert socket.sent == [IPCProtocol.ACK.value.encode()]


@pytest.mark.parametrize(
    ("payload_groups", "state_dict_info", "error"),
    [
        (
            [["decoder.weight"]],
            {
                "decoder.weight": ([1], torch.float32),
                "decoder.bias": ([1], torch.float32),
            },
            "missing keys",
        ),
        (
            [["decoder.weight"], ["decoder.weight"]],
            {"decoder.weight": ([1], torch.float32)},
            "duplicate keys",
        ),
        (
            [["decoder.weight", "decoder.weight"]],
            {"decoder.weight": ([1], torch.float32)},
            "duplicate keys",
        ),
        (
            [["unexpected.weight"]],
            {"decoder.weight": ([1], torch.float32)},
            "unexpected keys",
        ),
    ],
)
def test_real_quant_ipc_rejects_invalid_key_manifest(
    monkeypatch, payload_groups, state_dict_info, error
):
    backend = _import_vllm_quant_backend(monkeypatch)
    base_backend = _base_vllm_backend()
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]
    from nemo_rl.models.policy.utils import IPCProtocol

    payload_buffer = torch.tensor([1.0], dtype=torch.float32).view(torch.uint8)
    used_bytes = base_backend.calculate_aligned_size(payload_buffer.numel())
    payloads = [
        ("ipc-handle", keys, used_bytes * len(keys)) for keys in payload_groups
    ] + [IPCProtocol.COMPLETE]

    class FakeSocket:
        def __init__(self):
            self.payloads = iter(payloads)
            self.sent = []

        def recv_pyobj(self):
            return next(self.payloads)

        def send(self, payload):
            self.sent.append(payload)

    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(model=torch.nn.Linear(1, 1))
    extension.model_config = object()
    extension.device = torch.device("cuda:0")
    extension.zmq_socket = FakeSocket()
    extension.state_dict_info = state_dict_info
    extension.maybe_init_zmq = lambda: None
    extension._load_weights = lambda _weights: None
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda _self: True,
    )
    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        lambda _model, _model_config: pytest.fail(
            "an invalid refit must not be finalized"
        ),
    )
    monkeypatch.setattr(
        base_backend,
        "rebuild_cuda_tensor_from_ipc",
        lambda _ipc_handle, _device_index: payload_buffer,
    )
    monkeypatch.setattr(
        base_backend.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(backend.torch.accelerator, "synchronize", lambda: None)

    with pytest.raises(RuntimeError, match=error):
        extension.update_weights_via_ipc_zmq()
    assert extension.zmq_socket.sent == [IPCProtocol.ACK.value.encode()] * len(payloads)


def test_real_quant_ipc_payload_loads_weights_and_handles_gpt_oss(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    base_backend = _base_vllm_backend()
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]
    from nemo_rl.models.policy.utils import IPCProtocol

    payload_weight = torch.tensor([1.0, 2.0], dtype=torch.float32)
    payload_buffer = payload_weight.view(torch.uint8)
    used_bytes = base_backend.calculate_aligned_size(payload_weight.nbytes)
    loaded = []
    calls = []
    view_refs = []

    class FakeSocket:
        def __init__(self):
            self.payloads = iter(
                [
                    ("ipc-handle", ["decoder.weight"], used_bytes),
                    ("ipc-handle", ["decoder.bias"], used_bytes),
                    IPCProtocol.COMPLETE,
                ]
            )
            self.sent = []

        def recv_pyobj(self):
            return next(self.payloads)

        def send(self, payload):
            if len(self.sent) < 2:
                assert view_refs
                assert all(view_ref() is None for view_ref in view_refs)
                calls.append("views_released")
            self.sent.append(payload)

    model = _mark_as_modelopt_layer(torch.nn.Linear(1, 1))
    model_config = object()
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(architectures=["GptOssForCausalLM"])
        ),
    )
    extension.model_config = model_config
    extension.device = torch.device("cuda:0")
    extension.zmq_socket = FakeSocket()
    extension.state_dict_info = {
        "decoder.weight": ([2], torch.float32),
        "decoder.bias": ([2], torch.float32),
    }
    extension.maybe_init_zmq = lambda: None

    def load_weights(weights):
        for name, weight in weights:
            view_refs.append(weakref.ref(weight))
            loaded.append((name, weight.clone()))

    extension._load_weights = load_weights

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    monkeypatch.setattr(
        reload_mod,
        "initialize_layerwise_reload",
        lambda model_arg: calls.append(("initialize", model_arg)),
    )
    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        lambda model_arg, config_arg: calls.append(("finalize", model_arg, config_arg)),
    )
    monkeypatch.setattr(
        base_backend,
        "rebuild_cuda_tensor_from_ipc",
        lambda ipc_handle, device_index: payload_buffer,
    )
    monkeypatch.setattr(
        base_backend.torch.cuda,
        "current_stream",
        lambda: pytest.fail("real quant must not use a current-stream IPC ACK fence"),
    )
    monkeypatch.setattr(
        backend.torch.accelerator,
        "synchronize",
        lambda: calls.append("sync"),
    )
    monkeypatch.setattr(
        backend.torch.cuda, "empty_cache", lambda: calls.append("empty")
    )
    monkeypatch.setattr(base_backend.gc, "collect", lambda: calls.append("gc"))

    assert extension.update_weights_via_ipc_zmq() is True

    assert extension.zmq_socket.sent == [
        IPCProtocol.ACK.value.encode(),
        IPCProtocol.ACK.value.encode(),
        IPCProtocol.ACK.value.encode(),
    ]
    assert [name for name, _ in loaded] == ["decoder.weight", "decoder.bias"]
    for _, loaded_weight in loaded:
        torch.testing.assert_close(loaded_weight, payload_weight)
    assert calls == [
        ("initialize", model),
        "sync",
        "views_released",
        "sync",
        "views_released",
        ("finalize", model, model_config),
        "sync",
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


def _nvfp4_source_format() -> dict:
    return {
        "num_bits": "e2m1",
        "block_sizes": {
            -1: 16,
            "type": "dynamic",
            "scale_bits": "e4m3",
        },
    }


def test_resolve_nvfp4_real_quant_mode_detects_model_specific_w4a16(monkeypatch):
    resolved = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*mixer.experts.*weight_quantizer",
                "cfg": _nvfp4_source_format(),
            },
            {
                "quantizer_name": "*mlp.experts*weight_quantizer",
                "cfg": _nvfp4_source_format(),
            },
        ],
        "algorithm": "max",
    }
    monkeypatch.setattr(modelopt_utils, "resolve_quant_cfg", lambda _: resolved)

    assert resolve_nvfp4_real_quant_mode("custom-nvfp4-config") == "w4a16"


def test_resolve_nvfp4_real_quant_mode_detects_w4a4(monkeypatch):
    resolved = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*mlp.experts*weight_quantizer",
                "cfg": _nvfp4_source_format(),
            },
            {
                "quantizer_name": "*mlp.experts*input_quantizer",
                "cfg": _nvfp4_source_format(),
            },
            {
                "quantizer_name": "*mlp.experts*input_quantizer",
                "parent_class": "nn.LeakyReLU",
                "enable": False,
            },
        ],
        "algorithm": "max",
    }
    monkeypatch.setattr(modelopt_utils, "resolve_quant_cfg", lambda _: resolved)

    assert resolve_nvfp4_real_quant_mode("not-named-after-the-format") == "w4a4"


def test_resolve_nvfp4_real_quant_mode_honors_late_generic_disable(monkeypatch):
    resolved = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": _nvfp4_source_format(),
            },
            {
                "quantizer_name": "*mlp.experts*input_quantizer",
                "cfg": _nvfp4_source_format(),
            },
            {"quantizer_name": "*input_quantizer", "enable": False},
        ],
        "algorithm": "max",
    }
    monkeypatch.setattr(modelopt_utils, "resolve_quant_cfg", lambda _: resolved)

    assert resolve_nvfp4_real_quant_mode("disabled-input") == "w4a16"


@pytest.mark.parametrize(
    ("weight_format", "input_format", "error_match"),
    [
        (
            {"num_bits": "e4m3", "axis": None},
            {"num_bits": "e4m3", "axis": None},
            "only block-16 NVFP4.*weights",
        ),
        (
            _nvfp4_source_format(),
            {"num_bits": "e4m3", "axis": None},
            "only block-16 NVFP4.*input activations",
        ),
        (
            _nvfp4_source_format(),
            [_nvfp4_source_format()],
            "single NVFP4 input activations format",
        ),
        (
            _nvfp4_source_format(),
            {
                "num_bits": "e2m1",
                "block_sizes": {
                    -1: 32,
                    "type": "dynamic",
                    "scale_bits": "e4m3",
                },
            },
            "only block-16 NVFP4.*input activations",
        ),
    ],
    ids=["fp8", "w4a8", "sequential-activation", "unsupported-nvfp4-block"],
)
def test_resolve_nvfp4_real_quant_mode_rejects_unsupported_formats(
    monkeypatch,
    weight_format,
    input_format,
    error_match,
):
    resolved = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": weight_format},
            {"quantizer_name": "*input_quantizer", "cfg": input_format},
        ],
        "algorithm": "max",
    }
    monkeypatch.setattr(modelopt_utils, "resolve_quant_cfg", lambda _: resolved)

    with pytest.raises(ValueError, match=error_match):
        resolve_nvfp4_real_quant_mode("unsupported-real-quant-config")


def test_resolve_nvfp4_real_quant_mode_rejects_mixed_activation_formats(
    monkeypatch,
):
    resolved = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*mixer.experts.*weight_quantizer",
                "cfg": _nvfp4_source_format(),
            },
            {
                "quantizer_name": "*mlp.experts*weight_quantizer",
                "cfg": _nvfp4_source_format(),
            },
            {
                "quantizer_name": "*mixer.experts.*input_quantizer",
                "cfg": _nvfp4_source_format(),
            },
            {
                "quantizer_name": "*mlp.experts*input_quantizer",
                "cfg": {"num_bits": "e4m3", "axis": None},
            },
        ],
        "algorithm": "max",
    }
    monkeypatch.setattr(modelopt_utils, "resolve_quant_cfg", lambda _: resolved)

    with pytest.raises(ValueError, match="only block-16 NVFP4.*input activations"):
        resolve_nvfp4_real_quant_mode("mixed-input-formats")


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


def test_resolve_quant_cfg_defaults_missing_algorithm_to_max(monkeypatch):
    modelopt_recipe = pytest.importorskip("modelopt.recipe")

    monkeypatch.setattr(
        modelopt_recipe,
        "load_config",
        lambda config_name: {"quant_cfg": [{"name": config_name}]},
    )

    assert resolve_quant_cfg("unit-test-recipe") == {
        "quant_cfg": [{"name": "unit-test-recipe"}],
        "algorithm": "max",
    }


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


def test_register_nemo_modelopt_nvfp4_uses_public_vllm_registry(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)

    register_nemo_modelopt_nvfp4()

    assert set(fake_vllm.registry) == {
        NEMO_MODELOPT_W4A4,
        NEMO_MODELOPT_W4A16,
    }
    w4a4_config = fake_vllm.registry[NEMO_MODELOPT_W4A4]()
    assert w4a4_config.get_name() == NEMO_MODELOPT_W4A4

    source_config = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    w4a16_config = fake_vllm.registry[NEMO_MODELOPT_W4A16].from_config(source_config)
    assert source_config["quant_algo"] == "W4A16_NVFP4"
    assert w4a16_config.parsed_config["quant_algo"] == "NVFP4"
    assert w4a16_config.get_name() == NEMO_MODELOPT_W4A16

    with pytest.raises(ValueError, match="requires quant_algo='W4A16_NVFP4'"):
        fake_vllm.registry[NEMO_MODELOPT_W4A16].from_config({"quant_algo": "NVFP4"})


def test_registered_configs_select_only_the_exact_custom_override(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)
    register_nemo_modelopt_nvfp4()
    w4a4_config_cls = fake_vllm.registry[NEMO_MODELOPT_W4A4]
    w4a16_config_cls = fake_vllm.registry[NEMO_MODELOPT_W4A16]

    assert (
        w4a4_config_cls.override_quantization_method(
            {"quant_algo": "NVFP4"}, NEMO_MODELOPT_W4A4
        )
        == NEMO_MODELOPT_W4A4
    )
    assert (
        w4a16_config_cls.override_quantization_method(
            {"quantization": {"quant_algo": "W4A16_NVFP4"}},
            NEMO_MODELOPT_W4A16,
        )
        == NEMO_MODELOPT_W4A16
    )
    assert (
        w4a4_config_cls.override_quantization_method(
            {"quant_algo": "NVFP4"}, NEMO_MODELOPT_W4A16
        )
        is None
    )
    assert (
        w4a4_config_cls.override_quantization_method(
            {"quant_algo": "W4A16_NVFP4"}, NEMO_MODELOPT_W4A4
        )
        is None
    )
    assert (
        w4a16_config_cls.override_quantization_method(
            {"quant_algo": "W4A16_NVFP4"}, "modelopt"
        )
        is None
    )


def test_registered_w4a16_dense_method_supports_weight_loader_v2(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)

    register_nemo_modelopt_nvfp4()

    w4a16_config_cls = fake_vllm.registry[NEMO_MODELOPT_W4A16]
    assert fake_vllm.weight_loader_v2_supported == [
        w4a16_config_cls.LinearMethodCls.__name__
    ]


def test_registered_w4a4_moe_loader_is_sanitizer_compatible(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)
    register_nemo_modelopt_nvfp4()

    config = fake_vllm.registry[NEMO_MODELOPT_W4A4]()
    quant_method = config.FusedMoEMethodCls(config, object())

    class FakeMoeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_method = quant_method
            self.w13_input_scale = torch.nn.Parameter(torch.zeros(2, 2))
            self.w2_input_scale = torch.nn.Parameter(torch.zeros(2))

        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return expert_id

    layer = FakeMoeLayer()
    quant_method.create_weights(layer)

    w13_loader = layer.w13_input_scale.weight_loader
    assert isinstance(w13_loader, types.MethodType)
    assert w13_loader.__self__ is layer

    layer_ref_sentinel = object()
    layer.w13_input_scale.weight_loader = w13_loader.__func__.__get__(
        layer_ref_sentinel
    )
    assert layer.w13_input_scale.weight_loader.__self__ is layer_ref_sentinel
    layer.w13_input_scale.weight_loader = (
        layer.w13_input_scale.weight_loader.__func__.__get__(layer)
    )
    w13_loader = layer.w13_input_scale.weight_loader
    assert w13_loader.__self__ is layer

    assert w13_loader(
        layer.w13_input_scale,
        torch.tensor(1.0),
        "gate.input_scale",
        "w1",
        0,
        True,
    )
    assert w13_loader(
        layer.w13_input_scale,
        torch.tensor(2.0),
        "up.input_scale",
        "w3",
        0,
        True,
    )
    assert layer.w2_input_scale.weight_loader(
        layer.w2_input_scale,
        torch.tensor(3.0),
        "down.input_scale",
        "w2",
        1,
        True,
    )

    torch.testing.assert_close(layer.w13_input_scale[0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(layer.w2_input_scale, torch.tensor([0.0, 3.0]))


def test_registered_w4a4_moe_materializes_initial_input_scales(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)
    register_nemo_modelopt_nvfp4()
    config = fake_vllm.registry[NEMO_MODELOPT_W4A4]()
    quant_method = config.FusedMoEMethodCls(config, object())

    layer = torch.nn.Module()
    w13_input_scale = torch.nn.Parameter(
        torch.tensor([2.0]).expand(4), requires_grad=False
    )
    w2_input_scale = torch.nn.Parameter(
        torch.tensor([3.0]).expand(4), requires_grad=False
    )
    layer.register_parameter("w13_input_scale", w13_input_scale)
    layer.register_parameter("w2_input_scale", w2_input_scale)

    quant_method.process_weights_after_loading(layer)

    assert layer.w13_input_scale is w13_input_scale
    assert layer.w2_input_scale is w2_input_scale
    assert layer.w13_input_scale.is_contiguous()
    assert layer.w2_input_scale.is_contiguous()
    torch.testing.assert_close(layer.w13_input_scale, torch.full((4,), 2.0))
    torch.testing.assert_close(layer.w2_input_scale, torch.full((4,), 3.0))
    with torch.no_grad():
        layer.w13_input_scale.copy_(torch.arange(4, dtype=torch.float32))
        layer.w2_input_scale.copy_(torch.arange(4, dtype=torch.float32))


def test_registered_w4a4_moe_refreshes_stable_activation_scales(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)
    register_nemo_modelopt_nvfp4()
    config = fake_vllm.registry[NEMO_MODELOPT_W4A4]()
    quant_method = config.FusedMoEMethodCls(config, object())

    original_kernel = object()
    original_a1_gscale = torch.full((4,), 1.0)
    original_a2_gscale = torch.full((4,), 0.5)
    original_quant_config = types.SimpleNamespace(
        a1_gscale=original_a1_gscale,
        a2_gscale=original_a2_gscale,
    )
    quant_method.moe_kernel = original_kernel
    quant_method.moe_quant_config = original_quant_config
    a1_data_ptr = original_a1_gscale.data_ptr()
    a2_data_ptr = original_a2_gscale.data_ptr()

    layer = torch.nn.Module()
    layer.register_parameter(
        "w13_input_scale",
        torch.nn.Parameter(torch.full((4,), 4.0), requires_grad=False),
    )
    layer.register_parameter(
        "w2_input_scale",
        torch.nn.Parameter(torch.full((4,), 5.0), requires_grad=False),
    )

    quant_method.process_weights_after_loading(layer)

    assert quant_method.moe_kernel is original_kernel
    assert quant_method.moe_quant_config is original_quant_config
    assert original_quant_config.a1_gscale.data_ptr() == a1_data_ptr
    assert original_quant_config.a2_gscale.data_ptr() == a2_data_ptr
    torch.testing.assert_close(original_quant_config.a1_gscale, torch.full((4,), 0.25))
    torch.testing.assert_close(original_quant_config.a2_gscale, torch.full((4,), 0.2))


def test_registered_w4a16_dense_method_uses_marlin_weight_only(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)
    register_nemo_modelopt_nvfp4()
    config = fake_vllm.registry[NEMO_MODELOPT_W4A16].from_config(
        {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    )
    quant_method = config.LinearMethodCls(config)

    created_layer = torch.nn.Module()
    quant_method.create_weights(created_layer)
    assert not hasattr(created_layer, "input_scale")

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones(2, 1), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.tensor([[-1.0, 2.0], [0.5, -4.0]]),
        requires_grad=False,
    )
    layer.weight_scale_2 = torch.nn.Parameter(
        torch.tensor([2.0, 3.0]),
        requires_grad=False,
    )
    layer.output_size_per_partition = 2
    layer.input_size_per_partition = 2

    quant_method.process_weights_after_loading(layer)
    output = quant_method.apply(layer, torch.ones(1, 2))

    assert output == "output"
    assert not hasattr(layer, "weight_scale_2")
    torch.testing.assert_close(
        layer.weight_scale,
        torch.tensor([[1.0, 2.0], [0.5, 4.0]]),
    )
    torch.testing.assert_close(layer.weight_global_scale, torch.tensor(3.0))
    assert fake_vllm.events[0] == ("process_marlin_kernel", layer)
    event_name, kernel_args = fake_vllm.events[1]
    assert event_name == "apply_marlin_kernel"
    assert kernel_args["layer"] is layer
    torch.testing.assert_close(kernel_args["x"], torch.ones(1, 2))
    assert kernel_args["bias"] is None


@pytest.mark.parametrize(
    ("is_act_and_mul", "packed_hidden_size", "expected_padded_size"),
    [
        (False, 64, 192),
        (True, 32, 256),
    ],
)
def test_pad_nvfp4_moe_for_marlin_uses_hidden_size_tile_alignment(
    is_act_and_mul,
    packed_hidden_size,
    expected_padded_size,
):
    num_shards = 2 if is_act_and_mul else 1
    intermediate_size = 144
    w13 = torch.ones(
        1,
        num_shards * intermediate_size,
        packed_hidden_size,
    )
    w13_scale = torch.ones(1, num_shards * intermediate_size, 2)
    w2 = torch.ones(1, 2, intermediate_size // 2)
    w2_scale = torch.ones(1, 2, intermediate_size // 16)

    padded_w13, padded_w13_scale, padded_w2, padded_w2_scale, padded_size = (
        _pad_nvfp4_moe_for_marlin(
            w13,
            w13_scale,
            w2,
            w2_scale,
            is_act_and_mul=is_act_and_mul,
        )
    )

    assert padded_size == expected_padded_size
    assert padded_w13.shape == (
        1,
        num_shards * expected_padded_size,
        packed_hidden_size,
    )
    assert padded_w13_scale.shape == (1, num_shards * expected_padded_size, 2)
    assert padded_w2.shape == (1, 2, expected_padded_size // 2)
    assert padded_w2_scale.shape == (1, 2, expected_padded_size // 16)

    padded_w13 = padded_w13.view(
        1, num_shards, expected_padded_size, packed_hidden_size
    )
    padded_w13_scale = padded_w13_scale.view(1, num_shards, expected_padded_size, 2)
    assert torch.all(padded_w13[:, :, :intermediate_size] == 1)
    assert torch.count_nonzero(padded_w13[:, :, intermediate_size:]) == 0
    assert torch.all(padded_w13_scale[:, :, :intermediate_size] == 1)
    assert torch.count_nonzero(padded_w13_scale[:, :, intermediate_size:]) == 0
    assert torch.all(padded_w2[..., : intermediate_size // 2] == 1)
    assert torch.count_nonzero(padded_w2[..., intermediate_size // 2 :]) == 0
    assert torch.all(padded_w2_scale[..., : intermediate_size // 16] == 1)
    assert torch.count_nonzero(padded_w2_scale[..., intermediate_size // 16 :]) == 0


def test_registered_w4a16_moe_create_weights_keeps_checkpoint_layout(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)
    register_nemo_modelopt_nvfp4()
    config = fake_vllm.registry[NEMO_MODELOPT_W4A16].from_config(
        {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    )
    quant_method = config.FusedMoEMethodCls(
        config,
        types.SimpleNamespace(is_act_and_mul=False),
    )
    layer = torch.nn.Module()

    quant_method.create_weights(
        layer,
        num_experts=2,
        hidden_size=4096,
        intermediate_size_per_partition=672,
        params_dtype=torch.bfloat16,
    )

    assert not hasattr(layer, "w13_input_scale")
    assert not hasattr(layer, "w2_input_scale")
    assert fake_vllm.events == [
        (
            "native_create_weights",
            layer,
            (2, 4096, 672, torch.bfloat16),
            {},
        )
    ]


def test_registered_w4a16_moe_preserves_kernel_during_reload(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)
    register_nemo_modelopt_nvfp4()
    config = fake_vllm.registry[NEMO_MODELOPT_W4A16].from_config(
        {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    )
    quant_method = config.FusedMoEMethodCls(
        config,
        types.SimpleNamespace(is_act_and_mul=False),
    )
    original_kernel = object()
    original_quant_config = object()
    quant_method.moe_kernel = original_kernel
    quant_method.moe_quant_config = original_quant_config

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.ones(1, 80, 32))
    layer.w13_weight_scale = torch.nn.Parameter(-torch.ones(1, 80, 2))
    layer.w13_weight_scale_2 = torch.nn.Parameter(torch.ones(1, 1))
    layer.w2_weight = torch.nn.Parameter(torch.ones(1, 2, 40))
    layer.w2_weight_scale = torch.nn.Parameter(-torch.ones(1, 2, 5))
    layer.w2_weight_scale_2 = torch.nn.Parameter(torch.ones(1))
    layer.moe_config = types.SimpleNamespace(intermediate_size_per_partition=80)
    layer.shared_experts = None
    layer._maybe_init_expert_routing_tables = lambda: None

    quant_method.process_weights_after_loading(layer)

    assert quant_method.moe_kernel is original_kernel
    assert quant_method.moe_quant_config is original_quant_config
    assert layer.moe_config.intermediate_size_per_partition == 80
    assert layer.w13_weight.shape == (1, 128, 32)
    assert layer.w13_weight_scale.shape == (1, 128, 2)
    assert layer.w2_weight.shape == (1, 2, 64)
    assert layer.w2_weight_scale.shape == (1, 2, 8)
    assert torch.all(layer.w13_weight_scale >= 0)
    assert torch.all(layer.w2_weight_scale >= 0)
    assert fake_vllm.events == [("native_process_moe", 128)]
