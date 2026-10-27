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
from contextlib import contextmanager

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef

import nemo_rl.modelopt.utils as modelopt_utils
from nemo_rl.modelopt.utils import resolve_quant_cfg


@pytest.fixture(autouse=True)
def _install_optional_modelopt_config_api(monkeypatch):
    """Provide the ModelOpt recipe APIs when the optional dependency is absent."""
    try:
        import modelopt.recipe  # noqa: F401
        import modelopt.torch.quantization  # noqa: F401

        return
    except ImportError:
        pass

    module_names = (
        "modelopt",
        "modelopt.recipe",
        "modelopt.torch",
        "modelopt.torch.quantization",
    )
    for module_name in module_names:
        module = types.ModuleType(module_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)

    def missing_recipe(config_name):
        raise FileNotFoundError(config_name)

    sys.modules["modelopt.recipe"].load_config = missing_recipe


def _install_fake_vllm_worker(monkeypatch):
    """Install the minimal vLLM worker hierarchy needed by the backend import."""
    module_names = ["vllm", "vllm.distributed", "vllm.v1", "vllm.v1.worker"]
    modules = {}
    for module_name in module_names:
        module = types.ModuleType(module_name)
        module.__path__ = []
        modules[module_name] = module
        monkeypatch.setitem(sys.modules, module_name, module)

    gpu_worker_module = types.ModuleType("vllm.v1.worker.gpu_worker")

    class FakeVllmWorker:
        def load_model(self, *, load_dummy_weights: bool = False) -> None:
            raise AssertionError("Tests must replace the fake worker's load_model")

    gpu_worker_module.Worker = FakeVllmWorker
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker", gpu_worker_module)
    parallel_state_module = types.ModuleType("vllm.distributed.parallel_state")

    def get_pp_group() -> types.SimpleNamespace:
        return types.SimpleNamespace(is_last_rank=True)

    parallel_state_module.get_pp_group = get_pp_group
    monkeypatch.setitem(
        sys.modules, "vllm.distributed.parallel_state", parallel_state_module
    )
    modules["vllm"].distributed = modules["vllm.distributed"]
    modules["vllm.distributed"].parallel_state = parallel_state_module
    modules["vllm"].v1 = modules["vllm.v1"]
    modules["vllm.v1"].worker = modules["vllm.v1.worker"]
    modules["vllm.v1.worker"].gpu_worker = gpu_worker_module


def _clear_vllm_backend_modules(monkeypatch):
    for module_name in (
        "nemo_rl.modelopt.models.generation.vllm_quant_backend",
        "nemo_rl.models.generation.vllm.vllm_backend",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)


def _import_vllm_quant_backend(monkeypatch):
    """Import the NeMo-RL backend without requiring the vLLM C extension."""
    _install_fake_vllm_worker(monkeypatch)
    _install_fake_vllm_reload(monkeypatch)
    _install_fake_modelopt_tensor_quantizer(monkeypatch)
    _clear_vllm_backend_modules(monkeypatch)
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

    config_module = types.ModuleType("vllm.config")
    config_module.current = None

    @contextmanager
    def set_current_vllm_config(config):
        previous = config_module.current
        config_module.current = config
        try:
            yield
        finally:
            config_module.current = previous

    def get_current_vllm_config():
        if config_module.current is None:
            raise AssertionError("Current vLLM config is not set")
        return config_module.current

    config_module.set_current_vllm_config = set_current_vllm_config
    config_module.get_current_vllm_config = get_current_vllm_config
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)

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
    hpc_module = types.ModuleType("vllm.model_executor.layers.hpc")

    class HpcModule(torch.nn.Module):
        pass

    hpc_module.HpcModule = HpcModule
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.hpc", hpc_module)
    return reload_module


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


@pytest.mark.parametrize("llm_kwargs", [{}, {"moe_backend": "triton"}])
def test_configure_quant_engine_kwargs_for_fake_quant(
    monkeypatch, tmp_path, llm_kwargs
):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)

    quant_cfg = "quant.yaml"
    (tmp_path / quant_cfg).touch()
    monkeypatch.chdir(tmp_path)

    llm_kwargs = dict(llm_kwargs)
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
    assert llm_kwargs["moe_backend"] == "triton"
    assert "quantization" not in llm_kwargs


@pytest.mark.parametrize(
    "moe_backend", ["auto", "flashinfer_trtllm", "flashinfer_cutlass", None]
)
def test_configure_fake_quant_rejects_non_triton_backend(monkeypatch, moe_backend):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)
    llm_kwargs = {"moe_backend": moe_backend}

    with pytest.raises(
        ValueError, match="fake quantization requires moe_backend='triton'"
    ):
        worker_mod._configure_quant_engine_kwargs(
            {"quant_cfg": "NVFP4_DEFAULT_CFG"}, llm_kwargs
        )

    assert llm_kwargs["moe_backend"] == moe_backend
    assert "VLLM_QUANT_CFG" not in os.environ


def test_quant_worker_forwards_snapshot_pythonpath_to_inner_vllm_workers():
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )

    assert "PYTHONPATH" in worker_mod._EXTRA_ENV_VARS


def test_configure_quant_engine_kwargs_preserves_checkpoint_extension(monkeypatch):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)
    cfg = {
        "quant_cfg": "examples/modelopt/quant_configs/nvfp4_w4a8_fp8.yaml",
        "refit_transport": "nixl",
        "refit_cfg": {"nixl": {}},
    }
    llm_kwargs = {}

    worker_mod._configure_quant_engine_kwargs(cfg, llm_kwargs)

    assert llm_kwargs["worker_extension_cls"] == (
        "nemo_rl.modelopt.models.generation.vllm_quant_backend."
        "VllmQuantInternalWorkerExtensionWithCheckpointEngine"
    )


@pytest.mark.vllm
def test_fake_quant_worker_inherits_nixl_worker():
    # Optional native dependencies are installed only in the vLLM test environment.
    from nemo_rl.modelopt.models.generation import (
        vllm_quant_patch as patch_mod,
    )
    from nemo_rl.models.generation.vllm.vllm_backend import NixlVllmWorker

    assert issubclass(patch_mod.FakeQuantWorker, NixlVllmWorker)


@pytest.mark.vllm
def test_real_quant_worker_makes_shared_nvfp4_moe_scales_writable():
    # Optional native dependencies are installed only in the vLLM test environment.
    from nemo_rl.modelopt.models.generation import (
        vllm_quant_patch as patch_mod,
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.Module()
            self.experts.w13_input_scale = torch.nn.Parameter(
                torch.tensor(3.0).expand(4), requires_grad=False
            )
            self.experts.w2_input_scale = torch.nn.Parameter(
                torch.tensor(5.0).expand(4), requires_grad=False
            )

    model = Model()
    model.experts.w13_input_scale.weight_loader = object()

    patch_mod._make_nvfp4_moe_input_scales_writable(model)

    assert model.experts.w13_input_scale.stride() == (1,)
    assert model.experts.w2_input_scale.stride() == (1,)
    assert hasattr(model.experts.w13_input_scale, "weight_loader")
    model.experts.w13_input_scale.copy_(torch.arange(4, dtype=torch.float32))
    model.experts.w2_input_scale.copy_(torch.arange(4, dtype=torch.float32))


@pytest.mark.vllm
@pytest.mark.parametrize(
    "backend",
    [
        "VLLM_CUTLASS",
        "FLASHINFER_TRTLLM",
        "FLASHINFER_CUTLASS",
        "FLASHINFER_CUTEDSL_BATCHED",
        "MARLIN",
    ],
)
def test_real_quant_moe_scales_use_native_reload_copyback(backend):
    # Optional native dependencies are installed only in the vLLM test environment.
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        make_nvfp4_moe_quant_config,
    )
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_reload,
        get_layerwise_info,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    from nemo_rl.modelopt.models.generation import (
        vllm_quant_patch as patch_mod,
    )

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    method = object.__new__(ModelOptNvFp4FusedMoE)
    method.nvfp4_backend = NvFp4MoeBackend[backend]
    layer.quant_method = method

    def process_weights(target):
        target.weight = torch.nn.Parameter(target.weight.clone(), requires_grad=False)
        method.moe_quant_config = make_nvfp4_moe_quant_config(
            backend=method.nvfp4_backend,
            w13_scale=target.weight,
            w2_scale=target.weight,
            w13_scale_2=target.weight,
            w2_scale_2=target.weight,
            a13_scale=target.weight,
            a2_scale=target.weight * 2,
        )
        method.moe_kernel = object()

    method.process_weights_after_loading = process_weights
    # The real model loader records checkpoint metadata before cold processing.
    record_metadata_for_reloading(layer)
    process_weights(layer)
    patch_mod._preserve_moe_reload_scales(layer)
    config, kernel = method.moe_quant_config, method.moe_kernel
    if backend != "MARLIN":
        assert layer._nrl_a1_gscale is config.a1_gscale
        assert layer._nrl_a2_gscale is config.a2_gscale
    assert not get_layerwise_info(layer).restore_metadata[1]

    for value in (2.0, 3.0, 1.0):
        initialize_layerwise_reload(layer)
        assert not layer._buffers  # No derived scales are checkpoint load targets.
        assert get_layerwise_info(layer).load_numel_total == 2
        layer.weight.weight_loader(layer.weight, torch.full((2,), value))
        finalize_layerwise_reload(layer, None)
        assert method.moe_quant_config is config and method.moe_kernel is kernel
        if backend != "MARLIN":
            assert layer._nrl_a1_gscale is config.a1_gscale
            assert layer._nrl_a2_gscale is config.a2_gscale
            torch.testing.assert_close(config.a1_gscale, torch.full((2,), 1 / value))
            torch.testing.assert_close(
                config.a2_gscale, torch.full((2,), 1 / (value * 2))
            )
        assert config.w1_scale is layer.weight
        torch.testing.assert_close(layer.weight, torch.full((2,), value))


@pytest.mark.vllm
def test_real_quant_moe_rejects_unhandled_reload_backends() -> None:
    # Optional native dependencies are installed only in the vLLM test environment.
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE

    from nemo_rl.modelopt.models.generation import (
        vllm_quant_patch as patch_mod,
    )

    supported = {
        NvFp4MoeBackend.VLLM_CUTLASS,
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
        NvFp4MoeBackend.FLASHINFER_CUTLASS,
        NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED,
        NvFp4MoeBackend.MARLIN,
    }
    for backend in set(NvFp4MoeBackend) - supported:
        layer = torch.nn.Module()
        method = object.__new__(ModelOptNvFp4FusedMoE)
        method.nvfp4_backend = backend
        layer.quant_method = method
        with pytest.raises(ValueError, match=backend.name):
            patch_mod._preserve_moe_reload_scales(layer)


@pytest.mark.vllm
@pytest.mark.parametrize("backend", ["FLASHINFER_CUTLASS", "HPC"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_quant_fp8_moe_scales_survive_native_reload(
    backend: str, device: str
) -> None:
    # Optional native dependencies are installed only in the vLLM test environment.
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend,
        make_fp8_moe_quant_config,
    )
    from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8MoEMethod
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_reload,
        get_layerwise_info,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    from nemo_rl.modelopt.models.generation import (
        vllm_quant_patch as patch_mod,
    )

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA graph scale-storage regression requires a GPU")

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones(2, device=device), requires_grad=False)
    method = object.__new__(ModelOptFp8MoEMethod)
    method.fp8_backend = Fp8MoeBackend[backend]
    layer.quant_method = method

    def process_weights(target: torch.nn.Module) -> None:
        target.weight = torch.nn.Parameter(target.weight.clone(), requires_grad=False)
        method.moe_quant_config = make_fp8_moe_quant_config(
            fp8_backend=method.fp8_backend,
            w1_scale=target.weight,
            w2_scale=target.weight,
            a1_scale=target.weight,
            a2_scale=target.weight * 2,
            layer=target,
        )
        method.moe_kernel = object()

    method.process_weights_after_loading = process_weights
    with torch.device(device):
        record_metadata_for_reloading(layer)
    process_weights(layer)
    patch_mod._preserve_moe_reload_scales(layer)
    config, kernel = method.moe_quant_config, method.moe_kernel
    assert layer._nrl_a1_gscale is config.a1_gscale
    assert layer._nrl_a2_gscale is config.a2_gscale
    assert not get_layerwise_info(layer).restore_metadata[1]

    def consume_scales() -> torch.Tensor:
        return config.a1_gscale + config.a2_gscale + config.g1_alphas + config.g2_alphas

    if device == "cuda":
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = consume_scales()

    for value in (2.0, 3.0, 1.0):
        initialize_layerwise_reload(layer)
        assert not layer._buffers
        assert get_layerwise_info(layer).load_numel_total == 2
        layer.weight.weight_loader(layer.weight, torch.full((2,), value, device=device))
        finalize_layerwise_reload(layer, None)
        assert method.moe_quant_config is config and method.moe_kernel is kernel
        assert layer._nrl_a1_gscale is config.a1_gscale
        assert layer._nrl_a2_gscale is config.a2_gscale
        assert config.w1_scale is layer.weight
        assert config.g1_alphas is layer.g1_alphas
        assert config.g2_alphas is layer.g2_alphas
        torch.testing.assert_close(
            config.a1_gscale, torch.full((2,), 1 / value, device=device)
        )
        torch.testing.assert_close(
            config.a2_gscale, torch.full((2,), 1 / (2 * value), device=device)
        )
        eager = consume_scales()
        torch.testing.assert_close(
            eager,
            torch.full((2,), 1 / value + 1 / (2 * value) + 3 * value**2, device=device),
        )
        if device == "cuda":
            graph.replay()
            torch.testing.assert_close(captured, eager)


@pytest.mark.vllm
def test_real_quant_other_fp8_moe_backends_keep_native_processing() -> None:
    # Optional native dependencies are installed only in the vLLM test environment.
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
    from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8MoEMethod

    from nemo_rl.modelopt.models.generation import (
        vllm_quant_patch as patch_mod,
    )

    for backend in set(Fp8MoeBackend) - {
        Fp8MoeBackend.FLASHINFER_CUTLASS,
        Fp8MoeBackend.HPC,
    }:
        layer = torch.nn.Module()
        method = object.__new__(ModelOptFp8MoEMethod)
        method.fp8_backend = backend
        method.process_weights_after_loading = object()
        layer.quant_method = method
        original_process = method.process_weights_after_loading

        patch_mod._preserve_moe_reload_scales(layer)

        assert method.process_weights_after_loading is original_process
        assert not layer._buffers


@pytest.mark.vllm
def test_real_quant_worker_loads_model_outside_inference_mode(monkeypatch):
    # Optional native dependencies are installed only in the vLLM test environment.
    from nemo_rl.modelopt.models.generation import (
        vllm_quant_patch as patch_mod,
    )
    from nemo_rl.models.generation.vllm.vllm_backend import NixlVllmWorker

    worker = object.__new__(patch_mod.RealQuantWorker)
    worker.model_runner = types.SimpleNamespace(model=None)
    inference_modes = []

    def fake_load_model(self, **kwargs):
        inference_modes.append(torch.is_inference_mode_enabled())
        self.model_runner.model = torch.nn.Linear(1, 1)

    monkeypatch.setattr(
        NixlVllmWorker,
        "load_model",
        fake_load_model,
    )

    worker.load_model()

    assert inference_modes == [False]
    weight = worker.model_runner.model.weight
    assert not torch.is_inference(weight)
    with torch.no_grad():
        weight.copy_(torch.ones_like(weight))


def test_configure_quant_engine_kwargs_for_real_quant(monkeypatch):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)
    quantization_config = {"quant_method": "modelopt", "quant_algo": "FP8"}
    llm_kwargs = {
        "moe_backend": "flashinfer_trtllm",
        "kv_cache_dtype": "auto",
        "hf_overrides": {
            "trust_remote_code": True,
            "quantization_config": quantization_config,
        },
    }
    worker_mod._configure_quant_engine_kwargs(
        {"quant_cfg": None, "real_quant": True},
        llm_kwargs,
    )

    assert "VLLM_QUANT_CFG" not in os.environ
    assert llm_kwargs["moe_backend"] == "flashinfer_trtllm"
    assert llm_kwargs["worker_cls"] == (
        "nemo_rl.modelopt.models.generation.vllm_quant_patch.RealQuantWorker"
    )
    assert "quantization" not in llm_kwargs
    assert llm_kwargs["kv_cache_dtype"] == "auto"
    assert llm_kwargs["hf_overrides"] == {
        "trust_remote_code": True,
        "quantization_config": quantization_config,
    }
    assert llm_kwargs["worker_extension_cls"] == (
        "nemo_rl.modelopt.models.generation.vllm_quant_backend."
        "VllmQuantInternalWorkerExtension"
    )


@pytest.mark.parametrize(
    "cfg",
    [
        {
            "quant_cfg": None,
            "real_quant": True,
            "refit_transport": "nixl",
            "refit_cfg": {"nixl": {}},
        },
        {
            "quant_cfg": None,
            "real_quant": True,
            "refit_transport": "vllm_zmq_sparse",
        },
    ],
)
def test_configure_real_quant_rejects_transport_without_reload_lifecycle(cfg):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    llm_kwargs = {
        "hf_overrides": {
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}
        }
    }

    with pytest.raises(ValueError, match="native layerwise reload lifecycle"):
        worker_mod._configure_quant_engine_kwargs(cfg, llm_kwargs)


def test_configure_real_quant_requires_policy_config():
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )

    with pytest.raises(ValueError, match="policy-produced"):
        worker_mod._configure_quant_engine_kwargs(
            {"quant_cfg": None, "real_quant": True},
            {},
        )


def test_prepare_real_quant_generation_config_copies_policy_config():
    quantization_config = {"quant_method": "modelopt", "quant_algo": "FP8"}
    policy = types.SimpleNamespace(
        cfg={"quant_cfg": "FP8_DEFAULT_CFG", "megatron_cfg": {"enabled": True}},
        get_real_quantization_config=lambda: quantization_config,
    )
    generation_config = {"vllm_kwargs": {"hf_overrides": {"trust_remote_code": True}}}

    modelopt_utils.prepare_real_quant_generation_config(policy, generation_config)
    quantization_config["quant_algo"] = "modified"

    assert generation_config["vllm_kwargs"]["hf_overrides"] == {
        "trust_remote_code": True,
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"},
    }


def test_prepare_real_quant_generation_config_rejects_conflict():
    policy = types.SimpleNamespace(
        cfg={"quant_cfg": "FP8_DEFAULT_CFG", "megatron_cfg": {"enabled": True}},
        get_real_quantization_config=lambda: {
            "quant_method": "modelopt",
            "quant_algo": "FP8",
        },
    )
    generation_config = {
        "vllm_kwargs": {
            "hf_overrides": {
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                }
            }
        }
    }

    with pytest.raises(ValueError, match="conflicts"):
        modelopt_utils.prepare_real_quant_generation_config(policy, generation_config)


@pytest.mark.parametrize(
    ("policy_config", "error"),
    [
        (
            {"quant_cfg": "FP8_DEFAULT_CFG", "megatron_cfg": {"enabled": False}},
            "megatron_cfg",
        ),
        ({"quant_cfg": None, "megatron_cfg": {"enabled": True}}, "policy.quant_cfg"),
    ],
)
def test_prepare_real_quant_generation_config_validates_policy(policy_config, error):
    policy = types.SimpleNamespace(
        cfg=policy_config,
        get_real_quantization_config=lambda: pytest.fail("unexpected worker RPC"),
    )

    with pytest.raises(ValueError, match=error):
        modelopt_utils.prepare_real_quant_generation_config(policy, {})


@pytest.mark.parametrize("llm_kwargs", [{}, {"moe_backend": "flashinfer_trtllm"}])
def test_configure_quant_engine_kwargs_for_fake_quant_without_quant_cfg(
    monkeypatch, llm_kwargs
):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)

    llm_kwargs = dict(llm_kwargs)
    expected_backend = llm_kwargs.get("moe_backend", "triton")
    worker_mod._configure_quant_engine_kwargs({"quant_cfg": None}, llm_kwargs)

    assert llm_kwargs["moe_backend"] == expected_backend
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
            modelopt_utils.MODELOPT_REAL_QUANT_REFIT_TIMEOUT_MS,
        ),
        (
            "setsockopt",
            backend.zmq.RCVTIMEO,
            modelopt_utils.MODELOPT_REAL_QUANT_REFIT_TIMEOUT_MS,
        ),
    ]


@pytest.mark.parametrize("real_quant", [False, True])
def test_modelopt_refit_info_routes_legacy_fp8_config(
    monkeypatch: pytest.MonkeyPatch, real_quant: bool
) -> None:
    backend = _import_vllm_quant_backend(monkeypatch)
    base_backend = _base_vllm_backend()
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension._is_real_quant_model = lambda: real_quant
    state_dict_info: dict[str, object] = {"decoder.weight": ([1], torch.float32)}
    serialized_fp8_config: dict[str, object] = {"is_mx": True}
    expected_names = ["decoder.weight"]
    calls = []

    def prepare_refit_info(
        _self: object,
        info: dict[str, object],
        fp8_config: dict[str, object] | None = None,
    ) -> list[str] | None:
        calls.append((info, fp8_config))
        return None if fp8_config is None else expected_names

    monkeypatch.setattr(
        base_backend.VllmInternalWorkerExtension,
        "prepare_refit_info",
        prepare_refit_info,
    )

    result = extension.prepare_refit_info(state_dict_info, serialized_fp8_config)

    assert calls == [(state_dict_info, None if real_quant else serialized_fp8_config)]
    assert result is (None if real_quant else expected_names)


@pytest.mark.parametrize("real_quant", [False, True])
def test_modelopt_refit_info_preserves_base_validation_errors(
    monkeypatch: pytest.MonkeyPatch, real_quant: bool
) -> None:
    backend = _import_vllm_quant_backend(monkeypatch)
    base_backend = _base_vllm_backend()
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension._is_real_quant_model = lambda: real_quant

    def reject_refit_info(
        _self: object,
        _info: dict[str, object],
        _fp8_config: dict[str, object] | None = None,
    ) -> None:
        raise ValueError("invalid refit metadata")

    monkeypatch.setattr(
        base_backend.VllmInternalWorkerExtension,
        "prepare_refit_info",
        reject_refit_info,
    )

    with pytest.raises(ValueError, match="invalid refit metadata"):
        extension.prepare_refit_info({}, {"is_mx": True})


@pytest.mark.parametrize(
    ("quantization", "expected"),
    [
        ("modelopt", True),
        ("modelopt_fp4", True),
        ("modelopt_mxfp8", True),
        ("modelopt_mixed", True),
        ("fp8", False),
        (None, False),
    ],
)
def test_real_quant_model_detection_uses_native_vllm_method(
    monkeypatch, quantization, expected
):
    backend = _import_vllm_quant_backend(monkeypatch)
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(quantization=quantization)
        )
    )

    assert extension._is_real_quant_model() is expected


def test_real_quant_load_uses_canonical_hf_loader(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.device = torch.device("cuda:0")
    extension.model_runner = types.SimpleNamespace(
        vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(quantization="modelopt")
        )
    )
    source = torch.tensor([1.0, 2.0])
    loaded = []

    def load_canonical_weights(
        _self: object, weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        assert torch.get_default_device().type == "cpu"
        loaded.extend(weights)

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_load_full_hf_weights",
        load_canonical_weights,
    )

    extension._load_weights([("model.weight", source)])

    assert [name for name, _ in loaded] == ["model.weight"]
    torch.testing.assert_close(loaded[0][1], source)
    assert (
        loaded[0][1].untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    )


def test_fake_quant_load_weights_exposes_activation_quantizer_buffers(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    child = torch.nn.Module()
    child.weight = torch.nn.Parameter(torch.ones(1))
    child.input_quantizer = torch.nn.Module()
    child.input_quantizer.register_buffer("_amax", torch.tensor([1.0]))
    child.register_buffer("weight_quantizer_amax", torch.tensor([2.0]))
    child.self_attn = torch.nn.Module()
    child.self_attn.attn = torch.nn.Module()
    child.self_attn.attn.k_bmm_quantizer = torch.nn.Module()
    child.self_attn.attn.k_bmm_quantizer.register_buffer("_amax", torch.tensor([-1.0]))
    model = torch.nn.Module()
    model.child = child
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(model=model)
    seen_names = []

    def fake_base_load_weights(self, weights):
        assert [name for name, _ in weights] == [
            "child.self_attn.attn.k_bmm_quantizer._amax"
        ]
        params = dict(child.named_parameters())
        seen_names.extend(params)
        params["input_quantizer._amax"].weight_loader(
            params["input_quantizer._amax"],
            torch.tensor([3.0]),
        )
        params["self_attn.attn.k_bmm_quantizer._amax"].weight_loader(
            params["self_attn.attn.k_bmm_quantizer._amax"],
            torch.tensor([4.0]),
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

    assert (
        extension._load_weights(
            [("child.self_attn.k_bmm_quantizer._amax", torch.tensor([4.0]))]
        )
        == "loaded"
    )

    assert "weight" in seen_names
    assert "input_quantizer._amax" in seen_names
    assert "self_attn.attn.k_bmm_quantizer._amax" in seen_names
    assert "weight_quantizer_amax" not in seen_names
    assert not hasattr(child.input_quantizer._amax, "weight_loader")
    assert not hasattr(child.self_attn.attn.k_bmm_quantizer._amax, "weight_loader")
    torch.testing.assert_close(child.input_quantizer._amax, torch.tensor([3.0]))
    torch.testing.assert_close(
        child.self_attn.attn.k_bmm_quantizer._amax,
        torch.tensor([4.0]),
    )


def test_fake_quant_eager_input_amax_loader_supports_direct_vllm_load(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    model = torch.nn.Module()
    model.input_quantizer = torch.nn.Module()
    model.input_quantizer.register_buffer("_amax", torch.tensor([1.0]))
    model.k_bmm_quantizer = torch.nn.Module()
    model.k_bmm_quantizer.register_buffer("_amax", torch.tensor([2.0]))
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)

    with extension._attach_input_quantizer_amax_loaders(model):
        input_amax = model.input_quantizer._amax
        assert hasattr(input_amax, "weight_loader")
        assert not hasattr(model.k_bmm_quantizer._amax, "weight_loader")

        input_amax.weight_loader(input_amax, torch.tensor([3.0]))
        input_amax.weight_loader(input_amax, torch.tensor([2.0]))
        torch.testing.assert_close(input_amax, torch.tensor([3.0]))

    assert not hasattr(model.input_quantizer._amax, "weight_loader")


@pytest.mark.parametrize(
    ("transport", "expected"),
    [("ipc", True), ("collective", True), ("nccl_reshard", False)],
)
def test_real_quant_selects_native_layerwise_refit(monkeypatch, transport, expected):
    backend = _import_vllm_quant_backend(monkeypatch)
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda _self: True,
    )

    assert extension._uses_native_layerwise_refit(transport) is expected


@pytest.mark.parametrize(
    "fail_transfer", [False, True], ids=["success", "transfer-failure"]
)
def test_real_quant_collective_reload_uses_vllm_layerwise_lifecycle(
    monkeypatch, fail_transfer
):
    backend = _import_vllm_quant_backend(monkeypatch)
    base_backend = _base_vllm_backend()
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]

    model = torch.nn.Linear(1, 1)
    model_config = object()
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=object(),
    )
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

    def consume(**kwargs):
        if fail_transfer:
            raise ValueError("broadcast boom")
        calls.append(("consume", kwargs["post_unpack_func"].__name__))

    monkeypatch.setattr(base_backend, "packed_broadcast_consumer", consume)
    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        lambda model_arg, config_arg: calls.append(("finalize", model_arg, config_arg)),
    )
    monkeypatch.setattr(
        base_backend.torch.cuda,
        "synchronize",
        lambda: calls.append("sync"),
    )

    if fail_transfer:
        with pytest.raises(ValueError, match="broadcast boom"):
            extension.update_weights_from_collective()
        assert calls == [("initialize", model)]
        assert isinstance(extension._nrl_layerwise_reload_failure, ValueError)
    else:
        assert extension.update_weights_from_collective() is True
        assert calls == [
            ("initialize", model),
            ("consume", "_load_weights"),
            ("finalize", model, model_config),
            "sync",
        ]


@pytest.mark.parametrize(
    "method", ["update_weights_from_collective", "update_weights_via_ipc_zmq"]
)
def test_non_real_quant_reload_delegates(monkeypatch, method):
    backend = _import_vllm_quant_backend(monkeypatch)

    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: False,
    )
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        method,
        lambda self: "delegated",
    )

    assert getattr(extension, method)() == "delegated"


@pytest.mark.parametrize(
    "fail_finalize", [False, True], ids=["success", "finalize-failure"]
)
def test_real_quant_ipc_complete_finalizes_vllm_layerwise_reload_and_acks(
    monkeypatch, fail_finalize
):
    backend = _import_vllm_quant_backend(monkeypatch)
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]
    from nemo_rl.models.policy.utils import IPCProtocol

    model = torch.nn.Linear(1, 1)
    model_config = object()
    socket = types.SimpleNamespace(recv_pyobj=lambda: IPCProtocol.COMPLETE, sent=[])
    socket.send = socket.sent.append
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                world_size_across_dp=1, enable_eplb=False
            )
        ),
    )
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

    def finalize(model_arg, config_arg):
        if fail_finalize:
            raise RuntimeError("bad scales")
        calls.append(("finalize", model_arg, config_arg))

    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        finalize,
    )
    monkeypatch.setattr(
        _base_vllm_backend().torch.cuda,
        "synchronize",
        lambda: calls.append("sync"),
    )
    monkeypatch.setattr(
        backend.torch.cuda, "empty_cache", lambda: calls.append("empty")
    )

    if fail_finalize:
        with pytest.raises(RuntimeError, match="bad scales"):
            extension.update_weights_via_ipc_zmq()
        assert calls == [("initialize", model)]
        assert isinstance(extension._nrl_layerwise_reload_failure, RuntimeError)
    else:
        assert extension.update_weights_via_ipc_zmq() is True
        assert calls == [
            ("initialize", model),
            ("finalize", model, model_config),
            "sync",
            "empty",
        ]
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
    extension.model_runner = types.SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        vllm_config=types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                world_size_across_dp=1, enable_eplb=False
            )
        ),
    )
    extension.model_config = object()
    extension.device = torch.device("cuda:0")
    extension.zmq_socket = FakeSocket()
    extension.state_dict_info = state_dict_info
    extension.maybe_init_zmq = lambda: None
    extension._load_weights = lambda _weights: None
    extension.model_runner.model.load_weights = extension._load_weights
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


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "empty",
        "prepare",
        "pre_fence",
        "ack",
        "load",
        "load_fence",
        "load_and_fence",
    ],
)
@pytest.mark.parametrize("load_after_ack", [False, True])
def test_real_quant_ipc_payload_loads_weights_and_releases_transport_views(
    monkeypatch, caplog, failure, load_after_ack: bool
):
    backend = _import_vllm_quant_backend(monkeypatch)
    base_backend = _base_vllm_backend()
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]
    from nemo_rl.models.policy.utils import IPCProtocol

    empty = failure == "empty"
    success = failure in (None, "empty")
    payload_weight = torch.tensor([1.0, 2.0], dtype=torch.float32)
    allocation_bytes = base_backend.calculate_aligned_size(payload_weight.nbytes)
    used_bytes = 0 if empty else allocation_bytes
    payload_buffer = torch.zeros(allocation_bytes * 2, dtype=torch.uint8)
    loaded = []
    retained = []
    calls = []
    ipc_refs = []
    owned_storage_refs = []

    class FakeSocket:
        def __init__(self):
            self.payloads = iter(
                [
                    ("ipc-handle", [] if empty else ["decoder.weight"], used_bytes),
                    ("ipc-handle", [] if empty else ["decoder.bias"], used_bytes),
                    IPCProtocol.COMPLETE,
                ]
            )
            self.sent = []
            self.batch_index = -1
            self.complete = False

        def recv_pyobj(self):
            payload = next(self.payloads)
            self.complete = payload == IPCProtocol.COMPLETE
            if not self.complete:
                self.batch_index += 1
            return payload

        def send(self, payload):
            if failure != "prepare":
                assert all(ref() is None for ref in ipc_refs)
            if not self.complete:
                calls.append(("ack", self.batch_index))
                if failure == "ack" and self.batch_index == 0:
                    raise RuntimeError("ack failed")
            self.sent.append(payload)
            if not self.complete:
                # Native loading must see the owned copy, never reused IPC data.
                payload_buffer.fill_(0xA5)

    model = torch.nn.Linear(1, 1)
    model_config = object()
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                world_size_across_dp=1 if load_after_ack else 2,
                enable_eplb=False,
            )
        ),
    )
    extension.model_config = model_config
    extension.device = torch.device("cuda:0")
    extension.zmq_socket = FakeSocket()
    extension.state_dict_info = (
        {}
        if empty
        else {
            "decoder.weight": ([2], torch.float32),
            "decoder.bias": ([2], torch.float32),
        }
    )
    extension.maybe_init_zmq = lambda: None

    def load_weights(weights):
        calls.append(("load", extension.zmq_socket.batch_index))
        retained.append(weights)
        for name, weight in weights:
            torch.testing.assert_close(weight, payload_weight)
            loaded.append((name, weight.clone()))
        if failure in ("load", "load_and_fence"):
            raise RuntimeError("load failed")

    model.load_weights = load_weights
    extension._load_weights = lambda _: pytest.fail("owned IPC must load directly")
    original_prepare = extension._prepare_ipc_weight_buffer

    def prepare_buffer(buffer, size):
        if failure == "prepare":
            raise RuntimeError("prepare failed")
        owned, loader = original_prepare(buffer, size)
        # dtype views retain storage, not necessarily this Python Tensor wrapper.
        owned_storage_refs.append(StorageWeakRef(owned.untyped_storage()))
        return owned, loader

    extension._prepare_ipc_weight_buffer = prepare_buffer

    def rebuild_buffer(ipc_handle, device_index):
        payload_buffer[: payload_weight.nbytes].view(torch.float32).copy_(
            payload_weight
        )
        buffer = payload_buffer[:]
        ipc_refs.append(weakref.ref(buffer))
        return buffer

    def synchronize():
        index = extension.zmq_socket.batch_index
        after_ack = len(extension.zmq_socket.sent) > index
        calls.append(("fence", index, after_ack))
        if after_ack:
            assert all(ref() is None for ref in ipc_refs)
            if not empty:
                assert not owned_storage_refs[index].expired()
                assert owned_storage_refs[index] == StorageWeakRef(
                    retained[-1][0][1].untyped_storage()
                )
        else:
            assert ipc_refs[-1]() is not None
        if index == 0 and (
            (failure == "pre_fence" and not after_ack)
            or (
                failure in ("load_fence", "load_and_fence") and ("load", index) in calls
            )
        ):
            raise RuntimeError("fence failed")

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
    monkeypatch.setattr(base_backend, "rebuild_cuda_tensor_from_ipc", rebuild_buffer)
    monkeypatch.setattr(
        base_backend.torch.cuda,
        "current_stream",
        lambda: pytest.fail("real quant must use the device-wide IPC fence"),
    )
    monkeypatch.setattr(backend.torch.accelerator, "synchronize", synchronize)
    monkeypatch.setattr(
        base_backend.torch.cuda, "synchronize", lambda: calls.append("sync")
    )
    monkeypatch.setattr(
        backend.torch.cuda, "empty_cache", lambda: calls.append("empty")
    )
    monkeypatch.setattr(base_backend.gc, "collect", lambda: calls.append("gc"))

    error = {
        "pre_fence": "fence failed",
        "load_fence": "fence failed",
        "load_and_fence": "load failed",
    }.get(failure, f"{failure} failed")
    if success:
        assert extension.update_weights_via_ipc_zmq() is True
    else:
        with pytest.raises(RuntimeError, match=error):
            extension.update_weights_via_ipc_zmq()

    if success:
        expected_loads = 2
    elif failure == "prepare" or (load_after_ack and failure in ("pre_fence", "ack")):
        expected_loads = 0
    else:
        expected_loads = 1
    assert (
        sum(isinstance(call, tuple) and call[0] == "load" for call in calls)
        == expected_loads
    )
    assert extension.zmq_socket.sent == [IPCProtocol.ACK.value.encode()] * (
        0 if failure == "ack" else 3
    )
    if success:
        assert [[name for name, _ in batch] for batch in retained] == (
            [[], []] if empty else [["decoder.weight"], ["decoder.bias"]]
        )
        for index in range(2):
            ack = calls.index(("ack", index))
            load = calls.index(("load", index))
            assert (ack < load) == load_after_ack
            assert calls.index(("fence", index, False)) < ack
            if load_after_ack:
                assert load < calls.index(("fence", index, True))
        assert ("finalize", model, model_config) in calls
    else:
        assert not any(
            isinstance(call, tuple) and call[0] == "finalize" for call in calls
        )
        assert isinstance(extension._nrl_layerwise_reload_failure, RuntimeError)
        if failure == "load_and_fence":
            assert "fence failed" not in str(extension._nrl_layerwise_reload_failure)
        if load_after_ack and failure in ("load", "load_fence", "load_and_fence"):
            assert retained == [[]]
        if load_after_ack and failure == "ack":
            assert all(ref.expired() for ref in owned_storage_refs)
        with pytest.raises(RuntimeError, match="unusable after a failed"):
            extension.update_weights_via_ipc_zmq()
    if failure in ("load", "load_and_fence"):
        assert "IPC weight batch load failed" in caplog.text
        assert "decoder.weight" in caplog.text


@pytest.mark.parametrize(
    ("real_quant", "parallel_kwargs", "expected"),
    [
        (False, {}, False),
        (True, {"world_size_across_dp": 1, "enable_eplb": False}, True),
        (
            True,
            {
                "world_size_across_dp": 1,
                "enable_eplb": False,
                "enable_expert_parallel": True,
            },
            True,
        ),
        (
            True,
            {
                "world_size_across_dp": 2,
                "enable_eplb": False,
                "world_size": 1,
                "tensor_parallel_size": 1,
                "data_parallel_size": 2,
            },
            False,
        ),
        (
            True,
            {
                "world_size_across_dp": 2,
                "enable_eplb": False,
                "tensor_parallel_size": 1,
                "prefill_context_parallel_size": 2,
            },
            False,
        ),
        (
            True,
            {
                "world_size_across_dp": 2,
                "enable_eplb": False,
                "tensor_parallel_size": 2,
            },
            False,
        ),
        (
            True,
            {
                "world_size_across_dp": 2,
                "enable_eplb": False,
                "pipeline_parallel_size": 2,
            },
            False,
        ),
        (True, {"world_size_across_dp": 1, "enable_eplb": True}, False),
    ],
    ids=["fake", "singleton", "singleton-ep-flag", "dp", "pcp", "tp", "pp", "eplb"],
)
def test_ipc_post_ack_loading_requires_resolved_singleton_without_eplb(
    monkeypatch: pytest.MonkeyPatch,
    real_quant: bool,
    parallel_kwargs: dict[str, int | bool],
    expected: bool,
) -> None:
    backend = _import_vllm_quant_backend(monkeypatch)
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension._is_real_quant_model = lambda: real_quant
    extension.model_runner = types.SimpleNamespace()
    if real_quant:
        extension.model_runner.vllm_config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(**parallel_kwargs)
        )
    assert extension._can_load_ipc_weights_after_ack() is expected


def test_ipc_post_ack_loading_does_not_assume_missing_world_size_is_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _import_vllm_quant_backend(monkeypatch)
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension._is_real_quant_model = lambda: True
    extension.model_runner = types.SimpleNamespace(
        vllm_config=types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(enable_eplb=False)
        )
    )
    with pytest.raises(AttributeError, match="world_size_across_dp"):
        extension._can_load_ipc_weights_after_ack()


@pytest.mark.parametrize("real_quant", [False, True])
def test_ipc_packet_ownership_preserves_deferred_views(monkeypatch, real_quant):
    backend = _import_vllm_quant_backend(monkeypatch)
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    pending = []
    loader = lambda weights: pending.extend(weights)
    extension.model_runner = types.SimpleNamespace(
        model=types.SimpleNamespace(load_weights=loader)
    )
    extension._load_weights = loader
    extension._is_real_quant_model = lambda: real_quant
    source = torch.arange(4096, dtype=torch.int32).view(torch.uint8)[512:]
    used_bytes = 1024
    clones = []
    original_clone = torch.Tensor.clone

    def clone(tensor, *args, **kwargs):
        clones.append(tensor.numel())
        return original_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", clone)
    buffer, load_weights = extension._prepare_ipc_weight_buffer(source, used_bytes)
    assert buffer.numel() == (used_bytes if real_quant else source.numel())
    assert clones == ([used_bytes] if real_quant else [])
    assert (buffer is source) is (not real_quant)
    load_weights(
        [
            ("first", buffer[:8].view(torch.int32)),
            ("second", buffer[512:520].view(torch.int32)),
        ]
    )
    assert (
        pending[0][1].untyped_storage().data_ptr()
        == pending[1][1].untyped_storage().data_ptr()
    )
    assert (
        pending[0][1].untyped_storage().data_ptr()
        == source.untyped_storage().data_ptr()
    ) is (not real_quant)
    source.zero_()
    del buffer
    torch.testing.assert_close(
        pending[0][1],
        torch.tensor([128, 129] if real_quant else [0, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        pending[1][1],
        torch.tensor([256, 257] if real_quant else [0, 0], dtype=torch.int32),
    )


def test_weight_snapshot_returns_cpu_clone_and_missing_name_raises(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    model = torch.nn.Module()
    model.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.tensor([[1.0, -1.0]], dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(model=model)

    snapshot = extension.get_weight_snapshot("weight")
    model.weight.data.zero_()

    assert snapshot.device.type == "cpu"
    assert snapshot.dtype == torch.float32
    assert not torch.equal(
        snapshot, model.weight.detach().to(device="cpu", dtype=torch.float32)
    )
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
        "kv_amax": {},
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
