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
        pass

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


def test_configure_quant_engine_kwargs_for_fake_quant(monkeypatch, tmp_path):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)

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


def test_fake_quant_worker_inherits_nixl_worker():
    patch_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_patch"
    )
    from nemo_rl.models.generation.vllm.vllm_backend import NixlVllmWorker

    assert issubclass(patch_mod.FakeQuantWorker, NixlVllmWorker)


def test_configure_quant_engine_kwargs_for_real_quant(monkeypatch):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)
    quantization_config = {"quant_method": "modelopt", "quant_algo": "FP8"}
    llm_kwargs = {
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
    assert "worker_cls" not in llm_kwargs
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


def test_configure_quant_engine_kwargs_for_fake_quant_without_quant_cfg(monkeypatch):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)

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
    extension.device = torch.device("cpu")
    extension.model_runner = types.SimpleNamespace(
        vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(quantization="modelopt")
        )
    )
    source = torch.tensor([1.0, 2.0])
    loaded = []

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_load_full_hf_weights",
        lambda _self, weights: loaded.extend(weights),
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


def test_real_quant_collective_reload_uses_vllm_layerwise_lifecycle(monkeypatch):
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
        base_backend.torch.cuda,
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

    model = torch.nn.Linear(1, 1)
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=object(),
    )
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

    with pytest.raises(ValueError, match="broadcast boom"):
        extension.update_weights_from_collective()
    assert calls == [("initialize", model)]
    assert isinstance(extension._nrl_layerwise_reload_failure, ValueError)


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

    model = torch.nn.Linear(1, 1)
    model_config = object()
    socket = FakeSocket()
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=object(),
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
    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        lambda model_arg, config_arg: calls.append(("finalize", model_arg, config_arg)),
    )
    monkeypatch.setattr(
        _base_vllm_backend().torch.cuda,
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
        model=torch.nn.Linear(1, 1),
        vllm_config=object(),
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

    with pytest.raises(RuntimeError, match="bad scales"):
        extension.update_weights_via_ipc_zmq()
    assert socket.sent == [IPCProtocol.ACK.value.encode()]
    assert isinstance(extension._nrl_layerwise_reload_failure, RuntimeError)


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
        vllm_config=object(),
    )
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


def test_real_quant_ipc_payload_loads_weights_and_releases_transport_views(monkeypatch):
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

    model = torch.nn.Linear(1, 1)
    model_config = object()
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=object(),
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
        base_backend.torch.cuda,
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
