# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real Flint refit/launch helpers without Ray or backend imports."""

import ast
import logging
import multiprocessing
import os
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Generator, NotRequired, Optional

import pytest
import torch
from pydantic import TypeAdapter, ValidationError
from torch import nn
from torch.distributed.tensor import DTensor
from typing_extensions import TypedDict


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def backend_config_adapter():
    """Load the production YAML-facing schema without importing optional backends."""
    source = REPO_ROOT / "nemo_rl/models/policy/__init__.py"
    tree = ast.parse(source.read_text())
    schema = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AutomodelBackendConfig"
    )
    # typing_extensions permits this schema test on Python 3.12 as well as the
    # container's Python 3.13, where production uses typing.TypedDict directly.
    namespace = {"TypedDict": TypedDict, "NotRequired": NotRequired}
    exec(
        compile(ast.Module(body=[schema], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return TypeAdapter(namespace[schema.name])


@pytest.mark.parametrize("precision", [None, "float32", "torch.float64"])
def test_backend_schema_accepts_native_optional_gate_precision(
    backend_config_adapter, precision
):
    config = {
        "_target_": "nemo_automodel.components.models.common.BackendConfig",
        "gate_precision": precision,
    }
    assert backend_config_adapter.validate_python(config)["gate_precision"] == precision


def test_backend_schema_rejects_numeric_gate_precision(backend_config_adapter):
    with pytest.raises(ValidationError):
        backend_config_adapter.validate_python(
            {"_target_": "BackendConfig", "gate_precision": 123}
        )


def _load_functions(relative_path, names, namespace, class_name=None):
    """Compile production helpers alone so CPU tests need no training runtime."""
    source = REPO_ROOT / relative_path
    tree = ast.parse(source.read_text())
    body = tree.body
    if class_name is not None:
        body = next(
            node for node in body if getattr(node, "name", None) == class_name
        ).body
    functions = [
        node
        for node in body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == set(names)
    for node in functions:
        node.decorator_list = []
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace


@pytest.fixture
def refit_helpers():
    namespace = {
        "torch": torch,
        "nn": nn,
        "DTensor": DTensor,
        "LinearLoRA": type("UnusedLoRA", (), {}),
        "Any": Any,
        "Generator": Generator,
        "Optional": Optional,
    }
    path = "nemo_rl/models/policy/workers/automodel_policy_worker.py"
    _load_functions(
        path,
        {
            "_refit_tensor_dtype",
            "_maybe_merge_lora_weight",
            "_maybe_adapt_tensor_to_hf",
            "_module_params_generator",
            "automodel_params_generator",
        },
        namespace,
    )
    return _load_functions(
        path, {"prepare_refit_info"}, namespace, "AutomodelPolicyWorkerImpl"
    )


class _RenamingAdapter:
    def convert_single_tensor_to_hf(self, fqn, tensor, **kwargs):
        return [(fqn.replace("._fp32_params.", "."), tensor)]


class _MixedPrecisionModel(nn.Module):
    """Small native model whose sink's HF name loses its FP32 module marker."""

    _keep_in_fp32_modules = ["lm_head"]
    _keep_in_fp32_modules_strict = ["_fp32_params"]

    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(2, 2, bias=False, dtype=torch.float32)
        self.self_attn = nn.Module()
        self.self_attn._fp32_params = nn.Module()
        self.self_attn._fp32_params.register_parameter(
            "softmax_offset", nn.Parameter(torch.tensor([0.1234567, 0.9876543]))
        )
        self.bulk = nn.Linear(2, 2, bias=False, dtype=torch.float32)
        self.register_buffer("token_ids", torch.tensor([65537, 150143]))
        self.register_buffer("e_score_correction_bias", torch.tensor([0.1234567]))
        self.state_dict_adapter = _RenamingAdapter()
        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.tensor([[0.1234567, 0.9876543], [0.2345678, 0.8765432]])
            )


def test_refit_preserves_declared_fp32_values_after_adapter_rename(refit_helpers):
    model = _MixedPrecisionModel()
    payload = dict(refit_helpers["automodel_params_generator"](model, torch.bfloat16))

    assert payload["lm_head.weight"].dtype == torch.float32
    assert torch.equal(payload["lm_head.weight"], model.lm_head.weight)
    assert payload["self_attn.softmax_offset"].dtype == torch.float32
    assert torch.equal(
        payload["self_attn.softmax_offset"], model.self_attn._fp32_params.softmax_offset
    )
    assert payload["bulk.weight"].dtype == torch.bfloat16
    assert payload["e_score_correction_bias"].dtype == torch.float32
    assert torch.equal(payload["token_ids"], model.token_ids)


def test_refit_metadata_matches_payload_for_every_tensor(refit_helpers):
    model = _MixedPrecisionModel()
    worker = SimpleNamespace(model=model, dtype=torch.bfloat16, draft_model=None)
    metadata = refit_helpers["prepare_refit_info"](worker)
    payload = dict(refit_helpers["automodel_params_generator"](model, torch.bfloat16))

    assert set(metadata) == set(payload)
    for name, tensor in payload.items():
        assert metadata[name] == (tensor.shape, tensor.dtype)


def test_undeclared_fp32_head_keeps_existing_bf16_refit_behavior(refit_helpers):
    model = _MixedPrecisionModel()
    model._keep_in_fp32_modules = None
    model._keep_in_fp32_modules_strict = None
    payload = dict(refit_helpers["automodel_params_generator"](model, torch.bfloat16))

    assert payload["lm_head.weight"].dtype == torch.bfloat16
    assert payload["self_attn.softmax_offset"].dtype == torch.bfloat16
    assert payload["e_score_correction_bias"].dtype == torch.float32


@pytest.mark.parametrize("include_flint_args", [True, False])
def test_sglang_launch_forwards_explicit_flint_runtime_args(include_flint_args):
    functions = _load_functions(
        "nemo_rl/models/generation/sglang/sglang_worker.py",
        {"_compute_server_args"},
        {},
        "SGLangGenerationWorker",
    )
    explicit = {
        "attention_backend": "fa4",
        "moe_runner_backend": "flashinfer_trtllm_routed",
        "enable_fp32_lm_head": True,
        "swa_full_tokens_ratio": 0.05,
        "disable_flashinfer_autotune": True,
        "disable_radix_cache": True,
        "disable_overlap_schedule": True,
        "trust_remote_code": False,
    }
    config = {
        "sglang_server_config": {
            "num_gpus_per_engine": 1,
            "needs_offload": True,
            "cpu_weight_backup": True,
        },
        "model_path": "/model/flint-four-layers",
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "ep_size": 1,
        "random_seed": 42,
        "skip_server_warmup": True,
    }
    if include_flint_args:
        config.update(explicit)
    worker = SimpleNamespace(
        sglang_cfg={"sglang_cfg": config},
        num_gpus_per_engine=1,
        gpus_per_node=1,
        base_gpu_id=0,
        rank=0,
        _to_local_gpu_id=lambda gpu_id: gpu_id,
    )
    args = functions["_compute_server_args"](
        worker, "127.0.0.1:1234", 1235, "127.0.0.1", 1236
    )

    assert args["model_path"] == config["model_path"]
    for name, value in explicit.items():
        if include_flint_args:
            assert args[name] == value
        else:
            if name == "trust_remote_code":
                assert args[name] is True
            else:
                assert name not in args


_PLUGINS_LOADED = False


class _StableModelConfig:
    num_hidden_layers = 4


class _ProbeServerArgs:
    """Emulate ServerArgs caching an installed class only after plugin loading."""

    def __init__(self, **kwargs):
        self.host = kwargs["host"]
        self.node_rank = 0
        self.api_key = None
        self.model_config = (
            _StableModelConfig()
            if _PLUGINS_LOADED
            else type(
                "FlintConfig", (), {"__module__": "transformers_modules.missing"}
            )()
        )

    def url(self):
        return "http://127.0.0.1:1234"


def _receive_spawned_model_config(server_args, queue):
    queue.put(
        (
            type(server_args.model_config).__module__,
            server_args.model_config.num_hidden_layers,
        )
    )


def test_plugins_register_stable_config_before_server_args_spawn(monkeypatch):
    plugins = ModuleType("sglang.srt.plugins")
    plugins.load_plugins = lambda: monkeypatch.setattr(
        sys.modules[__name__], "_PLUGINS_LOADED", True
    )
    server_args_module = ModuleType("sglang.srt.server_args")
    server_args_module.ServerArgs = _ProbeServerArgs
    monkeypatch.setitem(sys.modules, plugins.__name__, plugins)
    monkeypatch.setitem(sys.modules, server_args_module.__name__, server_args_module)

    class ProbeProcess:
        def __init__(self, target, args):
            self.args = args

        def start(self):
            # This is the serialization boundary that failed before workers started.
            self.roundtrip_args = pickle.loads(pickle.dumps(self.args))

        def is_alive(self):
            return True

    functions = _load_functions(
        "nemo_rl/models/generation/sglang/sglang_worker.py",
        {"_launch_server_process"},
        {
            "multiprocessing": SimpleNamespace(
                Process=ProbeProcess, set_start_method=lambda *a, **kw: None
            ),
            "os": os,
            "logger": logging.getLogger(__name__),
            "_launch_server_with_parent_guard": _receive_spawned_model_config,
        },
        "SGLangGenerationWorker",
    )
    worker = SimpleNamespace(
        server_host="127.0.0.1",
        server_port=1234,
        node_rank=0,
        router_ip=None,
        router_port=None,
        _wait_server_healthy=lambda **kw: None,
    )
    functions["_launch_server_process"](worker, {"host": "127.0.0.1"})
    server_args = worker.process.roundtrip_args[0]

    # A real spawn imports the static class in a fresh process, independent of
    # the parent process's plugin loader and dynamically injected module stubs.
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_receive_spawned_model_config, args=(server_args, queue)
    )
    process.start()
    try:
        assert queue.get(timeout=30) == (__name__, 4)
        process.join(timeout=30)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        queue.close()
