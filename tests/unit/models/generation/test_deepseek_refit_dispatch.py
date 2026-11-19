# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks for merged V4/V4.1 refit control flow.

Compile the actual worker methods with mock vLLM transport hooks. This exercises
model dispatch and cleanup without requiring vLLM's CUDA-only imports.
"""

import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[4]
    / "nemo_rl/models/generation/vllm/vllm_backend.py"
)
METHODS = {
    "_uses_native_layerwise_refit",
    "_uses_deepseek_v4_fp8_refit",
    "_uses_deepseek_v41_fp8_refit",
    "_reject_unsupported_native_refit",
    "_weight_update_lifecycle",
    "_weight_update_errors_are_fatal",
}


@pytest.fixture
def worker_factory(monkeypatch):
    events = []

    def module(name, **attrs):
        result = ModuleType(name)
        result.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    module("vllm.config", set_current_vllm_config=lambda _: nullcontext())
    module("vllm.model_executor.model_loader")
    module(
        "vllm.model_executor.model_loader.reload",
        initialize_layerwise_reload=lambda _: events.append("initialize"),
        finalize_layerwise_reload=lambda *_: events.append("finalize"),
    )
    package = "nemo_rl.models.generation.vllm.quantization"
    helpers = {}
    for name in ("deepseek_v4_fp8", "deepseek_v41_fp8"):
        helpers[name] = module(
            f"{package}.{name}",
            prepare_refit=lambda _, label=name: (
                events.append((label, "prepare")),
                {label},
            )[1],
            finalize_refit=lambda _, label=name: events.append((label, "finalize")),
            restore_refit=lambda names, label=name: events.append(
                (label, "restore", names)
            ),
        )
    helpers["fp8"] = module(f"{package}.fp8", is_fp8_model=lambda config: config.fp8)
    module(package, **helpers)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "VllmInternalWorkerExtension"
    )
    cls.bases = []
    cls.decorator_list = []
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    isolated = ast.fix_missing_locations(
        ast.Module(body=[future, cls], type_ignores=[])
    )
    namespace = {
        "torch": torch,
        "contextmanager": contextmanager,
        "_reload_target_module_ids": lambda _: set(),
        "_refresh_hpc_modules_after_layerwise_reload": lambda _: None,
    }
    exec(compile(isolated, str(SOURCE), "exec"), namespace)

    def create(model_type, *, fp8=True):
        worker = namespace[cls.name]()
        worker.model_runner = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(model_type=model_type)),
            vllm_config=SimpleNamespace(fp8=fp8),
        )
        worker.model_config = object()
        worker.device = "cpu"
        worker._frozen_engram_initialized = False
        worker._nrl_layerwise_reload_failure = None
        worker._nrl_layerwise_reload_active = False
        worker._uses_unquantized_flashinfer_trtllm = lambda: False
        worker._validate_native_layerwise_refit = lambda _: None
        worker._maybe_process_mtp_drafter_after_loading = lambda: None
        return worker, events

    return create


@pytest.mark.parametrize(
    "model_type", ["deepseek_v4", "deepseek_v41", "deepseek_v41_text", "qwen3"]
)
@pytest.mark.parametrize("fp8", [False, True])
def test_model_and_transport_dispatch(worker_factory, model_type, fp8):
    worker, _ = worker_factory(model_type, fp8=fp8)
    expected = fp8 and model_type != "qwen3"
    assert worker._uses_native_layerwise_refit("collective") is expected
    assert worker._uses_native_layerwise_refit("ipc") is expected
    assert not worker._uses_native_layerwise_refit("nccl_reshard")
    if expected:
        with pytest.raises(RuntimeError, match="bypasses the model"):
            worker._reject_unsupported_native_refit("checkpoint_engine")
    else:
        worker._reject_unsupported_native_refit("checkpoint_engine")


@pytest.mark.parametrize(
    "model_type", ["deepseek_v4", "deepseek_v41", "deepseek_v41_text"]
)
@pytest.mark.parametrize("fail", [False, True])
def test_correct_hooks_and_failure_cleanup(worker_factory, model_type, fail):
    worker, events = worker_factory(model_type)
    helper = "deepseek_v4_fp8" if model_type == "deepseek_v4" else "deepseek_v41_fp8"
    failure = RuntimeError("stream failed")
    with pytest.raises(RuntimeError, match="stream failed") if fail else nullcontext():
        with worker._weight_update_lifecycle("collective") as finalize:
            events.append("stream")
            if fail:
                raise failure
            finalize()
    expected = [(helper, "prepare"), "initialize", "stream"]
    if not fail:
        expected += ["finalize", (helper, "finalize")]
    expected += [(helper, "restore", {helper})]
    assert events == expected
    assert not worker._nrl_layerwise_reload_active
    if fail:
        assert worker._nrl_layerwise_reload_failure is failure
        assert worker._weight_update_errors_are_fatal()
        with pytest.raises(RuntimeError, match="unusable"):
            with worker._weight_update_lifecycle("collective"):
                pytest.fail("A failed native refit must not be reused")


@pytest.mark.parametrize("model_type", ["deepseek_v41", "deepseek_v41_text"])
def test_frozen_engram_guard_precedes_reload(worker_factory, model_type):
    worker, events = worker_factory(model_type)
    worker._frozen_engram_initialized = True
    with pytest.raises(ValueError, match="Frozen Engram"):
        with worker._weight_update_lifecycle("collective"):
            pytest.fail("Frozen Engram must not enter full-model FP8 reload")
    assert events == []
