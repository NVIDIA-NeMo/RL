# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unquantized refit must work without importing optional FP8 backends."""

import builtins
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.vllm
@pytest.mark.parametrize("has_quant_config", [False, True])
@pytest.mark.parametrize("reload_iterator", [False, True])
def test_unquantized_refit_does_not_import_fp8(
    monkeypatch, has_quant_config, reload_iterator
):
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    config = SimpleNamespace(model_config=SimpleNamespace(architectures=[]))
    if has_quant_config:
        config.quant_config = None
    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.model_runner = SimpleNamespace(model=model, vllm_config=config)
    expected = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    weights = [("weight", expected)]
    received = []

    def load_weights(*, weights):
        with torch.no_grad():
            for name, tensor in weights:
                received.append((name, tensor))
                model.get_parameter(name).copy_(tensor)
        return {name for name, _ in received}

    model.load_weights = load_weights
    original_import = builtins.__import__

    def reject_fp8(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("nemo_rl.models.generation.vllm.quantization"):
            raise ModuleNotFoundError("Optional FP8 backend is unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_fp8)
    if reload_iterator:
        model.load_weights(weights=ext._prepare_reload_weight_iterator(iter(weights)))
    else:
        ext._load_hf_weights(weights)

    assert len(received) == 1
    assert received[0][0] == "weight"
    assert received[0][1] is expected
    torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)


@pytest.mark.vllm
def test_quantized_refit_still_requires_its_backend(monkeypatch):
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.model_runner = SimpleNamespace(
        model=torch.nn.Linear(2, 2, bias=False),
        vllm_config=SimpleNamespace(
            quant_config=object(), model_config=SimpleNamespace(architectures=[])
        ),
    )
    original_import = builtins.__import__

    def reject_fp8(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("nemo_rl.models.generation.vllm.quantization"):
            raise ModuleNotFoundError("Optional FP8 backend is unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_fp8)
    with pytest.raises(ModuleNotFoundError, match="Optional FP8 backend"):
        ext._load_hf_weights([])
    with pytest.raises(ModuleNotFoundError, match="Optional FP8 backend"):
        ext._prepare_reload_weight_iterator(iter([]))
