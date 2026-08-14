# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import types
from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from nemo_rl.modelopt.models.generation import vllm_quant_backend as backend
from nemo_rl.modelopt.models.generation.vllm_quant_worker import (
    VllmQuantAsyncGenerationWorker,
    VllmQuantGenerationWorker,
    _configure_quant_engine_kwargs,
)
from nemo_rl.modelopt.utils import (
    resolve_quant_cfg,
    validate_modelopt_real_quant_policy_config,
)
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)
from nemo_rl.models.generation.vllm.utils import resolve_generation_worker_cls


def _real_quant_config(*, colocated: bool = True) -> dict:
    return {
        "real_quant": True,
        "quant_cfg": None,
        "colocated": {"enabled": colocated},
        "_modelopt_quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "W4A16_NVFP4",
            "ignore": ["lm_head"],
        },
    }


def test_configure_real_quant_uses_policy_descriptor_unchanged(monkeypatch):
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)
    descriptor = _real_quant_config()["_modelopt_quantization_config"]
    original_hf_overrides = {"architectures": ["Qwen2ForCausalLM"]}
    llm_kwargs = {"hf_overrides": original_hf_overrides}

    _configure_quant_engine_kwargs(_real_quant_config(), llm_kwargs)

    assert llm_kwargs["hf_overrides"] == {
        "architectures": ["Qwen2ForCausalLM"],
        "quantization_config": descriptor,
    }
    assert llm_kwargs["weight_transfer_config"] == {"backend": "ipc"}
    assert "quantization" not in llm_kwargs
    assert original_hf_overrides == {"architectures": ["Qwen2ForCausalLM"]}
    assert os.environ["VLLM_MODELOPT_REAL_QUANT"] == "1"
    assert "VLLM_QUANT_CFG" not in os.environ


def test_configure_non_colocated_real_quant_selects_nccl_lifecycle():
    llm_kwargs = {}

    _configure_quant_engine_kwargs(
        _real_quant_config(colocated=False),
        llm_kwargs,
    )

    assert llm_kwargs["weight_transfer_config"] == {"backend": "nccl"}


def test_configure_real_quant_requires_policy_descriptor():
    config = _real_quant_config()
    del config["_modelopt_quantization_config"]

    with pytest.raises(ValueError, match="initialized policy"):
        _configure_quant_engine_kwargs(config, {})


def test_real_quant_selects_modelopt_worker_without_generation_recipe():
    default = "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"

    selected = resolve_generation_worker_cls(default, _real_quant_config())

    assert selected.endswith(".VllmQuantGenerationWorker")


@pytest.mark.parametrize(
    ("policy_config", "message"),
    [
        (
            {"megatron_cfg": {"enabled": False}, "quant_cfg": "NVFP4_DEFAULT_CFG"},
            "Megatron policy",
        ),
        ({"megatron_cfg": {"enabled": True}, "quant_cfg": None}, "policy.quant_cfg"),
    ],
)
def test_real_quant_requires_modelopt_megatron_policy(policy_config, message):
    with pytest.raises(ValueError, match=message):
        validate_modelopt_real_quant_policy_config(
            policy_config, {"backend": "vllm", "real_quant": True}
        )


def test_real_quant_rejects_non_vllm_backend():
    with pytest.raises(ValueError, match="supported only by vLLM"):
        validate_modelopt_real_quant_policy_config(
            {"megatron_cfg": {"enabled": True}, "quant_cfg": "NVFP4_DEFAULT_CFG"},
            {"backend": "sglang", "real_quant": True},
        )


def test_real_quant_cutlass_moe_requires_eager_execution():
    with pytest.raises(ValueError, match="enforce_eager=true"):
        validate_modelopt_real_quant_policy_config(
            {"megatron_cfg": {"enabled": True}, "quant_cfg": "NVFP4_DEFAULT_CFG"},
            {
                "backend": "vllm",
                "real_quant": True,
                "vllm_kwargs": {"moe_backend": "cutlass"},
            },
        )


@pytest.mark.parametrize(
    ("method_name", "base_method_name"),
    [
        ("update_weights_via_ipc_zmq", "update_weights_via_ipc_zmq"),
        ("update_weights_from_collective", "update_weights_from_collective"),
    ],
)
def test_failed_real_quant_refit_fails_fast_and_poisoning_is_terminal(
    monkeypatch, method_name, base_method_name
):
    worker_cls = VllmQuantGenerationWorker.__ray_metadata__.modified_class
    failed_worker = object.__new__(worker_cls)
    failed_worker.cfg = {"real_quant": True}
    failed_worker.llm = object()
    failed_worker.tokenizer = object()
    failed_worker.server_thread = None
    shutdown = MagicMock(side_effect=RuntimeError("cleanup would block"))
    monkeypatch.setattr(failed_worker, "shutdown", shutdown)
    monkeypatch.setattr(
        VllmGenerationWorkerImpl,
        base_method_name,
        lambda self: False,
    )

    with pytest.raises(RuntimeError, match="engine is unusable"):
        getattr(failed_worker, method_name)()
    shutdown.assert_not_called()
    assert failed_worker.llm is None
    assert failed_worker.tokenizer is None

    with pytest.raises(RuntimeError, match="unusable after a failed refit"):
        getattr(failed_worker, method_name)()

    fresh_worker = object.__new__(worker_cls)
    fresh_worker.cfg = {"real_quant": True}
    monkeypatch.setattr(
        VllmGenerationWorkerImpl,
        base_method_name,
        lambda self: True,
    )
    assert getattr(fresh_worker, method_name)()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "base_method_name"),
    [
        ("update_weights_via_ipc_zmq_async", "update_weights_via_ipc_zmq_async"),
        (
            "update_weights_from_collective_async",
            "update_weights_from_collective_async",
        ),
    ],
)
async def test_failed_async_real_quant_refit_fails_fast_and_poisoning_is_terminal(
    monkeypatch, method_name, base_method_name
):
    worker_cls = VllmQuantAsyncGenerationWorker.__ray_metadata__.modified_class
    failed_worker = object.__new__(worker_cls)
    failed_worker.cfg = {"real_quant": True}
    failed_worker.llm = object()
    failed_worker.tokenizer = object()
    failed_worker.server_thread = MagicMock()
    failed_worker.http_server = types.SimpleNamespace(should_exit=False)

    async def failed_refit(self):
        return False

    shutdown = AsyncMock(side_effect=RuntimeError("cleanup would block"))
    monkeypatch.setattr(failed_worker, "shutdown", shutdown)
    monkeypatch.setattr(
        VllmAsyncGenerationWorkerImpl,
        base_method_name,
        failed_refit,
    )

    with pytest.raises(RuntimeError, match="engine is unusable"):
        await getattr(failed_worker, method_name)()
    shutdown.assert_not_awaited()
    assert failed_worker.http_server.should_exit
    assert failed_worker.llm is None
    assert failed_worker.tokenizer is None

    with pytest.raises(RuntimeError, match="unusable after a failed refit"):
        await getattr(failed_worker, method_name)()

    fresh_worker = object.__new__(worker_cls)
    fresh_worker.cfg = {"real_quant": True}
    monkeypatch.setattr(
        VllmAsyncGenerationWorkerImpl,
        base_method_name,
        lambda self: _async_true(),
    )
    assert await getattr(fresh_worker, method_name)()


async def _async_true():
    return True


@pytest.mark.parametrize("refit_transport", ["nixl", "nccl_reshard"])
def test_vllm_constructor_rejects_real_quant_explicit_transport(refit_transport):
    config = _real_quant_config()
    config["refit_transport"] = refit_transport

    with pytest.raises(ValueError, match="refit_transport must be null"):
        VllmGeneration(MagicMock(), config)


def test_vllm_constructor_requires_real_quant_policy_descriptor():
    config = _real_quant_config()
    del config["_modelopt_quantization_config"]

    with pytest.raises(ValueError, match="initialized policy"):
        VllmGeneration(MagicMock(), config)


def test_configure_fake_quant_keeps_modelopt_worker(monkeypatch, tmp_path):
    recipe = tmp_path / "quant.yaml"
    recipe.write_text("quantize: {}\n")
    llm_kwargs = {}
    config = {
        "real_quant": False,
        "quant_cfg": str(recipe),
        "colocated": {"enabled": True},
    }

    _configure_quant_engine_kwargs(config, llm_kwargs)

    assert llm_kwargs["worker_cls"].endswith(".FakeQuantWorker")
    assert llm_kwargs["moe_backend"] == "triton"
    assert os.environ["VLLM_QUANT_CFG"] == str(recipe.resolve())
    assert "VLLM_MODELOPT_REAL_QUANT" not in os.environ


def test_real_quant_lifecycle_delegates_to_vllm_public_transaction(monkeypatch):
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    events = []
    monkeypatch.setattr(extension, "_is_real_quant_model", lambda: True)
    monkeypatch.setattr(
        extension,
        "start_weight_update",
        lambda: events.append("start"),
        raising=False,
    )
    monkeypatch.setattr(
        extension,
        "finish_weight_update",
        lambda: events.append("finish"),
        raising=False,
    )
    monkeypatch.setattr(
        extension,
        "_maybe_process_mtp_drafter_after_loading",
        lambda: events.append("mtp"),
    )
    monkeypatch.setattr(backend.torch.accelerator, "synchronize", lambda: None)

    with extension._weight_update_lifecycle("ipc") as finalize:
        events.append("load")
        finalize()

    assert events == ["start", "load", "finish", "mtp"]


def test_non_real_quant_lifecycle_delegates_to_base(monkeypatch):
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    monkeypatch.setattr(extension, "_is_real_quant_model", lambda: False)
    sentinel = object()
    events = []

    def base_lifecycle(self, transport):
        del self
        events.append(transport)
        return nullcontext(sentinel)

    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_weight_update_lifecycle",
        base_lifecycle,
    )

    with extension._weight_update_lifecycle("collective") as finalize:
        assert finalize is sentinel

    assert events == ["collective"]


def test_real_quant_load_owns_transport_tensors(monkeypatch):
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    monkeypatch.setattr(extension, "_is_real_quant_model", lambda: True)
    forwarded = []
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        lambda self, weights: forwarded.extend(weights),
    )
    source = torch.arange(8, dtype=torch.float32)

    extension._load_weights([("model.layers.0.weight", source)])

    assert forwarded[0][0] == "model.layers.0.weight"
    assert torch.equal(forwarded[0][1], source)
    assert (
        forwarded[0][1].untyped_storage().data_ptr()
        != source.untyped_storage().data_ptr()
    )


def test_real_quant_refit_errors_propagate(monkeypatch):
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    monkeypatch.setattr(extension, "_is_real_quant_model", lambda: True)
    monkeypatch.setattr(extension, "start_weight_update", lambda: None, raising=False)
    monkeypatch.setattr(
        extension,
        "finish_weight_update",
        lambda: (_ for _ in ()).throw(RuntimeError("broken")),
        raising=False,
    )
    monkeypatch.setattr(
        extension, "_maybe_process_mtp_drafter_after_loading", lambda: None
    )

    with pytest.raises(RuntimeError, match="finalization failed"):
        with extension._weight_update_lifecycle("ipc") as finalize:
            finalize()

    assert extension._weight_update_errors_are_fatal()


def test_resolve_quant_cfg_accepts_builtin_modelopt_constant(monkeypatch):
    import modelopt.torch.quantization as mtq

    sentinel = {"quant_cfg": [], "algorithm": "max"}
    monkeypatch.setattr(mtq, "UNIT_TEST_CFG", sentinel, raising=False)

    assert resolve_quant_cfg("UNIT_TEST_CFG") is sentinel


def test_resolve_quant_cfg_extracts_recipe_quantize_section(monkeypatch):
    import modelopt.recipe

    monkeypatch.setattr(
        modelopt.recipe,
        "load_config",
        lambda _: {"quantize": {"quant_cfg": [{"quantizer_name": "*"}]}},
    )

    assert resolve_quant_cfg("unit-test-recipe") == {
        "quant_cfg": [{"quantizer_name": "*"}],
        "algorithm": "max",
    }


def test_get_weight_snapshot_returns_copy():
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    model = torch.nn.Linear(2, 2, bias=False)
    extension.model_runner = types.SimpleNamespace(model=model)

    snapshot = extension.get_weight_snapshot("weight")

    assert torch.equal(snapshot, model.weight)
    assert snapshot.device.type == "cpu"
    assert snapshot.data_ptr() != model.weight.data_ptr()
