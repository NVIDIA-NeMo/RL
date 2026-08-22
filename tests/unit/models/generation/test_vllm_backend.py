# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# NOTE: vllm_backend imports `vllm` eagerly at module top, so it is only imported
# inside the test bodies (which are marked @pytest.mark.vllm). This keeps the
# module collectable in the non-vllm unit lane, where these tests are deselected.

import contextlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from safetensors.torch import save_file


def _make_collective_update_extension(backend):
    ext = backend.VllmInternalWorkerExtension.__new__(
        backend.VllmInternalWorkerExtension
    )
    state_info = object()
    ext.state_dict_info = {"model.weight": state_info}
    ext.model_update_group = object()
    ext.model_runner = SimpleNamespace(model=torch.nn.Module(), vllm_config=object())
    ext.model_config = object()
    ext.device = object()
    return ext, state_info


def _write_sharded_checkpoint(model_dir, shards):
    """Write safetensors shards plus a model.safetensors.index.json.

    Args:
        model_dir: Directory (pathlib.Path) to write the checkpoint into.
        shards: Mapping of shard filename -> {weight_name: tensor}.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for shard_name, tensors in shards.items():
        save_file(tensors, str(model_dir / shard_name))
        for name in tensors:
            weight_map[name] = shard_name
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)


def _make_extension_with_drafter(mtp_start_layer_idx, num_mtp_layers):
    """Build a VllmInternalWorkerExtension with a mocked drafter and stubbed refit."""
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.device = torch.device("cpu")
    predictor = SimpleNamespace(
        mtp_start_layer_idx=mtp_start_layer_idx, num_mtp_layers=num_mtp_layers
    )
    ext.model_runner = MagicMock()
    draft_model = torch.nn.Module()
    setattr(draft_model, "model", predictor)
    ext.model_runner.drafter.model = draft_model
    # Isolate this test from _load_draft_weights internals.
    ext._load_draft_weights = MagicMock()
    return ext


def _patch_vllm_postload(monkeypatch):
    """Stub the vLLM post-load helpers imported inside load_mtp_weights_from_disk."""
    monkeypatch.setattr(
        "vllm.config.set_current_vllm_config", lambda cfg: contextlib.nullcontext()
    )
    process_weights = MagicMock()
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        process_weights,
    )
    return process_weights


def _make_mtp_refit_extension(
    *, method="mtp", from_disk=False, has_drafter=True, draft_model_config=None
):
    """Build an extension for exercising the MTP-refit drafter gating.

    The drafter here is fed from the refit stream (co-trained MTP layer), as
    opposed to the disk-load path built by ``_make_extension_with_drafter``.

    Returns:
        (ext, drafter_model): drafter_model is None when has_drafter is False.
    """
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.device = torch.device("cpu")
    ext._mtp_drafter_from_disk = from_disk

    spec_config = (
        None
        if method is None
        else SimpleNamespace(method=method, draft_model_config=draft_model_config)
    )
    drafter_model = SimpleNamespace(load_weights=MagicMock()) if has_drafter else None
    ext.model_runner = SimpleNamespace(
        vllm_config=SimpleNamespace(speculative_config=spec_config),
        drafter=SimpleNamespace(model=drafter_model) if has_drafter else None,
    )
    return ext, drafter_model


def _make_unquantized_moe_model(moe_backend: str) -> SimpleNamespace:
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        UnquantizedMoeBackend,
    )
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    quant_method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
    quant_method.unquantized_backend = UnquantizedMoeBackend(moe_backend)
    module = SimpleNamespace(quant_method=quant_method)
    return SimpleNamespace(modules=lambda: [module])


@pytest.mark.vllm
def test_refresh_hpc_modules_after_layerwise_reload(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    class FakeHpcModule:
        def __init__(self):
            self.process_weights_after_loading = MagicMock()

    hpc_module = FakeHpcModule()
    other_module = object()
    model = SimpleNamespace(modules=lambda: [other_module, hpc_module])
    monkeypatch.setattr("vllm.model_executor.layers.hpc.HpcModule", FakeHpcModule)

    vllm_backend._refresh_hpc_modules_after_layerwise_reload(model)

    hpc_module.process_weights_after_loading.assert_called_once_with(model)


class _DeferredReloadLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Parameter(torch.zeros(2))
        self.second = torch.nn.Parameter(torch.zeros(2))


class _DeferredReloadModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = _DeferredReloadLayer()

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        for name, loaded_weight in weights:
            param = params[name]
            weight_loader = getattr(param, "weight_loader", None)
            assert callable(weight_loader)
            weight_loader(param, loaded_weight)


@pytest.mark.vllm
def test_unquantized_weight_update_uses_layerwise_reload(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    call_order = []
    model = _make_unquantized_moe_model("FlashInfer TRTLLM")
    model_config = object()
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="auto"), quant_config=None
    )

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(model=model, vllm_config=vllm_config)
    ext.model_config = model_config
    ext.device = torch.device("cpu")
    ext._maybe_process_mtp_drafter_after_loading = lambda: call_order.append("mtp")
    ext._maybe_process_fp8_kv_cache = MagicMock()

    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    @contextlib.contextmanager
    def set_current_vllm_config(config):
        assert config is vllm_config
        call_order.append("config_enter")
        try:
            yield
        finally:
            call_order.append("config_exit")

    monkeypatch.setattr("vllm.config.set_current_vllm_config", set_current_vllm_config)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.initialize_layerwise_reload",
        lambda reload_model: call_order.append(("initialize", reload_model)),
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.finalize_layerwise_reload",
        lambda reload_model, config: call_order.append(
            ("finalize", reload_model, config)
        ),
    )
    monkeypatch.setattr(
        vllm_backend,
        "_refresh_hpc_modules_after_layerwise_reload",
        lambda reload_model: call_order.append(("hpc", reload_model)),
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        lambda *_args: pytest.fail(
            "unquantized refit must use vLLM's native layerwise reload lifecycle"
        ),
    )

    for _ in range(2):
        with ext._weight_update_lifecycle("collective") as finalize:
            call_order.append("load")
            finalize()
        assert ext._nrl_layerwise_reload_active is False

    expected_cycle = [
        "config_enter",
        ("initialize", model),
        "load",
        ("finalize", model, model_config),
        ("hpc", model),
        "mtp",
        "config_exit",
    ]
    assert call_order == expected_cycle * 2
    ext._maybe_process_fp8_kv_cache.assert_not_called()


@pytest.mark.vllm
def test_unquantized_nccl_reshard_keeps_existing_refit_lifecycle(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    model = _make_unquantized_moe_model("FlashInfer TRTLLM")
    model_config = object()
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="flashinfer_trtllm"),
        quant_config=None,
    )
    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(model=model, vllm_config=vllm_config)
    ext.model_config = model_config
    ext.device = torch.device("cpu")
    ext._maybe_process_mtp_drafter_after_loading = MagicMock()
    ext._maybe_process_fp8_kv_cache = MagicMock()

    monkeypatch.setattr(
        "vllm.config.set_current_vllm_config", lambda _: contextlib.nullcontext()
    )
    process = MagicMock()
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        process,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.initialize_layerwise_reload",
        lambda _: pytest.fail(
            "NCCL reshard must preserve the direct-buffer parameter mapping"
        ),
    )

    with ext._weight_update_lifecycle("nccl_reshard") as finalize:
        finalize()

    process.assert_called_once_with(model, model_config, ext.device)
    ext._maybe_process_mtp_drafter_after_loading.assert_called_once_with()
    ext._maybe_process_fp8_kv_cache.assert_called_once_with()


@pytest.mark.vllm
def test_layerwise_reload_preserves_deferred_weight_across_buffer_reuse(monkeypatch):
    from vllm.model_executor.model_loader.reload import record_metadata_for_reloading

    from nemo_rl.models.generation.vllm import vllm_backend

    model = _DeferredReloadModel()
    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(model=model, vllm_config=object())
    ext.model_config = None
    ext.device = torch.device("cpu")
    ext._uses_unquantized_flashinfer_trtllm = lambda: True
    ext._validate_native_layerwise_refit = lambda: None
    ext._maybe_process_mtp_drafter_after_loading = MagicMock()

    monkeypatch.setattr(
        "vllm.config.set_current_vllm_config", lambda _: contextlib.nullcontext()
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    transport_buffer = torch.empty(2)
    record_metadata_for_reloading(model)

    with ext._weight_update_lifecycle("collective") as finalize:
        transport_buffer.copy_(torch.tensor([1.0, 2.0]))
        ext._load_full_hf_weights([("layer.first", transport_buffer)])

        transport_buffer.copy_(torch.tensor([7.0, 8.0]))
        ext._load_full_hf_weights([("layer.second", transport_buffer)])
        finalize()

    torch.testing.assert_close(model.layer.first, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(model.layer.second, torch.tensor([7.0, 8.0]))


@pytest.mark.vllm
def test_layerwise_reload_detaches_deferred_transport_weights(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    source = torch.ones(4)
    unrelated = torch.full((2,), 7.0)
    source_args = SimpleNamespace(arguments={"loaded_weight": source[:2]})
    unrelated_args = SimpleNamespace(arguments={"loaded_weight": unrelated})
    model = SimpleNamespace(modules=lambda: [object()])
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.layerwise.get_layerwise_info",
        lambda _module: SimpleNamespace(
            loaded_weights=[("source", source_args), ("other", unrelated_args)]
        ),
    )

    vllm_backend._detach_pending_layerwise_weights(
        model, {source.untyped_storage().data_ptr()}
    )

    detached = source_args.arguments["loaded_weight"]
    assert detached.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert unrelated_args.arguments["loaded_weight"] is unrelated
    source.zero_()
    torch.testing.assert_close(detached, torch.ones(2))


@pytest.mark.vllm
def test_layerwise_reload_preserves_weight_load_error(monkeypatch, caplog):
    from nemo_rl.models.generation.vllm import vllm_backend

    load_error = RuntimeError("load failed")
    model = SimpleNamespace(load_weights=MagicMock(side_effect=load_error))
    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(model=model)
    ext._nrl_layerwise_reload_active = True
    monkeypatch.setattr(
        vllm_backend,
        "_detach_pending_layerwise_weights",
        MagicMock(side_effect=RuntimeError("detach failed")),
    )

    with pytest.raises(RuntimeError, match="load failed") as exc_info:
        ext._load_full_hf_weights([("model.weight", torch.ones(1))])

    assert exc_info.value is load_error
    assert "Failed to detach deferred weights" in caplog.text


@pytest.mark.vllm
def test_layerwise_reload_propagates_detach_error_after_successful_load(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    detach_error = RuntimeError("detach failed")
    model = SimpleNamespace(load_weights=MagicMock())
    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(model=model)
    ext._nrl_layerwise_reload_active = True
    monkeypatch.setattr(
        vllm_backend,
        "_detach_pending_layerwise_weights",
        MagicMock(side_effect=detach_error),
    )

    with pytest.raises(RuntimeError, match="detach failed") as exc_info:
        ext._load_full_hf_weights([("model.weight", torch.ones(1))])

    assert exc_info.value is detach_error
    model.load_weights.assert_called_once()


@pytest.mark.vllm
def test_fp8_flashinfer_trtllm_keeps_existing_refit_lifecycle(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    model = object()
    model_config = object()
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="flashinfer_trtllm"),
        quant_config=object(),
    )
    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(model=model, vllm_config=vllm_config)
    ext.model_config = model_config
    ext.device = torch.device("cpu")
    ext._maybe_process_mtp_drafter_after_loading = MagicMock()
    ext._maybe_process_fp8_kv_cache = MagicMock()

    monkeypatch.setattr(
        "vllm.config.set_current_vllm_config", lambda _: contextlib.nullcontext()
    )
    process = MagicMock()
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        process,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.initialize_layerwise_reload",
        lambda _: pytest.fail("FP8 must not use the unquantized reload lifecycle"),
    )

    with ext._weight_update_lifecycle("collective") as finalize:
        finalize()

    process.assert_called_once_with(model, model_config, ext.device)
    ext._maybe_process_mtp_drafter_after_loading.assert_called_once_with()
    ext._maybe_process_fp8_kv_cache.assert_called_once_with()


@pytest.mark.vllm
def test_extension_capability_can_disable_unquantized_reload():
    from nemo_rl.models.generation.vllm import vllm_backend

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(
        model=object(),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(moe_backend="flashinfer_trtllm"),
            quant_config=None,
        ),
    )
    ext._supports_unquantized_flashinfer_trtllm_refit = lambda: False

    assert ext._uses_unquantized_flashinfer_trtllm() is False


@pytest.mark.vllm
def test_realized_moe_backend_controls_native_refit_lifecycle():
    from nemo_rl.models.generation.vllm import vllm_backend

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(
        model=_make_unquantized_moe_model("TRITON"),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(moe_backend="flashinfer_trtllm"),
            quant_config=None,
        ),
    )

    assert ext._uses_unquantized_flashinfer_trtllm() is False

    ext.model_runner.model = _make_unquantized_moe_model("FlashInfer TRTLLM")
    ext.model_runner.vllm_config.kernel_config.moe_backend = "auto"

    assert ext._uses_unquantized_flashinfer_trtllm() is True


@pytest.mark.vllm
def test_quantized_model_does_not_use_unquantized_refit_lifecycle():
    from nemo_rl.models.generation.vllm import vllm_backend

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(
        model=_make_unquantized_moe_model("FlashInfer TRTLLM"),
        vllm_config=SimpleNamespace(quant_config=object()),
    )

    assert ext._uses_unquantized_flashinfer_trtllm() is False


@pytest.mark.vllm
@pytest.mark.parametrize(
    ("moe_backend", "quant_config", "expected"),
    [
        ("FlashInfer TRTLLM", None, True),
        ("TRITON", None, False),
        ("FlashInfer TRTLLM", object(), False),
    ],
)
def test_weight_update_errors_are_fatal_only_for_native_trtllm_refit(
    moe_backend, quant_config, expected
):
    from nemo_rl.models.generation.vllm import vllm_backend

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(
        model=_make_unquantized_moe_model(moe_backend),
        vllm_config=SimpleNamespace(quant_config=quant_config),
    )

    assert ext._weight_update_errors_are_fatal() is expected


@pytest.mark.vllm
def test_unquantized_reload_rejects_cotrained_mtp_during_prepare():
    from nemo_rl.models.generation.vllm import vllm_backend

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(
        model=_make_unquantized_moe_model("FlashInfer TRTLLM"),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(moe_backend="flashinfer_trtllm"),
            quant_config=None,
        ),
    )
    ext._mtp_drafter_refit_enabled = lambda: True

    with pytest.raises(RuntimeError, match="co-trained MTP drafter"):
        ext.prepare_refit_info({"model.weight": object()})

    assert not hasattr(ext, "state_dict_info")


@pytest.mark.vllm
def test_failed_unquantized_reload_marks_worker_unusable(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(
        model=_make_unquantized_moe_model("FlashInfer TRTLLM"),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(moe_backend="flashinfer_trtllm"),
            quant_config=None,
        ),
    )
    ext.model_config = object()
    ext.device = torch.device("cpu")
    ext._mtp_drafter_refit_enabled = lambda: False
    monkeypatch.setattr(
        "vllm.config.set_current_vllm_config", lambda _: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.initialize_layerwise_reload",
        lambda _: None,
    )

    failure = RuntimeError("load failed")
    with pytest.raises(RuntimeError, match="load failed"):
        with ext._weight_update_lifecycle("collective"):
            raise failure

    assert ext._nrl_layerwise_reload_failure is failure
    with pytest.raises(RuntimeError, match="worker is unusable"):
        with ext._weight_update_lifecycle("collective"):
            pass


@pytest.mark.vllm
def test_update_weights_from_collective_reraises_on_fatal_native_refit(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(
        model=_make_unquantized_moe_model("FlashInfer TRTLLM"),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(moe_backend="flashinfer_trtllm"),
            quant_config=None,
        ),
    )
    ext.model_config = object()
    ext.device = torch.device("cpu")
    ext.state_dict_info = {"model.weight": ((1,), torch.float32)}
    ext.model_update_group = object()
    ext._mtp_drafter_refit_enabled = lambda: False
    monkeypatch.setattr(
        "vllm.config.set_current_vllm_config", lambda _: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.initialize_layerwise_reload",
        lambda _: None,
    )

    def failing_consumer(**_kwargs):
        raise RuntimeError("transport load failed")

    monkeypatch.setattr(vllm_backend, "packed_broadcast_consumer", failing_consumer)

    # The fatal-native path must re-raise instead of swallowing into False.
    with pytest.raises(RuntimeError, match="transport load failed"):
        ext.update_weights_from_collective()

    assert ext._nrl_layerwise_reload_failure is not None
    with pytest.raises(RuntimeError, match="worker is unusable"):
        ext.update_weights_from_collective()


@pytest.mark.vllm
def test_native_collective_refit_uses_one_transport_buffer(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    ext, _ = _make_collective_update_extension(vllm_backend)
    ext._uses_unquantized_flashinfer_trtllm = lambda: True

    @contextlib.contextmanager
    def lifecycle(_transport):
        yield lambda: None

    ext._weight_update_lifecycle = lifecycle
    observed_num_buffers = None

    def consume(*, iterator, group, src, post_unpack_func, num_buffers=None):
        nonlocal observed_num_buffers
        observed_num_buffers = num_buffers

    monkeypatch.setattr(vllm_backend, "packed_broadcast_consumer", consume)
    monkeypatch.setattr(vllm_backend.gc, "collect", lambda: None)
    monkeypatch.setattr(vllm_backend.torch.cuda, "empty_cache", lambda: None)

    assert ext.update_weights_from_collective() is True
    assert observed_num_buffers == 1


@pytest.mark.vllm
def test_sparse_delta_refit_rejected_for_native_trtllm_backend():
    from nemo_rl.models.generation.vllm import vllm_backend

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(
        model=_make_unquantized_moe_model("FlashInfer TRTLLM"),
        vllm_config=SimpleNamespace(quant_config=None),
    )

    with pytest.raises(RuntimeError, match="sparse-delta refit does not support"):
        ext.prepare_sparse_delta_refit_info({})
    with pytest.raises(RuntimeError, match="sparse-delta refit does not support"):
        ext.update_weights_from_decoded_sparse_payload(b"")


@pytest.mark.vllm
@pytest.mark.parametrize("with_mtp", [False, True])
def test_update_weights_from_collective_processes_weights_after_loading(
    monkeypatch, with_mtp
):
    from nemo_rl.models.generation.vllm import vllm_backend

    call_order = []
    process_calls = []
    draft_model = torch.nn.Module() if with_mtp else None
    draft_model_config = object() if with_mtp else None

    def process_weights_after_loading(model, model_config, device):
        call_order.append("process_mtp" if model is draft_model else "process_main")
        process_calls.append((model, model_config, device))

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        process_weights_after_loading,
    )
    ext, expected_state_info = _make_collective_update_extension(vllm_backend)
    if with_mtp:
        ext._mtp_drafter_from_disk = False
        ext.model_runner.drafter = SimpleNamespace(model=draft_model)
        ext.model_runner.vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="mtp", draft_model_config=draft_model_config
            )
        )

    @contextlib.contextmanager
    def set_current_vllm_config(config):
        assert config is ext.model_runner.vllm_config
        call_order.append("config_enter")
        try:
            yield
        finally:
            call_order.append("config_exit")

    monkeypatch.setattr("vllm.config.set_current_vllm_config", set_current_vllm_config)

    def load_weights(weights):
        call_order.append("load")
        assert weights == [("model.weight", "weight-value")]

    def packed_broadcast_consumer(
        iterator, group, src, post_unpack_func, num_buffers=None
    ):
        call_order.append("broadcast")
        assert list(iterator) == [("model.weight", expected_state_info)]
        assert group is ext.model_update_group
        assert src == 0
        post_unpack_func([("model.weight", "weight-value")])

    ext._load_weights = load_weights
    ext._maybe_process_fp8_kv_cache = lambda: call_order.append("kv")
    monkeypatch.setattr(
        vllm_backend, "packed_broadcast_consumer", packed_broadcast_consumer
    )
    monkeypatch.setattr(vllm_backend.gc, "collect", lambda: call_order.append("gc"))
    monkeypatch.setattr(
        vllm_backend.torch.cuda,
        "empty_cache",
        lambda: call_order.append("empty_cache"),
    )

    assert ext.update_weights_from_collective() is True

    expected_process_calls = [(ext.model_runner.model, ext.model_config, ext.device)]
    expected_call_order = [
        "broadcast",
        "load",
        "config_enter",
        "process_main",
        "config_exit",
    ]
    if with_mtp:
        expected_process_calls.append((draft_model, draft_model_config, ext.device))
        expected_call_order.extend(["config_enter", "process_mtp", "config_exit"])
    expected_call_order.extend(["kv", "gc", "empty_cache"])

    assert process_calls == expected_process_calls
    assert call_order == expected_call_order


@pytest.mark.vllm
@pytest.mark.parametrize(
    "method_name",
    ["update_weights_via_ipc_zmq", "update_weights_from_collective"],
)
@pytest.mark.parametrize(
    "worker_results, expected", [([True, True], True), ([True, False], False)]
)
def test_sync_weight_updates_check_every_internal_worker(
    method_name, worker_results, expected
):
    """A failure on a later PP rank must not be hidden by rank zero success."""
    from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl

    worker = VllmGenerationWorkerImpl.__new__(VllmGenerationWorkerImpl)
    worker.cfg = {"vllm_cfg": {"async_engine": False}}
    worker.llm = SimpleNamespace(collective_rpc=MagicMock(return_value=worker_results))

    assert getattr(worker, method_name)() is expected


@pytest.mark.vllm
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name",
    ["update_weights_via_ipc_zmq_async", "update_weights_from_collective_async"],
)
@pytest.mark.parametrize(
    "worker_results, expected", [([True, True], True), ([True, False], False)]
)
async def test_async_weight_updates_check_every_internal_worker(
    method_name, worker_results, expected
):
    """Async refit also reports failures from every internal PP rank."""
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {
        "vllm_cfg": {
            "async_engine": True,
            "reset_encoder_cache_after_weight_update": True,
        }
    }
    worker.llm = SimpleNamespace(
        collective_rpc=AsyncMock(return_value=worker_results),
        reset_encoder_cache=AsyncMock(),
    )

    assert await getattr(worker, method_name)() is expected
    if expected:
        worker.llm.reset_encoder_cache.assert_awaited_once_with()
    else:
        worker.llm.reset_encoder_cache.assert_not_awaited()


@pytest.mark.vllm
@pytest.mark.asyncio
async def test_async_weight_update_skips_encoder_cache_reset_when_disabled():
    """Text-only and in-flight refit users retain the existing cache behavior."""
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {"vllm_cfg": {"async_engine": True}}
    worker.llm = SimpleNamespace(
        collective_rpc=AsyncMock(return_value=[True]),
        reset_encoder_cache=AsyncMock(),
    )

    assert await worker.update_weights_from_collective_async() is True
    worker.llm.reset_encoder_cache.assert_not_awaited()


@pytest.mark.vllm
@pytest.mark.asyncio
async def test_async_weight_update_fails_when_encoder_cache_reset_fails():
    """A successful refit must not resume with stale multimodal encoder outputs."""
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {
        "vllm_cfg": {
            "async_engine": True,
            "reset_encoder_cache_after_weight_update": True,
        }
    }
    worker.llm = SimpleNamespace(
        collective_rpc=AsyncMock(return_value=[True]),
        reset_encoder_cache=AsyncMock(side_effect=RuntimeError("reset failed")),
    )

    assert await worker.update_weights_from_collective_async() is False


@pytest.mark.vllm
@pytest.mark.asyncio
async def test_nccl_reshard_refit_resets_encoder_cache():
    """NCCL-reshard refits invalidate encoder outputs just like other transports."""
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {
        "vllm_cfg": {
            "async_engine": True,
            "reset_encoder_cache_after_weight_update": True,
        }
    }
    worker.llm = SimpleNamespace(
        collective_rpc=AsyncMock(return_value=[True]),
        reset_encoder_cache=AsyncMock(),
    )

    assert await worker.nccl_reshard_refit_async() is True
    worker.llm.reset_encoder_cache.assert_awaited_once_with()


@pytest.mark.vllm
def test_update_weights_via_ipc_acks_manifest_error_and_returns_false(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend
    from nemo_rl.models.policy.utils import IPCProtocol

    class FakeSocket:
        def __init__(self):
            self.sent = []

        def recv_pyobj(self):
            return IPCProtocol.COMPLETE

        def send(self, payload):
            self.sent.append(payload)

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.state_dict_info = {"model.weight": (torch.Size([1]), torch.float32)}
    ext.zmq_socket = FakeSocket()
    ext.maybe_init_zmq = lambda: None

    @contextlib.contextmanager
    def lifecycle(_transport):
        yield lambda: pytest.fail("an incomplete transfer must not be finalized")

    ext._weight_update_lifecycle = lifecycle

    assert ext.update_weights_via_ipc_zmq() is False
    assert ext.zmq_socket.sent == [IPCProtocol.ACK.value.encode()]


@pytest.mark.vllm
def test_read_mtp_layer_weights_from_checkpoint_filters_and_reads(tmp_path):
    """Only the requested MTP layer tensors are read, across the shards holding them."""
    from nemo_rl.models.generation.vllm.vllm_backend import (
        _read_mtp_layer_weights_from_checkpoint,
    )

    model_dir = tmp_path / "ckpt"
    mtp_block = torch.randn(4, 4)
    mtp_head = torch.randn(2, 4)
    other_layer = torch.randn(4, 4)
    embed = torch.randn(8, 4)
    # MTP layer index is 2; layer 0 and the top-level embed must be ignored.
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00002.safetensors": {
                "model.layers.2.mlp.up_proj.weight": mtp_block,  # MTP, shard 1
                "model.layers.0.mlp.up_proj.weight": other_layer,  # non-MTP, same shard
            },
            "model-00002-of-00002.safetensors": {
                "model.layers.2.shared_head.head.weight": mtp_head,  # MTP, shard 2
                "model.embed_tokens.weight": embed,  # non-MTP, no layer index
            },
        },
    )

    weights = _read_mtp_layer_weights_from_checkpoint(str(model_dir), {2})

    by_name = dict(weights)
    assert set(by_name) == {
        "model.layers.2.mlp.up_proj.weight",
        "model.layers.2.shared_head.head.weight",
    }
    assert torch.equal(by_name["model.layers.2.mlp.up_proj.weight"], mtp_block)
    assert torch.equal(by_name["model.layers.2.shared_head.head.weight"], mtp_head)


@pytest.mark.vllm
def test_load_mtp_weights_from_disk_loads_only_mtp_layer(tmp_path, monkeypatch):
    """Success path: only MTP-layer weights are handed to the drafter, then post-loaded."""
    model_dir = tmp_path / "ckpt"
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00001.safetensors": {
                "model.layers.2.mlp.up_proj.weight": torch.randn(4, 4),  # MTP
                "model.layers.2.embed_tokens.weight": torch.randn(8, 4),  # MTP
                "model.layers.0.mlp.up_proj.weight": torch.randn(4, 4),  # non-MTP
                "model.embed_tokens.weight": torch.randn(8, 4),  # non-MTP
            }
        },
    )
    ext = _make_extension_with_drafter(mtp_start_layer_idx=2, num_mtp_layers=1)
    process_weights = _patch_vllm_postload(monkeypatch)

    result = ext.load_mtp_weights_from_disk(str(model_dir))

    assert result is True
    ext._load_draft_weights.assert_called_once()
    loaded_names = {name for name, _ in ext._load_draft_weights.call_args[0][0]}
    assert loaded_names == {
        "model.layers.2.mlp.up_proj.weight",
        "model.layers.2.embed_tokens.weight",
    }
    process_weights.assert_called_once()


@pytest.mark.vllm
def test_load_mtp_weights_from_disk_uses_layerwise_reload_for_trtllm(
    tmp_path, monkeypatch
):
    """TRTLLM draft weights reload into their preserved runtime storage."""
    from nemo_rl.models.generation.vllm import vllm_backend

    model_dir = tmp_path / "ckpt"
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00001.safetensors": {
                "model.layers.2.mlp.up_proj.weight": torch.randn(4, 4),
            }
        },
    )
    ext = _make_extension_with_drafter(mtp_start_layer_idx=2, num_mtp_layers=1)
    draft_model = ext._get_drafter_model()
    draft_model.modules = _make_unquantized_moe_model("FlashInfer TRTLLM").modules
    call_order = []

    monkeypatch.setattr(
        "vllm.config.set_current_vllm_config", lambda _: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.initialize_layerwise_reload",
        lambda model: call_order.append(("initialize", model)),
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.finalize_layerwise_reload",
        lambda model, config: call_order.append(("finalize", model, config)),
    )
    monkeypatch.setattr(
        vllm_backend,
        "_refresh_hpc_modules_after_layerwise_reload",
        lambda model: call_order.append(("hpc", model)),
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        lambda *_: pytest.fail("TRTLLM draft reload must use the layerwise path"),
    )
    ext._load_draft_weights.side_effect = lambda _: call_order.append("load")

    assert ext.load_mtp_weights_from_disk(str(model_dir)) is True
    assert call_order == [
        ("initialize", draft_model),
        "load",
        (
            "finalize",
            draft_model,
            ext.model_runner.vllm_config.speculative_config.draft_model_config,
        ),
        ("hpc", draft_model),
    ]


@pytest.mark.vllm
@pytest.mark.parametrize("is_last_rank", [False, True])
def test_load_mtp_weights_from_disk_without_drafter(
    tmp_path, monkeypatch, is_last_rank
):
    """Only the pipeline stage that owns the drafter requires it to exist."""
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.device = torch.device("cpu")
    ext.model_runner = MagicMock()
    ext.model_runner.drafter = None
    ext._load_draft_weights = MagicMock()
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_backend.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=is_last_rank),
    )

    if is_last_rank:
        with pytest.raises(RuntimeError, match="drafter model is unavailable"):
            ext.load_mtp_weights_from_disk(str(tmp_path))
    else:
        assert ext.load_mtp_weights_from_disk(str(tmp_path)) is False
    ext._load_draft_weights.assert_not_called()


@pytest.mark.vllm
def test_load_mtp_weights_from_disk_raises_when_mtp_weights_missing(
    tmp_path, monkeypatch
):
    """A checkpoint without the MTP layer(s) fails loudly instead of silently."""
    model_dir = tmp_path / "ckpt"
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00001.safetensors": {
                "model.layers.0.mlp.up_proj.weight": torch.randn(4, 4),
                "model.embed_tokens.weight": torch.randn(8, 4),
            }
        },
    )
    ext = _make_extension_with_drafter(mtp_start_layer_idx=2, num_mtp_layers=1)
    _patch_vllm_postload(monkeypatch)

    with pytest.raises(ValueError, match="No MTP layer weights"):
        ext.load_mtp_weights_from_disk(str(model_dir))
    ext._load_draft_weights.assert_not_called()


@pytest.mark.vllm
def test_load_weights_routes_only_policy_weights_to_mtp_drafter(monkeypatch):
    """The MTP path receives policy weights, while Eagle gets draft-prefixed ones."""
    from nemo_rl.models.generation.vllm.quantization import fp8
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    main_model = SimpleNamespace(load_weights=MagicMock())
    ext.model_runner = SimpleNamespace(
        model=main_model,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(architectures=[])),
    )
    ext._load_draft_weights = MagicMock()
    ext._maybe_refit_mtp_drafter = MagicMock()
    monkeypatch.setattr(fp8, "is_fp8_model", lambda _: False)

    policy_weights = [("model.weight", "policy-value")]
    draft_weights = [("weight", "draft-value")]
    ext._load_weights(policy_weights + [("draft.weight", "draft-value")])

    main_model.load_weights.assert_called_once_with(weights=policy_weights)
    ext._load_draft_weights.assert_called_once_with(draft_weights)
    ext._maybe_refit_mtp_drafter.assert_called_once_with(policy_weights)


@pytest.mark.vllm
@pytest.mark.parametrize(
    "method, from_disk, has_drafter, expected",
    [
        ("mtp", False, True, True),  # co-trained MTP drafter refit from policy stream
        ("deepseek_mtp", False, True, True),  # same, DeepSeek naming
        ("mtp", True, True, False),  # served once from disk -> leave static weights
        ("eagle3", False, True, False),  # non-MTP drafter uses the draft. prefix path
        (None, False, True, False),  # speculative decoding disabled
        ("mtp", False, False, False),  # vLLM built no drafter
    ],
)
def test_mtp_drafter_refit_enabled(method, from_disk, has_drafter, expected):
    """The refit-into-drafter path only fires for a co-trained MTP drafter."""
    ext, _ = _make_mtp_refit_extension(
        method=method, from_disk=from_disk, has_drafter=has_drafter
    )
    assert ext._mtp_drafter_refit_enabled() is expected


@pytest.mark.vllm
def test_maybe_refit_mtp_drafter_loads_when_enabled():
    """A co-trained MTP drafter is fed the (vocab-trimmed) policy weights on refit."""
    ext, drafter_model = _make_mtp_refit_extension(method="mtp", from_disk=False)
    weights = [("mtp.layers.0.weight", "w0")]
    trimmed = [("mtp.layers.0.weight", "trimmed")]
    # Isolate from _trim_vocab_padding, which needs a real vLLM module tree.
    ext._trim_vocab_padding = MagicMock(return_value=trimmed)

    ext._maybe_refit_mtp_drafter(weights)

    ext._trim_vocab_padding.assert_called_once_with(drafter_model, weights)
    drafter_model.load_weights.assert_called_once_with(weights=trimmed)


@pytest.mark.vllm
@pytest.mark.parametrize(
    "method, from_disk",
    [
        ("mtp", True),  # disk-served MTP drafter must not be reloaded on refit
        ("eagle3", False),  # non-MTP drafter is handled elsewhere
    ],
)
def test_maybe_refit_mtp_drafter_noop_when_gated(method, from_disk):
    """The drafter is left untouched for the disk-load path and non-MTP drafters."""
    ext, drafter_model = _make_mtp_refit_extension(method=method, from_disk=from_disk)
    ext._trim_vocab_padding = MagicMock()

    ext._maybe_refit_mtp_drafter([("mtp.layers.0.weight", "w0")])

    ext._trim_vocab_padding.assert_not_called()
    drafter_model.load_weights.assert_not_called()


@pytest.mark.vllm
def test_maybe_process_mtp_drafter_after_loading_when_enabled(monkeypatch):
    """The refit MTP drafter is finalized against its own draft_model_config."""
    draft_model_config = object()
    ext, drafter_model = _make_mtp_refit_extension(
        method="mtp", from_disk=False, draft_model_config=draft_model_config
    )
    process_weights = _patch_vllm_postload(monkeypatch)

    ext._maybe_process_mtp_drafter_after_loading()

    process_weights.assert_called_once_with(
        drafter_model, draft_model_config, ext.device
    )


@pytest.mark.vllm
def test_maybe_process_mtp_drafter_after_loading_noop_when_disk_loaded(monkeypatch):
    """The disk-load path already finalized its weights, so refit skips reprocessing."""
    ext, _ = _make_mtp_refit_extension(method="mtp", from_disk=True)
    process_weights = _patch_vllm_postload(monkeypatch)

    ext._maybe_process_mtp_drafter_after_loading()

    process_weights.assert_not_called()
