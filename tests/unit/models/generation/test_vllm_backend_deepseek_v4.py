# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import contextlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytestmark = pytest.mark.vllm


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v3"])
@pytest.mark.parametrize("fp8_enabled", [False, True])
def test_collective_reload_api_guard(monkeypatch, model_type, fp8_enabled):
    from nemo_rl.models.generation.vllm import vllm_backend
    from nemo_rl.models.generation.vllm.quantization import fp8

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type=model_type)
    ext.model_runner = SimpleNamespace(
        model=model, vllm_config=object(), reload_weights=Mock()
    )
    ext.state_dict_info = {}
    ext.model_update_group = object()
    ext._prepare_reload_weight_iterator = lambda weights: weights
    receiver = Mock(return_value=iter([]))
    monkeypatch.setattr(fp8, "is_fp8_model", lambda _config: fp8_enabled)
    monkeypatch.setattr(vllm_backend, "packed_broadcast_consumer", receiver)
    monkeypatch.setattr(vllm_backend.torch.cuda, "empty_cache", lambda: None)

    if model_type == "deepseek_v4" and fp8_enabled:
        with pytest.raises(RuntimeError, match="Set refit_with_reload_api=False"):
            ext._update_weights_from_collective(refit_with_reload_api=True)
        receiver.assert_not_called()
        ext.model_runner.reload_weights.assert_not_called()
    else:
        assert ext._update_weights_from_collective(refit_with_reload_api=True)
        receiver.assert_called_once()
        ext.model_runner.reload_weights.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v3"])
@pytest.mark.parametrize("fp8_enabled", [False, True])
async def test_checkpoint_engine_refit_guard(monkeypatch, model_type, fp8_enabled):
    from nemo_rl.models.generation.vllm import vllm_backend
    from nemo_rl.models.generation.vllm.quantization import fp8

    ext = vllm_backend.VllmInternalWorkerExtensionWithCheckpointEngine.__new__(
        vllm_backend.VllmInternalWorkerExtensionWithCheckpointEngine
    )
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type=model_type)
    ext.model_runner = SimpleNamespace(model=model, vllm_config=object())
    ext._uses_unquantized_flashinfer_trtllm = lambda: False
    ext._maybe_process_fp8_kv_cache = lambda: None

    async def empty_batches():
        if False:
            yield

    receiver = Mock(side_effect=empty_batches)
    ext.checkpoint_engine = SimpleNamespace(receive_weight_batches=receiver)
    monkeypatch.setattr(fp8, "is_fp8_model", lambda _config: fp8_enabled)

    if model_type == "deepseek_v4" and fp8_enabled:
        with pytest.raises(RuntimeError, match="checkpoint-engine.*DeepSeek V4 FP8"):
            await ext._update_weights_from_checkpoint_engine_async()
        receiver.assert_not_called()
    else:
        assert await ext._update_weights_from_checkpoint_engine_async()
        receiver.assert_called_once()


@pytest.mark.parametrize("transport", ["ipc", "collective"])
def test_deepseek_v4_fp8_selects_supported_native_refit(monkeypatch, transport):
    from nemo_rl.models.generation.vllm import vllm_backend
    from nemo_rl.models.generation.vllm.quantization import fp8

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="deepseek_v4")
    ext.model_runner = SimpleNamespace(model=model, vllm_config=object())
    ext._uses_unquantized_flashinfer_trtllm = lambda: False
    monkeypatch.setattr(fp8, "is_fp8_model", lambda _config: True)

    assert ext._uses_native_layerwise_refit(transport)
    ext._validate_native_layerwise_refit(transport)


def test_weight_update_lifecycle_uses_layerwise_reload_for_deepseek_v4_fp8(
    monkeypatch,
):
    """DeepSeek V4 FP8 refits stream layerwise instead of a full post-load pass."""
    import vllm.config
    from vllm.model_executor.model_loader import reload as vllm_reload

    from nemo_rl.models.generation.vllm import vllm_backend
    from nemo_rl.models.generation.vllm.quantization import deepseek_v4_fp8, fp8

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    model = torch.nn.Module()
    ext.model_runner = SimpleNamespace(model=model, vllm_config=object())
    ext.model_config = object()
    ext.device = "cpu"
    ext._uses_native_layerwise_refit = lambda _transport: True
    ext._validate_native_layerwise_refit = lambda _transport: None
    ext._uses_deepseek_v4_fp8_refit = lambda: True
    ext._nrl_layerwise_reload_failure = None
    call_order = []

    monkeypatch.setattr(fp8, "is_fp8_model", lambda _config: True)
    monkeypatch.setattr(deepseek_v4_fp8, "is_model", lambda _model: True)
    monkeypatch.setattr(
        deepseek_v4_fp8,
        "prepare_refit",
        lambda arg: (call_order.append(("prepare_refit", arg)), {"attn_sink"})[1],
    )
    monkeypatch.setattr(
        deepseek_v4_fp8,
        "finalize_refit",
        lambda arg: call_order.append(("finalize_refit", arg)),
    )
    monkeypatch.setattr(
        deepseek_v4_fp8,
        "restore_refit",
        lambda added: call_order.append(("restore_refit", added)),
    )
    monkeypatch.setattr(
        vllm_reload,
        "initialize_layerwise_reload",
        lambda arg: call_order.append(("init_reload", arg)),
    )
    monkeypatch.setattr(
        vllm_reload,
        "finalize_layerwise_reload",
        lambda arg, _config: call_order.append(("finalize_reload", arg)),
    )
    monkeypatch.setattr(
        vllm.config,
        "set_current_vllm_config",
        lambda _config: contextlib.nullcontext(),
    )
    ext._maybe_process_mtp_drafter_after_loading = lambda: call_order.append(
        ("mtp", None)
    )
    monkeypatch.setattr(
        vllm_backend,
        "_refresh_hpc_modules_after_layerwise_reload",
        lambda arg: call_order.append(("hpc", arg)),
    )
    monkeypatch.setattr(
        vllm_backend.torch.cuda,
        "synchronize",
        lambda: call_order.append(("sync", None)),
    )

    with ext._weight_update_lifecycle("collective") as finalize:
        call_order.append(("stream", None))
        finalize()

    assert call_order == [
        ("prepare_refit", model),
        ("init_reload", model),
        ("stream", None),
        ("finalize_reload", model),
        ("finalize_refit", model),
        ("hpc", model),
        ("mtp", None),
        ("sync", None),
        ("restore_refit", {"attn_sink"}),
    ]


def test_deepseek_v4_layerwise_failure_restores_global_state(monkeypatch):
    import vllm.config
    from vllm.model_executor.model_loader import reload as vllm_reload

    from nemo_rl.models.generation.vllm import vllm_backend
    from nemo_rl.models.generation.vllm.quantization import deepseek_v4_fp8

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(model=torch.nn.Module(), vllm_config=object())
    ext.model_config = object()
    ext.device = "cpu"
    ext._uses_native_layerwise_refit = lambda _transport: True
    ext._validate_native_layerwise_refit = lambda _transport: None
    ext._uses_deepseek_v4_fp8_refit = lambda: True
    ext._nrl_layerwise_reload_failure = None
    restored = []

    monkeypatch.setattr(
        vllm.config,
        "set_current_vllm_config",
        lambda _config: contextlib.nullcontext(),
    )
    monkeypatch.setattr(vllm_reload, "initialize_layerwise_reload", lambda _model: None)
    monkeypatch.setattr(deepseek_v4_fp8, "prepare_refit", lambda _model: {"attn_sink"})
    monkeypatch.setattr(
        deepseek_v4_fp8, "restore_refit", lambda added: restored.append(added)
    )

    failure = RuntimeError("stream failed")
    with pytest.raises(RuntimeError, match="stream failed"):
        with ext._weight_update_lifecycle("collective"):
            raise failure

    assert ext._nrl_layerwise_reload_failure is failure
    assert ext._nrl_layerwise_reload_active is False
    assert restored == [{"attn_sink"}]


@pytest.mark.parametrize("context_name", ["config", "device"])
def test_deepseek_v4_context_entry_failure_preserves_original_error(
    monkeypatch, context_name
):
    import vllm.config

    from nemo_rl.models.generation.vllm import vllm_backend
    from nemo_rl.models.generation.vllm.quantization import deepseek_v4_fp8

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(model=torch.nn.Module(), vllm_config=object())
    ext.device = "cpu"
    ext._uses_native_layerwise_refit = lambda _transport: True
    ext._validate_native_layerwise_refit = lambda _transport: None
    ext._uses_deepseek_v4_fp8_refit = lambda: True
    failure = RuntimeError("context entry failed")
    restored = []

    class FailingContext:
        def __enter__(self):
            raise failure

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        vllm.config,
        "set_current_vllm_config",
        lambda _config: FailingContext()
        if context_name == "config"
        else contextlib.nullcontext(),
    )
    if context_name == "device":
        monkeypatch.setattr(
            vllm_backend.torch, "device", lambda _device: FailingContext()
        )
    monkeypatch.setattr(
        deepseek_v4_fp8, "restore_refit", lambda added: restored.append(added)
    )

    with pytest.raises(RuntimeError, match="context entry failed") as exc_info:
        with ext._weight_update_lifecycle("ipc"):
            pytest.fail("The refit stream must not start after context entry fails")

    assert exc_info.value is failure
    assert ext._nrl_layerwise_reload_failure is failure
    assert ext._nrl_layerwise_reload_active is False
    assert restored == [set()]


def test_weight_update_lifecycle_keeps_full_post_load_for_non_deepseek_models(
    monkeypatch,
):
    import vllm.config
    from vllm.model_executor.model_loader import utils as loader_utils

    from nemo_rl.models.generation.vllm import vllm_backend
    from nemo_rl.models.generation.vllm.quantization import deepseek_v4_fp8, fp8

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    model = torch.nn.Module()
    ext.model_runner = SimpleNamespace(model=model, vllm_config=object())
    ext.model_config = object()
    ext.device = "cpu"
    ext._uses_native_layerwise_refit = lambda _transport: False
    call_order = []

    monkeypatch.setattr(fp8, "is_fp8_model", lambda _config: True)
    monkeypatch.setattr(deepseek_v4_fp8, "is_model", lambda _model: False)
    monkeypatch.setattr(
        deepseek_v4_fp8,
        "prepare_refit",
        lambda _model: call_order.append(("prepare_refit", None)),
    )
    monkeypatch.setattr(
        loader_utils,
        "process_weights_after_loading",
        lambda *_args: call_order.append(("post_load", None)),
    )
    monkeypatch.setattr(
        vllm.config,
        "set_current_vllm_config",
        lambda _config: contextlib.nullcontext(),
    )
    ext._maybe_process_mtp_drafter_after_loading = lambda: call_order.append(
        ("mtp", None)
    )
    ext._maybe_process_fp8_kv_cache = lambda: call_order.append(("kv", None))

    with ext._weight_update_lifecycle("collective") as finalize:
        call_order.append(("stream", None))
        finalize()

    assert call_order == [
        ("stream", None),
        ("post_load", None),
        ("mtp", None),
        ("kv", None),
    ]
