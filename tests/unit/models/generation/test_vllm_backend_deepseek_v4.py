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

import pytest

pytestmark = pytest.mark.vllm


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
    model = object()
    ext.model_runner = SimpleNamespace(model=model, vllm_config=object())
    ext.model_config = object()
    ext.device = "cpu"
    ext._uses_native_layerwise_refit = lambda _transport: True
    ext._validate_native_layerwise_refit = lambda: None
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
    ext.model_runner = SimpleNamespace(model=object(), vllm_config=object())
    ext.model_config = object()
    ext.device = "cpu"
    ext._uses_native_layerwise_refit = lambda _transport: True
    ext._validate_native_layerwise_refit = lambda: None
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
    model = object()
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
