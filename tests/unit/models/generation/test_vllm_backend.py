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
from unittest.mock import MagicMock

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
    ext.model_runner = SimpleNamespace(model=object(), vllm_config=object())
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
    ext.model_runner.drafter.model = SimpleNamespace(model=predictor)
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


@pytest.mark.vllm
def test_update_weights_from_collective_processes_weights_after_loading(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    call_order = []
    process_calls = []

    def process_weights_after_loading(model, model_config, device):
        call_order.append("process")
        process_calls.append((model, model_config, device))

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        process_weights_after_loading,
    )
    ext, expected_state_info = _make_collective_update_extension(vllm_backend)

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

    def packed_broadcast_consumer(iterator, group, src, post_unpack_func):
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

    assert process_calls == [(ext.model_runner.model, ext.model_config, ext.device)]
    assert call_order == [
        "broadcast",
        "load",
        "config_enter",
        "process",
        "config_exit",
        "kv",
        "gc",
        "empty_cache",
    ]


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
def test_load_mtp_weights_from_disk_returns_false_without_drafter(tmp_path):
    """When vLLM has not built a drafter, the load is skipped (no exception)."""
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.device = torch.device("cpu")
    ext.model_runner = MagicMock()
    ext.model_runner.drafter = None
    ext._load_draft_weights = MagicMock()

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


class _TinyRadioVisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(initializer_factor=1.0)
        self.ls1 = torch.nn.Parameter(torch.full((4,), 0.001))
        self.ls2 = torch.nn.Parameter(torch.full((4,), -0.001))
        self.projection = torch.nn.Parameter(torch.full((4,), 0.25))


def _make_extension_with_radio(architecture):
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.model_runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(architectures=[architecture])
        ),
        model=SimpleNamespace(vision_model=_TinyRadioVisionModel()),
    )
    return ext


@pytest.mark.vllm
def test_initialize_folded_nemotron_radio_layerscale_while_awake():
    ext = _make_extension_with_radio("NemotronH_Nano_Omni_Reasoning_V3")

    initialized = ext._initialize_nemotron_omni_radio_layerscale()

    vision_model = ext.model_runner.model.vision_model
    assert initialized == 2
    assert torch.equal(vision_model.ls1, torch.ones_like(vision_model.ls1))
    assert torch.equal(vision_model.ls2, torch.ones_like(vision_model.ls2))
    assert torch.equal(
        vision_model.projection, torch.full_like(vision_model.projection, 0.25)
    )


@pytest.mark.vllm
def test_prepare_refit_info_does_not_mutate_folded_layerscale():
    ext = _make_extension_with_radio("NemotronH_Nano_Omni_Reasoning_V3")

    state_dict_info = {"language_model.weight": ((4, 4), torch.bfloat16)}
    ext.prepare_refit_info(state_dict_info)

    vision_model = ext.model_runner.model.vision_model
    assert ext.state_dict_info is state_dict_info
    assert torch.equal(vision_model.ls1, torch.full_like(vision_model.ls1, 0.001))
    assert torch.equal(vision_model.ls2, torch.full_like(vision_model.ls2, -0.001))


@pytest.mark.vllm
def test_prepare_refit_info_rejects_explicit_nemotron_radio_layerscale():
    ext = _make_extension_with_radio("NemotronH_Super_Omni_Reasoning_V3")

    with pytest.raises(RuntimeError, match="explicit RADIO LayerScale"):
        ext.prepare_refit_info(
            {
                "vision_model.radio_model.model.blocks.0.ls1": (
                    (4,),
                    torch.bfloat16,
                )
            }
        )


@pytest.mark.vllm
def test_prepare_refit_info_leaves_non_nemotron_model_unchanged():
    ext = _make_extension_with_radio("SomeOtherArchitecture")

    ext.prepare_refit_info({})
    assert ext._initialize_nemotron_omni_radio_layerscale() == 0

    vision_model = ext.model_runner.model.vision_model
    assert torch.equal(vision_model.ls1, torch.full_like(vision_model.ls1, 0.001))
    assert torch.equal(vision_model.ls2, torch.full_like(vision_model.ls2, -0.001))
