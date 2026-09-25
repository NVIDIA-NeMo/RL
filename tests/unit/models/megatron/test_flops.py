# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.flops import compute_bridge_batch_flops


@pytest.fixture
def calculator(monkeypatch):
    # CPU tests exercise our adapter without importing the GPU-only Bridge stack.
    module = ModuleType("megatron.bridge.training.utils")
    module.flop_utils = SimpleNamespace(
        num_floating_point_operations=MagicMock(return_value=120.0),
        vit_flops_from_grid_thw=MagicMock(return_value=torch.tensor(30.0)),
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.flop_utils


def _batch():
    return BatchedDataDict(
        {
            "input_ids": torch.zeros(2, 16, dtype=torch.long),
            "input_lengths": torch.tensor([7, 11]),
            "token_mask": torch.zeros(2, 16),
            "sample_mask": torch.ones(2),
        }
    )


def _config():
    return SimpleNamespace(peft=None, model=SimpleNamespace(seq_length=4096))


def test_real_lengths_not_context_limit_padding_or_loss_mask(calculator):
    config = _config()
    assert compute_bridge_batch_flops(config, _batch()) == 120
    calculator.num_floating_point_operations.assert_called_once_with(
        config,
        batch_size=2,
        seqlen_sum=18,
        seqlen_squared_sum=170,
        num_vision_patches=0,
    )
    assert config.model.seq_length == 4096
    calculator.vit_flops_from_grid_thw.assert_not_called()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_images_and_videos_are_counted_once(calculator, packed, nested):
    config, data = _config(), _batch()
    vision = SimpleNamespace(spatial_merge_size=2)
    if nested:
        config.model.thinker_config = SimpleNamespace(vision_config=vision)
    else:
        config.model.vision_config = vision
    images = torch.tensor([[1, 4, 4], [1, 6, 6]])
    videos = torch.tensor([[3, 8, 8]])
    data["image_grid_thw"] = PackedTensor([images], dim_to_pack=0) if packed else images
    data["video_grid_thw"] = PackedTensor([videos], dim_to_pack=0) if packed else videos
    assert compute_bridge_batch_flops(config, data) == 180
    assert calculator.vit_flops_from_grid_thw.call_count == 2
    assert torch.equal(
        calculator.vit_flops_from_grid_thw.call_args_list[0].args[1], images
    )
    assert torch.equal(
        calculator.vit_flops_from_grid_thw.call_args_list[1].args[1], videos
    )


@pytest.mark.parametrize(
    "flag", ["freeze_language_model", "freeze_vision_model", "freeze_vision_projection"]
)
@pytest.mark.parametrize("source", ["provider", "runtime"])
@pytest.mark.parametrize("grid_key", [None, "image_grid_thw", "video_grid_thw"])
def test_frozen_model_fallback_depends_on_active_modules(
    calculator, flag, source, grid_key
):
    config, data = _config(), _batch()
    options = {}
    if source == "provider":
        setattr(config.model, flag, True)
    else:
        options["freeze_config"] = {flag: True}
    if grid_key is not None:
        data[grid_key] = torch.tensor([[1, 4, 4]])
    if flag == "freeze_language_model" or grid_key is not None:
        with pytest.raises(NotImplementedError, match="frozen"):
            compute_bridge_batch_flops(config, data, **options)
        calculator.num_floating_point_operations.assert_not_called()
    else:
        assert compute_bridge_batch_flops(config, data, **options) == 120


def test_false_freeze_flags_and_empty_media_keep_text_estimate(calculator):
    config, data = _config(), _batch()
    config.model.freeze_vision_model = True
    data["image_grid_thw"] = PackedTensor([None, None], dim_to_pack=0)
    data["video_grid_thw"] = torch.empty(0, 3, dtype=torch.long)
    assert (
        compute_bridge_batch_flops(
            config, data, freeze_config={"freeze_language_model": False}
        )
        == 120
    )


def _packed_batch():
    data = _batch()
    data["input_ids"] = torch.zeros(2, 24, dtype=torch.long)
    data["input_lengths"] = torch.tensor([24, 8])
    data["cu_seqlens"] = PackedTensor(
        [torch.tensor([0, 7, 18]), torch.tensor([0, 5])], dim_to_pack=0
    )
    data["cu_seqlens_padded"] = PackedTensor(
        [torch.tensor([0, 8, 24]), torch.tensor([0, 8])], dim_to_pack=0
    )
    return data


def test_packed_attention_uses_source_lengths_without_padding(calculator):
    config = _config()
    assert compute_bridge_batch_flops(config, _packed_batch()) == 120
    calculator.num_floating_point_operations.assert_called_once_with(
        config,
        batch_size=3,
        seqlen_sum=23,
        seqlen_squared_sum=195,
        num_vision_patches=0,
    )


@pytest.mark.parametrize("boundaries", [[1, 7], [0, 7, 7], [0, 9, 8], [0, 25], [0]])
def test_malformed_packed_lengths_are_errors(calculator, boundaries):
    data = _packed_batch()
    data["cu_seqlens"] = PackedTensor(
        [torch.tensor(boundaries), torch.tensor([0, 5])], dim_to_pack=0
    )
    with pytest.raises(ValueError, match="cu_seqlens"):
        compute_bridge_batch_flops(_config(), data)


@pytest.mark.parametrize("rows", [[torch.tensor([0, 7])], [None, torch.tensor([0, 5])]])
def test_missing_packed_boundaries_are_errors(calculator, rows):
    data = _packed_batch()
    data["cu_seqlens"] = PackedTensor(rows, dim_to_pack=0)
    with pytest.raises(ValueError, match="cu_seqlens"):
        compute_bridge_batch_flops(_config(), data)


def test_peft_explicitly_requests_fallback(calculator):
    config = _config()
    config.peft = object()
    with pytest.raises(NotImplementedError, match="PEFT"):
        compute_bridge_batch_flops(config, _batch())


@pytest.mark.parametrize("key", ["input_features", "audio_features", "audio_signal"])
def test_audio_explicitly_requests_fallback(calculator, key):
    data = _batch()
    data[key] = torch.zeros(2, 3)
    with pytest.raises(NotImplementedError, match="audio"):
        compute_bridge_batch_flops(_config(), data)
    calculator.num_floating_point_operations.assert_not_called()
    calculator.vit_flops_from_grid_thw.assert_not_called()


def test_media_without_grids_is_not_misreported_as_text_only(calculator):
    data = _batch()
    data["pixel_values"] = torch.zeros(2, 3)
    with pytest.raises(NotImplementedError, match="image_grid_thw"):
        compute_bridge_batch_flops(_config(), data)


def test_text_only_shard_of_vision_batch_needs_no_grid(calculator):
    data = _batch()
    data["pixel_values"] = PackedTensor([None, None], dim_to_pack=0)
    assert compute_bridge_batch_flops(_config(), data) == 120.0


def test_old_provider_estimator_requests_fallback(calculator):
    config = _config()
    config.model._get_num_floating_point_operations = lambda batch_size: 999.0
    with pytest.raises(NotImplementedError, match="runtime lengths"):
        compute_bridge_batch_flops(config, _batch())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -10.0])
def test_invalid_estimates_are_errors(calculator, value):
    calculator.num_floating_point_operations.return_value = value
    with pytest.raises(ValueError, match="invalid training FLOPs"):
        compute_bridge_batch_flops(_config(), _batch())


def test_unexpected_calculator_error_propagates(calculator):
    calculator.num_floating_point_operations.side_effect = RuntimeError(
        "broken estimator"
    )
    with pytest.raises(RuntimeError, match="broken estimator"):
        compute_bridge_batch_flops(_config(), _batch())


def test_invalid_lengths_are_errors(calculator):
    data = _batch()
    data["input_lengths"] = torch.tensor([0, 11])
    with pytest.raises(ValueError, match="positive length"):
        compute_bridge_batch_flops(_config(), data)


def _real_config():
    return SimpleNamespace(
        peft=None,
        model=SimpleNamespace(
            hidden_size=64,
            num_layers=3,
            seq_length=4096,
            ffn_hidden_size=128,
            num_attention_heads=8,
            num_query_groups=2,
            kv_channels=8,
            vocab_size=256,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            num_moe_experts=None,
            mtp_num_layers=None,
            moe_ffn_hidden_size=None,
            moe_shared_expert_intermediate_size=None,
            gated_linear_unit=True,
            multi_latent_attention=False,
            activation_func=torch.nn.functional.silu,
        ),
    )


@pytest.mark.mcore
@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_real_bridge_matches_backend_agnostic_text_formula(family):
    pytest.importorskip("megatron.bridge.training.utils.flop_utils")
    from transformers import LlamaConfig, Qwen3Config

    from nemo_rl.utils.flops_tracker import FLOPTracker

    config_cls = LlamaConfig if family == "llama" else Qwen3Config
    config = config_cls(
        hidden_size=64,
        num_hidden_layers=3,
        intermediate_size=128,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=256,
    )
    tracker = FLOPTracker.from_config(family, config)
    tracker.track_batch([7, 11])
    assert compute_bridge_batch_flops(_real_config(), _batch()) == pytest.approx(
        tracker.total_flops
    )


@pytest.mark.mcore
@pytest.mark.parametrize("nested", [False, True])
def test_real_bridge_adds_exact_variable_image_work(nested):
    pytest.importorskip("megatron.bridge.training.utils.flop_utils")
    config, data = _real_config(), _batch()
    decoder = compute_bridge_batch_flops(config, data)
    config.model.vision_config = SimpleNamespace(
        depth=2,
        hidden_size=8,
        intermediate_size=24,
        spatial_merge_size=2,
        out_hidden_size=64,
    )
    if nested:
        config.model.thinker_config = SimpleNamespace(
            vision_config=config.model.vision_config
        )
        del config.model.vision_config
    data["image_grid_thw"] = torch.tensor([[1, 4, 4], [1, 6, 6]])
    # 52 patches; attention stays within each image: 16^2 + 36^2 = 1552.
    # 13 merged tokens. Forward encoder + merger, times 3 for training.
    vision = 3 * (
        2 * ((8 * 8**2 + 4 * 8 * 24) * 52 + 4 * 8 * 1552)
        + 13 * (2 * 32**2 + 2 * 32 * 64)
    )
    assert compute_bridge_batch_flops(config, data) == pytest.approx(decoder + vision)


@pytest.mark.mcore
@pytest.mark.parametrize("family", ["qwen3", "qwen25_omni"])
@pytest.mark.parametrize("lengths", [[607, 607], [220, 236]])
def test_model_flops_match_megatron_lm_and_bridge_vision(family, lengths):
    """Compare real upstream calculators without loading weights or downloading models."""
    from megatron.bridge.training.utils.flop_utils import vit_flops_from_grid_thw
    from megatron.training.training import num_floating_point_operations
    from transformers import Qwen2_5OmniConfig, Qwen3Config

    hf = Qwen3Config() if family == "qwen3" else Qwen2_5OmniConfig()
    text = hf if family == "qwen3" else hf.thinker_config.text_config
    config = _real_config()
    model = config.model
    model.hidden_size = text.hidden_size
    model.num_layers = text.num_hidden_layers
    model.ffn_hidden_size = text.intermediate_size
    model.num_attention_heads = text.num_attention_heads
    model.num_query_groups = text.num_key_value_heads
    model.kv_channels = getattr(text, "head_dim", None) or (
        text.hidden_size // text.num_attention_heads
    )
    model.vocab_size = text.vocab_size
    args = SimpleNamespace(
        **vars(model),
        group_query_attention=True,
        decoder_seq_length=None,
        num_experts=None,
        moe_latent_size=None,
        swiglu=True,
        attention_output_gate=False,
        experimental_attention_variant=None,
        hybrid_layer_pattern=None,
        padded_vocab_size=(text.vocab_size + 127) // 128 * 128,
    )
    data = _batch()
    data["input_lengths"] = torch.tensor(lengths)
    data["input_ids"] = torch.zeros(2, max(lengths), dtype=torch.long)
    data["token_mask"] = torch.zeros_like(data["input_ids"])
    decoder = num_floating_point_operations(
        args,
        batch_size=2,
        total_real_tokens_in_batch=sum(lengths),
        seqlen_squared_sum_in_batch=sum(length**2 for length in lengths),
    )
    assert compute_bridge_batch_flops(config, data) == pytest.approx(decoder, rel=1e-12)
    if family == "qwen25_omni":
        # Megatron-LM's training calculator covers the decoder, not vision.
        model.thinker_config = hf.thinker_config
        grids = torch.tensor([[1, 22, 34], [1, 16, 24]])
        data["image_grid_thw"] = PackedTensor([grids[:1], grids[1:]], dim_to_pack=0)
        vision = float(vit_flops_from_grid_thw(config, grids))
        assert vision > 0
        assert compute_bridge_batch_flops(config, data) == pytest.approx(
            decoder + vision, rel=1e-12
        )
