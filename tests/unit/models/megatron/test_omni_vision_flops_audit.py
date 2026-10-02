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

"""Independent Omni vision matrix/convolution FLOPs, without downloading weights.

These fixtures exclude elementwise operations and activation recomputation.
Raw image pixels have no gradients; patch convolution therefore has a weight
backward pass, but no input backward pass. Every encoder parameter is trainable.
"""

import pytest
import torch
from types import SimpleNamespace
from torch.utils.flop_counter import FlopCounterMode
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniVisionEncoderConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniVisionEncoder,
)


# Two layers: h=8, MLP=16, two heads, patch=2x2x1, RGB, merge=2x2,
# merger output=16. Each layer has four attention projections and THREE
# gated-MLP projections. A 4x4 attention window holds sixteen patches.
# For the 8x8 mixed fixture, forward work is:
# projections/MLPs 163840 + attention 163840 + merger 49152 + conv 12288.
# Backward doubles all but conv (pixels need no gradient): 765952.
CASES = [
    pytest.param(4, True, 72704, 142336, id="one-window-full"),
    pytest.param(4, False, 72704, 142336, id="one-window-mixed"),
    pytest.param(8, True, 487424, 962560, id="four-windows-full"),
    pytest.param(8, False, 389120, 765952, id="four-windows-mixed"),
]


def _vision_config(full_attention):
    config = Qwen2_5OmniVisionEncoderConfig(
        depth=2,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=1,
        in_channels=3,
        spatial_merge_size=2,
        out_hidden_size=16,
        window_size=8,
        fullatt_block_indexes=[0, 1] if full_attention else [1],
    )
    # Eager attention exposes matmuls to the independent PyTorch counter.
    config._attn_implementation = "eager"
    return config


@pytest.mark.parametrize("side,full_attention,forward_flops,backward_flops", CASES)
def test_omni_vision_reference_matches_executed_operators(
    side, full_attention, forward_flops, backward_flops
):
    """Catch wrong MLP arity, attention boundaries, or patch backward counts."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        encoder = Qwen2_5OmniVisionEncoder(_vision_config(full_attention)).train()
        pixels = torch.randn(side * side, 12)
        grid = torch.tensor([[1, side, side]])
        with FlopCounterMode(display=False) as forward:
            output = encoder(pixels, grid_thw=grid).pooler_output
        with FlopCounterMode(display=False) as backward:
            output.sum().backward()

    assert forward.get_total_flops() == forward_flops
    assert backward.get_total_flops() == backward_flops
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize("side,full_attention,forward_flops,backward_flops", CASES)
def test_omni_vision_estimate_matches_independent_training_work(
    side, full_attention, forward_flops, backward_flops
):
    """The estimator used by NeMo-RL must count the actual Omni architecture."""
    from nemo_rl.models.megatron.omni_flops import qwen25_omni_vision_flops

    grid = torch.tensor([[1, side, side]])
    assert qwen25_omni_vision_flops(_vision_config(full_attention), grid) == (
        forward_flops + backward_flops
    )


@pytest.mark.mcore
@pytest.mark.parametrize("side,full_attention,forward_flops,backward_flops", CASES)
def test_pinned_bridge_generic_vision_estimate_needs_omni_correction(
    side, full_attention, forward_flops, backward_flops
):
    """Keep the upstream comparison explicit until Bridge supports Omni vision."""
    from megatron.bridge.training.utils.flop_utils import vit_flops_from_grid_thw

    config = SimpleNamespace(
        model=SimpleNamespace(
            hidden_size=16, vision_config=_vision_config(full_attention)
        )
    )
    upstream = float(vit_flops_from_grid_thw(config, torch.tensor([[1, side, side]])))
    # The pinned generic formula counts two MLP projections, full attention in
    # every layer, and no patch convolution. It ignores fullatt_block_indexes.
    assert upstream == (184320 if side == 4 else 1327104)
    assert upstream != forward_flops + backward_flops


@pytest.mark.parametrize(
    "grids",
    [
        [[1, 6, 10]],
        [[2, 6, 4]],
        [[1, 4, 4], [1, 6, 10]],
    ],
)
def test_omni_edge_windows_and_media_boundaries_match_operators(grids):
    from nemo_rl.models.megatron.omni_flops import qwen25_omni_vision_flops

    config = _vision_config(False)
    grid = torch.tensor(grids)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        encoder = Qwen2_5OmniVisionEncoder(config).train()
        pixels = torch.randn(sum(t * h * w for t, h, w in grids), 12)
        with FlopCounterMode(display=False) as counter:
            encoder(pixels, grid_thw=grid).pooler_output.sum().backward()
    assert qwen25_omni_vision_flops(config, grid) == counter.get_total_flops()


@pytest.mark.parametrize(
    "grid",
    [
        torch.tensor([1, 4, 4]),
        torch.tensor([[1.0, 4.0, 4.0]]),
        torch.tensor([[0, 4, 4]]),
        torch.tensor([[1, 3, 4]]),
        torch.tensor([[1, -4, 4]]),
    ],
)
def test_omni_malformed_grids_are_errors(grid):
    from nemo_rl.models.megatron.omni_flops import qwen25_omni_vision_flops

    with pytest.raises(ValueError):
        qwen25_omni_vision_flops(_vision_config(False), grid)


def test_omni_empty_media_has_no_encoder_work():
    from nemo_rl.models.megatron.omni_flops import qwen25_omni_vision_flops

    assert (
        qwen25_omni_vision_flops(
            _vision_config(False), torch.empty((0, 3), dtype=torch.int64)
        )
        == 0
    )
