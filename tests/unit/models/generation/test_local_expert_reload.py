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

import inspect

import pytest
import torch

from nemo_rl.models.generation.vllm.local_expert_reload import copy_local_bf16_expert


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_local_projection_replay(gated: bool, padded: bool, reverse: bool) -> None:
    width, hidden, experts = 8, 16, 3
    padded_width, padded_hidden = (12, 20) if padded else (width, hidden)
    w13 = torch.full(
        (experts, padded_width * (2 if gated else 1), padded_hidden),
        -1,
        dtype=torch.bfloat16,
    )
    w2 = torch.full((experts, padded_hidden, padded_width), -1, dtype=torch.bfloat16)
    projections = (
        ["gate_proj", "up_proj", "down_proj"] if gated else ["up_proj", "down_proj"]
    )
    if reverse:
        projections.reverse()
    for update in (1, 2, 3):
        for projection in projections:
            is_down = projection == "down_proj"
            shape = (experts, hidden, width) if is_down else (experts, width, hidden)
            value = (
                update * 10 + {"gate_proj": 1, "up_proj": 2, "down_proj": 3}[projection]
            )
            weight = torch.full(shape, value, dtype=torch.bfloat16)
            copy_local_bf16_expert(
                w2 if is_down else w13, weight, projection=projection, gated=gated
            )
        expected13 = torch.zeros_like(w13)
        if gated:
            expected13[:, :width, :hidden] = update * 10 + 1
        start = padded_width if gated else 0
        expected13[:, start : start + width, :hidden] = update * 10 + 2
        expected2 = torch.zeros_like(w2)
        expected2[:, :hidden, :width] = update * 10 + 3
        torch.testing.assert_close(w13, expected13, rtol=0, atol=0)
        torch.testing.assert_close(w2, expected2, rtol=0, atol=0)


def test_invalid_payload_does_not_modify_destination() -> None:
    target = torch.full((2, 16, 8), 7, dtype=torch.bfloat16)
    before = target.clone()
    with pytest.raises(ValueError, match="exceeds"):
        copy_local_bf16_expert(
            target,
            torch.ones((2, 9, 8), dtype=torch.bfloat16),
            projection="up_proj",
            gated=True,
        )
    torch.testing.assert_close(target, before, rtol=0, atol=0)


def test_meta_pass_does_not_skip_real_padding_initialization() -> None:
    meta = torch.empty((2, 24, 20), dtype=torch.bfloat16, device="meta")
    payload = torch.ones((2, 8, 16), dtype=torch.bfloat16)
    copy_local_bf16_expert(meta, payload, projection="up_proj", gated=True)
    actual = torch.full((2, 24, 20), -1, dtype=torch.bfloat16)
    copy_local_bf16_expert(actual, payload, projection="up_proj", gated=True)
    assert torch.all(actual[:, :12] == -1)
    assert torch.all(actual[:, 20:] == 0)
    assert torch.all(actual[:, 12:20, 16:] == 0)


def test_vllm_counter_counts_only_payload() -> None:
    pytest.importorskip("vllm")
    from vllm.model_executor.model_loader.reload.meta import get_numel_loaded

    meta = torch.empty((2, 24, 20), dtype=torch.bfloat16, device="meta")
    payload = torch.ones((2, 8, 16), dtype=torch.bfloat16)
    bound = inspect.signature(copy_local_bf16_expert).bind(
        meta, payload, projection="up_proj", gated=True
    )
    count, _ = get_numel_loaded(copy_local_bf16_expert, bound)
    assert count == payload.numel()
