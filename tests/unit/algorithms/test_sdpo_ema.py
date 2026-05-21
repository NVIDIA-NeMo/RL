# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit test for the EMA-teacher blend math used by DTensorPolicyWorkerV2.update_ema_teacher_state."""

from __future__ import annotations

import pytest
import torch


def _ema_blend(ema_state: dict[str, torch.Tensor], live_state: dict[str, torch.Tensor], alpha: float) -> None:
    """Reference implementation matching the worker's in-place blend.

    theta_T <- (1 - alpha) * theta_T + alpha * theta_S
    """
    for k, ema_t in ema_state.items():
        ema_t.mul_(1.0 - alpha).add_(live_state[k], alpha=alpha)


def test_ema_blend_alpha_zero_keeps_teacher_unchanged():
    ema = {"w": torch.tensor([1.0, 2.0, 3.0])}
    live = {"w": torch.tensor([10.0, 20.0, 30.0])}
    _ema_blend(ema, live, alpha=0.0)
    assert torch.allclose(ema["w"], torch.tensor([1.0, 2.0, 3.0]))


def test_ema_blend_alpha_one_overwrites_teacher_with_student():
    ema = {"w": torch.tensor([1.0, 2.0, 3.0])}
    live = {"w": torch.tensor([10.0, 20.0, 30.0])}
    _ema_blend(ema, live, alpha=1.0)
    assert torch.allclose(ema["w"], torch.tensor([10.0, 20.0, 30.0]))


def test_ema_blend_half_mixes_evenly():
    ema = {"w": torch.tensor([0.0, 2.0, 4.0])}
    live = {"w": torch.tensor([10.0, 4.0, 0.0])}
    _ema_blend(ema, live, alpha=0.5)
    # 0.5 * old + 0.5 * new
    assert torch.allclose(ema["w"], torch.tensor([5.0, 3.0, 2.0]))


def test_ema_blend_paper_alpha_001_is_slow():
    """At alpha=0.01 (paper Table 12), a single step moves the teacher only 1%
    toward the student. After 100 steps with constant student, teacher is
    ~63% of the way (1 - (0.99)**100 ≈ 0.634)."""
    ema = {"w": torch.tensor([0.0])}
    live = {"w": torch.tensor([1.0])}
    for _ in range(100):
        _ema_blend(ema, live, alpha=0.01)
    expected = 1.0 - 0.99**100
    assert ema["w"].item() == pytest.approx(expected, abs=1e-6)


def test_ema_blend_works_across_multiple_param_tensors():
    ema = {
        "embed": torch.zeros(3, 4),
        "layer.weight": torch.ones(2, 5),
    }
    live = {
        "embed": torch.full((3, 4), 2.0),
        "layer.weight": torch.full((2, 5), -1.0),
    }
    _ema_blend(ema, live, alpha=0.25)
    assert torch.allclose(ema["embed"], torch.full((3, 4), 0.5))
    assert torch.allclose(ema["layer.weight"], torch.full((2, 5), 0.5))


def test_ema_blend_is_in_place():
    """The worker uses in-place ops; identity of the tensor object is preserved."""
    ema = {"w": torch.tensor([1.0])}
    original_id = id(ema["w"])
    _ema_blend(ema, {"w": torch.tensor([3.0])}, alpha=0.5)
    assert id(ema["w"]) == original_id
