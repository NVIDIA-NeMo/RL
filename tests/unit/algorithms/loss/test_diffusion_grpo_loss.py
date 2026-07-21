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
import math

import pytest
import torch

from nemo_rl.algorithms.loss.diffusion_grpo import DiffusionGRPOLossFn
from nemo_rl.models.diffusion.interfaces import DiffusionLossConfig


def _cfg(beta: float = 0.0, adv_clip_max: float | None = 5.0) -> DiffusionLossConfig:
    return DiffusionLossConfig(
        ratio_clip_min=0.2,
        ratio_clip_max=0.2,
        adv_clip_max=adv_clip_max,
        beta=beta,
    )


def test_zero_advantage_yields_zero_policy_loss():
    fn = DiffusionGRPOLossFn(_cfg())
    B, T = 2, 4
    curr = torch.zeros(B, T)
    gen = torch.zeros(B, T)
    adv = torch.zeros(B, T)
    timestep_mask = torch.ones(B, T)
    sample_mask = torch.ones(B)

    loss, metrics = fn(curr, gen, adv, timestep_mask, sample_mask)

    assert torch.allclose(loss, torch.tensor(0.0))
    assert torch.allclose(metrics["policy_loss"], torch.tensor(0.0))
    assert torch.allclose(metrics["mean_ratio"], torch.tensor(1.0))
    assert torch.allclose(metrics["approx_kl"], torch.tensor(0.0))


def test_clipped_upper_branch_active_for_negative_advantage_high_ratio():
    fn = DiffusionGRPOLossFn(_cfg())
    B, T = 1, 1
    curr = torch.tensor([[math.log(2.0)]])  # ratio = 2.0
    gen = torch.tensor([[0.0]])
    adv = torch.tensor([[-1.0]])  # negative advantage
    timestep_mask = torch.ones(B, T)
    sample_mask = torch.ones(B)

    loss, metrics = fn(curr, gen, adv, timestep_mask, sample_mask)

    # unclipped = -(-1.0) * 2.0 = 2.0
    # clipped   = -(-1.0) * clamp(2.0, 0.8, 1.2) = 1.2
    # max(unclipped, clipped) = 2.0 → policy_loss = 2.0
    assert torch.allclose(loss, torch.tensor(2.0))
    # ratio - 1 = 1.0 > clip_max (0.2)
    assert torch.allclose(metrics["clipfrac_higher"], torch.tensor(1.0))
    assert torch.allclose(metrics["clipfrac_lower"], torch.tensor(0.0))


def test_kl_term_is_quadratic_in_mean_offset():
    fn = DiffusionGRPOLossFn(_cfg(beta=1.0))
    B, T, C, H, W = 1, 1, 1, 2, 2
    curr = torch.zeros(B, T)
    gen = torch.zeros(B, T)
    adv = torch.zeros(B, T)
    timestep_mask = torch.ones(B, T)
    sample_mask = torch.ones(B)
    current_mean = torch.full((B, T, C, H, W), 0.5)
    reference_mean = torch.zeros(B, T, C, H, W)
    std_dev = torch.full((B, T, 1, 1, 1), 0.5)

    loss, metrics = fn(
        curr,
        gen,
        adv,
        timestep_mask,
        sample_mask,
        current_mean=current_mean,
        reference_mean=reference_mean,
        std_dev=std_dev,
    )

    # kl per spatial element = (0.5 - 0)^2 / (2 * 0.5^2) = 0.5
    expected_kl = torch.tensor(0.5)
    assert torch.allclose(metrics["kl_loss"], expected_kl)
    # advantage is zero → policy_loss is zero; loss equals beta * kl
    assert torch.allclose(loss, expected_kl)


def test_kl_term_disabled_when_beta_zero_even_with_means():
    fn = DiffusionGRPOLossFn(_cfg(beta=0.0))
    B, T, C, H, W = 1, 1, 1, 2, 2
    loss, metrics = fn(
        torch.zeros(B, T),
        torch.zeros(B, T),
        torch.zeros(B, T),
        torch.ones(B, T),
        torch.ones(B),
        current_mean=torch.full((B, T, C, H, W), 1.0),
        reference_mean=torch.zeros(B, T, C, H, W),
        std_dev=torch.ones(B, T, 1, 1, 1),
    )
    assert "kl_loss" not in metrics
    assert torch.allclose(loss, torch.tensor(0.0))


def test_timestep_mask_excludes_windowed_steps():
    fn = DiffusionGRPOLossFn(_cfg())
    B, T = 1, 4
    # Only the first two steps are "inside" the window.
    timestep_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    sample_mask = torch.ones(B)
    curr = torch.tensor(
        [[math.log(2.0), math.log(2.0), math.log(10.0), math.log(10.0)]]
    )
    gen = torch.zeros(B, T)
    adv = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])

    loss, metrics = fn(curr, gen, adv, timestep_mask, sample_mask)

    # Inside the window: ratio=2.0, adv=-1 → pg = max(2.0, 1.2) = 2.0
    # Outside the window: ignored entirely.
    assert torch.allclose(loss, torch.tensor(2.0))
    # mean_ratio counts only masked-in entries (both equal 2.0)
    assert torch.allclose(metrics["mean_ratio"], torch.tensor(2.0))


def test_sample_mask_zeros_out_entire_trajectory():
    fn = DiffusionGRPOLossFn(_cfg())
    B, T = 2, 3
    timestep_mask = torch.ones(B, T)
    # Sample 0 contributes, sample 1 is masked out entirely.
    sample_mask = torch.tensor([1.0, 0.0])
    curr = torch.zeros(B, T)
    gen = torch.zeros(B, T)
    adv = torch.tensor([[1.0, 1.0, 1.0], [-100.0, -100.0, -100.0]])

    loss, _ = fn(curr, gen, adv, timestep_mask, sample_mask)

    # Sample-1 advantages are completely ignored. Sample-0: adv=1, ratio=1 →
    # pg = max(-1, -1) = -1.
    assert torch.allclose(loss, torch.tensor(-1.0))


def test_advantage_clip_max_applied_before_ratio():
    fn = DiffusionGRPOLossFn(_cfg(adv_clip_max=2.0))
    B, T = 1, 1
    curr = torch.zeros(B, T)
    gen = torch.zeros(B, T)
    adv = torch.tensor([[10.0]])  # clamped to 2.0
    timestep_mask = torch.ones(B, T)
    sample_mask = torch.ones(B)

    loss, _ = fn(curr, gen, adv, timestep_mask, sample_mask)

    # ratio=1, clipped adv=2 → pg = max(-2, -2) = -2.
    assert torch.allclose(loss, torch.tensor(-2.0))


def test_advantage_clip_symmetric_for_negative_advantage():
    fn = DiffusionGRPOLossFn(_cfg(adv_clip_max=2.0))
    B, T = 1, 1
    curr = torch.zeros(B, T)
    gen = torch.zeros(B, T)
    adv = torch.tensor([[-10.0]])  # clamped to -2.0
    timestep_mask = torch.ones(B, T)
    sample_mask = torch.ones(B)

    loss, _ = fn(curr, gen, adv, timestep_mask, sample_mask)

    # ratio=1, clipped adv=-2 → pg = max(2, 2) = 2.
    assert torch.allclose(loss, torch.tensor(2.0))


def test_advantage_clip_disabled_when_none():
    fn = DiffusionGRPOLossFn(_cfg(adv_clip_max=None))
    B, T = 1, 1
    curr = torch.zeros(B, T)
    gen = torch.zeros(B, T)
    adv = torch.tensor([[10.0]])  # not clamped
    timestep_mask = torch.ones(B, T)
    sample_mask = torch.ones(B)

    loss, _ = fn(curr, gen, adv, timestep_mask, sample_mask)

    # ratio=1, adv stays 10 → pg = -10.
    assert torch.allclose(loss, torch.tensor(-10.0))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_loss_is_differentiable_in_curr_logprob(seed: int):
    torch.manual_seed(seed)
    fn = DiffusionGRPOLossFn(_cfg(beta=0.1))
    B, T, C, H, W = 2, 3, 1, 2, 2
    curr = torch.randn(B, T, requires_grad=True)
    gen = torch.randn(B, T)
    adv = torch.randn(B, T)
    timestep_mask = torch.ones(B, T)
    sample_mask = torch.ones(B)
    current_mean = torch.randn(B, T, C, H, W, requires_grad=True)
    reference_mean = torch.randn(B, T, C, H, W)
    std_dev = torch.ones(B, T, 1, 1, 1)

    loss, _ = fn(
        curr,
        gen,
        adv,
        timestep_mask,
        sample_mask,
        current_mean=current_mean,
        reference_mean=reference_mean,
        std_dev=std_dev,
    )
    loss.backward()

    assert curr.grad is not None and torch.isfinite(curr.grad).all()
    assert current_mean.grad is not None and torch.isfinite(current_mean.grad).all()
