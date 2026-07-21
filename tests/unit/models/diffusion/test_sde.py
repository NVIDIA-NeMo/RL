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

from nemo_rl.models.diffusion.sde import compute_window_mask, sde_step_with_logprob


class FakeFlowMatchScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([1000.0, 750.0, 500.0])
        self.sigmas = torch.tensor([1.0, 0.6, 0.2, 0.0])

    def index_for_timestep(self, timestep: float | torch.Tensor) -> int:
        timestep_tensor = torch.as_tensor(timestep)
        matches = torch.nonzero(self.timesteps == timestep_tensor.cpu(), as_tuple=False)
        return int(matches[0].item())


def test_sde_step_recomputes_logprob_from_prev_sample():
    scheduler = FakeFlowMatchScheduler()
    model_output = torch.tensor(
        [
            [[0.1, -0.2], [0.3, -0.4]],
            [[-0.5, 0.6], [-0.7, 0.8]],
        ]
    )
    sample = torch.tensor(
        [
            [[1.0, 1.2], [1.4, 1.6]],
            [[-1.0, -1.2], [-1.4, -1.6]],
        ]
    )
    timestep = torch.tensor([750.0, 750.0])
    noise_level = 0.7

    sigma = torch.tensor(0.6).view(1, 1, 1)
    sigma_prev = torch.tensor(0.2).view(1, 1, 1)
    dt = sigma_prev - sigma
    std_dev_t = torch.sqrt(sigma / (1 - sigma)) * noise_level
    expected_mean = sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
    expected_mean = (
        expected_mean
        + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )
    prev_sample = expected_mean + 0.25 * std_dev_t * torch.sqrt(-dt)
    std = std_dev_t * torch.sqrt(-dt)
    expected_logprob = (
        -((prev_sample - expected_mean) ** 2) / (2 * std**2)
        - torch.log(std)
        - math.log(math.sqrt(2 * math.pi))
    ).mean(dim=(1, 2))

    actual_prev_sample, actual_logprob, actual_mean, actual_std_dev_t = (
        sde_step_with_logprob(
            scheduler,
            model_output,
            timestep,
            sample,
            noise_level=noise_level,
            prev_sample=prev_sample,
        )
    )

    torch.testing.assert_close(actual_prev_sample, prev_sample)
    torch.testing.assert_close(actual_logprob, expected_logprob)
    torch.testing.assert_close(actual_mean, expected_mean)
    torch.testing.assert_close(actual_std_dev_t, std_dev_t.expand_as(actual_std_dev_t))


def test_sde_step_sampling_is_seeded():
    scheduler = FakeFlowMatchScheduler()
    model_output = torch.zeros(2, 2, 2)
    sample = torch.ones(2, 2, 2)
    timestep = torch.tensor([750.0, 750.0])

    first = sde_step_with_logprob(
        scheduler,
        model_output,
        timestep,
        sample,
        noise_level=0.7,
        generator=torch.Generator().manual_seed(123),
    )
    second = sde_step_with_logprob(
        scheduler,
        model_output,
        timestep,
        sample,
        noise_level=0.7,
        generator=torch.Generator().manual_seed(123),
    )

    for first_tensor, second_tensor in zip(first, second):
        torch.testing.assert_close(first_tensor, second_tensor)


def test_sde_step_uses_float32_for_bfloat16_inputs():
    scheduler = FakeFlowMatchScheduler()
    model_output = torch.zeros(1, 2, 2, dtype=torch.bfloat16)
    sample = torch.ones(1, 2, 2, dtype=torch.bfloat16)
    prev_sample = torch.ones(1, 2, 2, dtype=torch.bfloat16)

    outputs = sde_step_with_logprob(
        scheduler,
        model_output,
        torch.tensor([750.0]),
        sample,
        noise_level=0.7,
        prev_sample=prev_sample,
    )

    for output in outputs:
        assert output.dtype == torch.float32


def test_cps_step_recomputes_logprob_from_prev_sample():
    scheduler = FakeFlowMatchScheduler()
    model_output = torch.tensor([[[0.2, -0.1], [0.4, -0.3]]])
    sample = torch.tensor([[[1.0, -1.0], [0.5, -0.5]]])
    timestep = torch.tensor([750.0])
    noise_level = 0.4

    sigma = torch.tensor(0.6).view(1, 1, 1)
    sigma_prev = torch.tensor(0.2).view(1, 1, 1)
    expected_std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)
    pred_original_sample = sample - sigma * model_output
    noise_estimate = sample + model_output * (1 - sigma)
    expected_mean = pred_original_sample * (1 - sigma_prev)
    expected_mean = expected_mean + noise_estimate * torch.sqrt(
        sigma_prev**2 - expected_std_dev_t**2
    )
    prev_sample = expected_mean + 0.5
    expected_logprob = -((prev_sample - expected_mean) ** 2).mean(dim=(1, 2))

    actual_prev_sample, actual_logprob, actual_mean, actual_std_dev_t = (
        sde_step_with_logprob(
            scheduler,
            model_output,
            timestep,
            sample,
            noise_level=noise_level,
            prev_sample=prev_sample,
            sde_type="cps",
        )
    )

    torch.testing.assert_close(actual_prev_sample, prev_sample)
    torch.testing.assert_close(actual_logprob, expected_logprob)
    torch.testing.assert_close(actual_mean, expected_mean)
    torch.testing.assert_close(actual_std_dev_t, expected_std_dev_t)


def test_window_mask_full_when_size_is_none():
    mask = compute_window_mask(8, 0, None)
    assert torch.equal(mask, torch.ones(8))


def test_window_mask_partial_window():
    mask = compute_window_mask(8, 2, 3)
    assert torch.equal(mask, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))


def test_window_mask_clamped_at_end():
    mask = compute_window_mask(8, 6, 5)
    assert torch.equal(mask, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]))


def test_window_mask_rejects_negative_size():
    with pytest.raises(ValueError, match="non-negative"):
        compute_window_mask(8, 0, -1)


def test_window_mask_rejects_out_of_range_start():
    with pytest.raises(ValueError, match="out of range"):
        compute_window_mask(8, 9, 1)


def test_sde_step_rejects_unknown_sde_type():
    scheduler = FakeFlowMatchScheduler()

    with pytest.raises(ValueError, match="Unsupported SDE type"):
        sde_step_with_logprob(
            scheduler,
            torch.zeros(1, 2, 2),
            torch.tensor([750.0]),
            torch.ones(1, 2, 2),
            noise_level=0.7,
            sde_type="unknown",
        )


def test_flow_match_scheduler_contract():
    """Guard the informal scheduler contract against diffusers API drift.

    `sde_step_with_logprob` and the worker's `_load_pipeline` assume the real
    `FlowMatchEulerDiscreteScheduler` exposes `timesteps` (1-D), `sigmas`
    (1-D with a terminal sentinel entry) and `index_for_timestep(t) -> int`.
    """
    pytest.importorskip("diffusers")

    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler

    T = 8
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    sigmas = np.linspace(1.0, 1 / T, T)
    # Same call shape as DiffusionPolicyWorker._load_pipeline.
    scheduler.set_timesteps(T, device="cpu", sigmas=sigmas.tolist(), mu=0.8)

    assert scheduler.timesteps.ndim == 1
    assert len(scheduler.timesteps) == T
    # One trailing sentinel sigma so sigmas[idx + 1] is valid for every step.
    assert scheduler.sigmas.ndim == 1
    assert len(scheduler.sigmas) == T + 1
    assert float(scheduler.sigmas[-1]) == 0.0

    for i, t in enumerate(scheduler.timesteps):
        assert scheduler.index_for_timestep(t) == i
