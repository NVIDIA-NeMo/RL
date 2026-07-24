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
from typing import Any

import torch


def compute_window_mask(
    num_steps: int,
    window_start: int,
    window_size: int | None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a `[num_steps]` mask that is 1 inside `[window_start, window_start + window_size)` and 0 elsewhere.

    Used by both the rollout adapter (to decide which steps emit a non-zero
    `generation_logprob`) and the loss (to mask out steps outside the active
    SDE window). When `window_size` is None, the mask is 1 for every step.
    """
    if window_size is None:
        return torch.ones(num_steps, device=device, dtype=dtype)
    if window_size < 0:
        raise ValueError(f"window_size must be non-negative, got {window_size}")
    if window_start < 0 or window_start > num_steps:
        raise ValueError(
            f"window_start={window_start} out of range for num_steps={num_steps}"
        )
    mask = torch.zeros(num_steps, device=device, dtype=dtype)
    end = min(window_start + window_size, num_steps)
    mask[window_start:end] = 1.0
    return mask


def _timestep_values(timestep: float | torch.Tensor) -> list[float | torch.Tensor]:
    if isinstance(timestep, torch.Tensor):
        if timestep.ndim == 0:
            return [timestep]
        return list(timestep)
    return [timestep]


def sde_step_with_logprob(
    scheduler: Any,
    model_output: torch.Tensor,
    timestep: float | torch.Tensor,
    sample: torch.Tensor,
    *,
    noise_level: float,
    prev_sample: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    sde_type: str = "sde",
    reduce_per_sample: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Step a flow-matching scheduler and return the transition log probability.

    When ``reduce_per_sample=True`` (default) the log-probability is mean-reduced
    over every non-batch dimension and returned with shape ``[B]``. When False
    it is returned at full per-element resolution matching ``sample.shape``,
    so downstream code can aggregate per-(latent-token) for cross-stack parity
    with verl-omni's `FlowMatchSDEDiscreteScheduler.sample_previous_step`.
    """
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    step_indices = [
        int(scheduler.index_for_timestep(t)) for t in _timestep_values(timestep)
    ]
    sigmas = torch.as_tensor(
        scheduler.sigmas, device=sample.device, dtype=torch.float32
    )
    sigma = sigmas[step_indices].view(-1, *([1] * (sample.ndim - 1)))
    sigma_prev = sigmas[[idx + 1 for idx in step_indices]].view(
        -1, *([1] * (sample.ndim - 1))
    )
    sigma_max = sigmas[1]
    dt = sigma_prev - sigma

    if sde_type == "sde":
        std_dev_t = (
            torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))
            * noise_level
        )
        prev_sample_mean = sample * (1 + std_dev_t.square() / (2 * sigma) * dt)
        prev_sample_mean = (
            prev_sample_mean
            + model_output * (1 + std_dev_t.square() * (1 - sigma) / (2 * sigma)) * dt
        )
        sqrt_negative_dt = torch.sqrt(-dt)
        if prev_sample is None:
            variance_noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = (
                prev_sample_mean + std_dev_t * sqrt_negative_dt * variance_noise
            )

        std = std_dev_t * sqrt_negative_dt
        log_prob = (
            -(prev_sample.detach() - prev_sample_mean).square() / (2 * std.square())
            - torch.log(std)
            - math.log(math.sqrt(2 * math.pi))
        )
    elif sde_type == "cps":
        std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)
        pred_original_sample = sample - sigma * model_output
        noise_estimate = sample + model_output * (1 - sigma)
        prev_sample_mean = pred_original_sample * (1 - sigma_prev)
        prev_sample_mean = prev_sample_mean + noise_estimate * torch.sqrt(
            sigma_prev.square() - std_dev_t.square()
        )
        if prev_sample is None:
            variance_noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        log_prob = -(prev_sample.detach() - prev_sample_mean).square()
    else:
        raise ValueError(f"Unsupported SDE type: {sde_type}")

    if reduce_per_sample:
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample, log_prob, prev_sample_mean, std_dev_t
