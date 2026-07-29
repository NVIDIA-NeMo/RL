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
"""Diffusion-GRPO loss.

This module intentionally does NOT implement
:class:`nemo_rl.algorithms.loss.interfaces.LossFunction`. The Protocol's
``__call__(data, global_valid_seqs, global_valid_toks, **kwargs)`` and the
``LossType.{TOKEN,SEQUENCE}_LEVEL`` taxonomy are coupled to next-token /
sequence semantics; mapping diffusion timestep semantics onto them would either
distort the abstractions or require renaming the args at every call site. The
diffusion training loop calls this loss directly from
:class:`nemo_rl.models.diffusion.workers.diffusion_worker.DiffusionPolicyWorker`,
so Protocol conformance is not needed.

The formula mirrors verl-omni
``verl_omni/trainer/diffusion/diffusion_algos.py::FlowGRPOLoss.compute_loss``
which itself follows
https://github.com/yifan123/flow_grpo/blob/main/scripts/train_sd3_fast.py#L885.
"""

from typing import Any

import torch

from nemo_rl.algorithms.utils import masked_mean
from nemo_rl.models.diffusion.interfaces import DiffusionLossConfig


class DiffusionGRPOLossFn:
    """Clipped policy-gradient loss for diffusion GRPO with optional Gaussian KL.

    Inputs are all shaped ``[B, T]`` (batch times generations-per-prompt times
    inference steps). The mean tensors used by the KL term are shaped
    ``[B, T, C, H, W]``.

    The advantage tensor is clamped to ``adv_clip_max`` before the ratio
    computation, to match verl-omni's FlowGRPOLoss behaviour.
    """

    def __init__(self, cfg: DiffusionLossConfig | dict[str, Any]):
        # Normalize through the schema so both validated dicts (from the
        # entrypoint's model_dump) and partial dicts get schema defaults.
        self.cfg = DiffusionLossConfig.model_validate(cfg).model_dump()

    def __call__(
        self,
        curr_logprob: torch.Tensor,
        generation_logprob: torch.Tensor,
        advantages: torch.Tensor,
        timestep_mask: torch.Tensor,
        sample_mask: torch.Tensor,
        *,
        current_mean: torch.Tensor | None = None,
        reference_mean: torch.Tensor | None = None,
        std_dev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        ratio_clip_min: float = self.cfg["ratio_clip_min"]
        ratio_clip_max: float = self.cfg["ratio_clip_max"]
        adv_clip_max: float | None = self.cfg["adv_clip_max"]
        beta: float = self.cfg["beta"]

        device = curr_logprob.device
        aggregate_per_sample: bool = bool(self.cfg["aggregate_logprobs_per_sample"])
        advantages = advantages.to(device=device, dtype=curr_logprob.dtype)
        if adv_clip_max is not None:
            advantages = advantages.clamp(-adv_clip_max, adv_clip_max)
        generation_logprob = generation_logprob.to(
            device=device, dtype=curr_logprob.dtype
        )
        timestep_mask = timestep_mask.to(device=device)
        sample_mask = sample_mask.to(device=device)
        if aggregate_per_sample:
            # Experimental sum-aggregation, NOT verl-omni semantics: verl-omni
            # keeps log_prob at [B] by averaging over all non-batch dims and
            # never sums along T, so its per-(sample, step) ratio is what the
            # default False path below reproduces. Summing here inflates the
            # log-ratio scale by ~window_size, which is incompatible with the
            # 1e-4-scale ratio clip used for verl-omni parity.
            #   [B, T] input (per-step-aggregated logprob) collapses to [B].
            tm = timestep_mask.float()
            while tm.ndim < curr_logprob.ndim:
                tm = tm.unsqueeze(-1)
            if curr_logprob.ndim <= 2:
                # [B, T] → [B]. Keep all elements masked-summed.
                curr_logprob = (curr_logprob * tm).sum(
                    dim=tuple(range(1, curr_logprob.ndim))
                )
                generation_logprob = (generation_logprob * tm).sum(
                    dim=tuple(range(1, generation_logprob.ndim))
                )
                advantages = advantages[:, 0] if advantages.ndim > 1 else advantages
                timestep_mask = torch.ones_like(curr_logprob)
            else:
                # Per-element mode: [B, T, N, packed_C, ...]. Sum T and any
                # trailing channel dims, keep the FIRST non-batch spatial
                # axis (treated as the latent-token axis). The resulting
                # ratio has shape [B, N_token]. Experimental; verl-omni never
                # produces a per-token ratio (its log_prob is [B]).
                # dims to sum: T (dim 1) and dims 3..ndim-1 (everything past N)
                sum_dims = (1,) + tuple(range(3, curr_logprob.ndim))
                curr_logprob = (curr_logprob * tm).sum(dim=sum_dims)
                generation_logprob = (generation_logprob * tm).sum(dim=sum_dims)
                # advantages broadcast from [B] to [B, N_token].
                if advantages.ndim > 1:
                    advantages = advantages[:, 0]
                advantages = advantages.unsqueeze(-1).expand_as(curr_logprob)
                timestep_mask = torch.ones_like(curr_logprob)
                sample_mask = sample_mask.unsqueeze(-1).expand_as(curr_logprob)
        log_ratio = curr_logprob - generation_logprob
        ratio = log_ratio.exp()
        unclipped = -advantages * ratio
        clipped = -advantages * ratio.clamp(1.0 - ratio_clip_min, 1.0 + ratio_clip_max)
        pg = torch.maximum(unclipped, clipped)

        if aggregate_per_sample:
            mask = timestep_mask.float() * sample_mask.float()
        else:
            mask = timestep_mask.float() * sample_mask.float().unsqueeze(-1)
        policy_loss = masked_mean(pg, mask)

        with torch.no_grad():
            clipfrac_higher = masked_mean(
                ((ratio - 1.0) > ratio_clip_max).float(), mask
            )
            clipfrac_lower = masked_mean(((1.0 - ratio) > ratio_clip_min).float(), mask)
            clipfrac = clipfrac_higher + clipfrac_lower
            approx_kl = masked_mean(-log_ratio, mask)
            mean_ratio = masked_mean(ratio, mask)
            # ratio_min / ratio_max only reflect entries inside the mask;
            # use a large/small sentinel for masked-out positions.
            masked_ratio_for_min = torch.where(
                mask > 0, ratio, torch.full_like(ratio, float("inf"))
            )
            masked_ratio_for_max = torch.where(
                mask > 0, ratio, torch.full_like(ratio, float("-inf"))
            )
            ratio_min = masked_ratio_for_min.min()
            ratio_max = masked_ratio_for_max.max()

        metrics: dict[str, Any] = {
            "policy_loss": policy_loss.detach(),
            "approx_kl": approx_kl.detach(),
            "clipfrac": clipfrac.detach(),
            "clipfrac_higher": clipfrac_higher.detach(),
            "clipfrac_lower": clipfrac_lower.detach(),
            "mean_ratio": mean_ratio.detach(),
            "ratio_min": ratio_min.detach(),
            "ratio_max": ratio_max.detach(),
        }

        loss = policy_loss
        if (
            beta > 0
            and current_mean is not None
            and reference_mean is not None
            and std_dev is not None
        ):
            spatial_dims = tuple(range(2, current_mean.ndim))
            kl_per_step = (
                (current_mean - reference_mean).pow(2) / (2.0 * std_dev.pow(2))
            ).mean(dim=spatial_dims)
            kl_loss = masked_mean(kl_per_step, mask)
            loss = policy_loss + beta * kl_loss
            metrics["kl_loss"] = kl_loss.detach()

        return loss, metrics
