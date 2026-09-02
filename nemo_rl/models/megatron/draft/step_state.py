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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from nemo_rl.algorithms.loss.draft import DraftLossStats

DRAFT_STEP_PAYLOAD_KEY = "_draft_step_payload"
DRAFT_LOSS_METRIC_KEY = "draft_loss"


@dataclass(frozen=True, slots=True)
class DraftStepPayload:
    """Detached raw EAGLE statistics carried through loss metrics."""

    stats: DraftLossStats


@dataclass(slots=True)
class DraftStepState:
    """Accumulate one-bin EAGLE loss state across split microbatches."""

    _local_numerators: torch.Tensor | None = None
    _local_counts: torch.Tensor | None = None
    _weights: torch.Tensor | None = None
    _global_counts: torch.Tensor | None = None

    @staticmethod
    def metric_payload(stats: DraftLossStats) -> DraftStepPayload:
        """Build a non-differentiable payload safe for metric transport."""
        detached_stats = DraftLossStats(
            numerators=stats.numerators.detach().clone(),
            counts=stats.counts.detach().clone(),
            weights=stats.weights.detach().clone(),
        )
        return DraftStepPayload(stats=detached_stats)

    @property
    def active(self) -> bool:
        return self._local_counts is not None

    @property
    def local_numerators(self) -> torch.Tensor:
        if self._local_numerators is None:
            raise RuntimeError("draft step has no accumulated statistics")
        return self._local_numerators

    @property
    def local_counts(self) -> torch.Tensor:
        if self._local_counts is None:
            raise RuntimeError("draft step has no accumulated statistics")
        return self._local_counts

    def accumulate(self, payload: DraftStepPayload) -> None:
        """Add one detached microbatch payload to this optimizer step."""
        stats = payload.stats
        if stats.numerators.shape != (1,):
            raise ValueError(
                "current EAGLE draft step state requires exactly one bin, "
                f"got {stats.numerators.shape}."
            )
        if not torch.equal(stats.weights, torch.ones_like(stats.weights)):
            raise ValueError("current one-bin EAGLE draft loss requires unit weight")
        if self._local_numerators is None:
            self._local_numerators = stats.numerators.detach().clone()
            self._local_counts = stats.counts.detach().clone()
            self._weights = stats.weights.detach().clone()
            return
        if stats.numerators.shape != self._local_numerators.shape:
            raise ValueError(
                "draft statistics shape changed within a step, "
                f"from {self._local_numerators.shape} to {stats.numerators.shape}."
            )
        assert self._local_counts is not None
        assert self._weights is not None
        if not torch.equal(stats.weights, self._weights):
            raise ValueError("draft loss bin weights changed within a step")
        self._local_numerators.add_(stats.numerators.detach())
        self._local_counts.add_(stats.counts.detach())

    def counts_for_reduction(self, reference: torch.Tensor) -> torch.Tensor:
        """Return local counts on the existing policy-count reduction tensor."""
        if self._local_counts is None:
            return reference.new_zeros(0)
        return self._local_counts.to(device=reference.device, dtype=reference.dtype)

    def set_global_counts(self, counts: torch.Tensor) -> None:
        """Record the DP-reduced one-bin denominator."""
        if not self.active:
            if counts.numel() != 0:
                raise ValueError("inactive draft step cannot accept global counts")
            return
        if counts.shape != (1,):
            raise ValueError(
                f"global draft counts must contain one bin, got {counts.shape}"
            )
        self._global_counts = counts.detach().clone()

    def _normalization_scale(self) -> float:
        if self._global_counts is None or self._weights is None:
            raise RuntimeError("global draft counts have not been finalized")
        denominator = (
            self._global_counts.to(dtype=torch.float32)
            * self._weights.to(device=self._global_counts.device, dtype=torch.float32)
        ).sum()
        if denominator.item() <= 0:
            return 0.0
        return float((1.0 / denominator).item())

    def normalize_metric(self, value: Any) -> Any:
        """Normalize a raw local draft numerator by the global draft count."""
        scale = self._normalization_scale()
        if isinstance(value, torch.Tensor):
            return value.detach() * scale
        return value * scale

    def correct_main_grads(
        self,
        parameters: Iterable[Any],
        *,
        policy_normalization_count: torch.Tensor,
    ) -> None:
        """Correct draft-tagged main grads after policy normalization."""
        policy_count = float(policy_normalization_count.detach().item())
        draft_scale = self._normalization_scale()
        correction = policy_count * draft_scale if policy_count > 0 else 0.0
        matched = 0
        for param in parameters:
            if getattr(param, "grad_norm_group", None) != "draft":
                continue
            matched += 1
            main_grad = getattr(param, "main_grad", None)
            if main_grad is not None:
                main_grad.mul_(correction)
        if self.active and matched == 0:
            # An accumulated payload means draft params took part in the step,
            # so tagging must have run. A zero match silently leaves the draft
            # grads on the policy denominator.
            raise RuntimeError(
                "draft step state is active but no parameter carries "
                "grad_norm_group='draft'; draft parameter tagging did not run."
            )
