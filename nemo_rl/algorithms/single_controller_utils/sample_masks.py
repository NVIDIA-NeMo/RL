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
"""Which rows feed the group baseline when some rows are masked from the loss.

The advantage stage builds ``final_sample_mask`` = finalizer validity x not
env-flagged (``mask_sample``, including ``env.mask_sample_rules``) x not
truncated (when ``grpo.overlong_filtering``) x not seq-logprob-error masked.
That mask gates the gradient and, by default, also the per-prompt baseline
and std the estimator computes.

``grpo.masked_sample_rewards_in_baseline: true`` splits the two for rows
flagged *incomplete* (env flag or overlong filtering): they still get no
gradient, but their reward keeps counting in their siblings' baseline/std, so
a harness or verifier timeout still reads as a failure for the group instead
of silently raising the group's pass rate. Placeholder rows (no tokens,
``sample_mask`` 0) and seq-logprob-error rows stay out of both either way.
"""

from __future__ import annotations

import torch


def baseline_valid_mask(
    *,
    sample_mask: torch.Tensor,
    final_sample_mask: torch.Tensor,
    incomplete: torch.Tensor,
    keep_incomplete_rewards: bool,
) -> torch.Tensor:
    """Per-row weights for the estimator's baseline/std.

    Args:
        sample_mask: ``(B,)`` finalizer validity with loss weights (placeholder
            rows are 0).
        final_sample_mask: ``(B,)`` what trains after every mask was applied.
        incomplete: ``(B,)`` bool, rows masked because they were flagged
            incomplete (env ``mask_sample`` or overlong filtering).
        keep_incomplete_rewards: the ``grpo.masked_sample_rewards_in_baseline``
            switch.

    Returns:
        ``final_sample_mask`` unchanged when the switch is off. Otherwise the
        incomplete rows that the finalizer considered valid re-enter with their
        ``sample_mask`` weight; every other row keeps its ``final_sample_mask``.
    """
    if not keep_incomplete_rewards:
        return final_sample_mask
    if (
        incomplete.shape != sample_mask.shape
        or final_sample_mask.shape != sample_mask.shape
    ):
        raise ValueError(
            "baseline_valid_mask expects matching (B,) shapes, got "
            f"sample_mask={tuple(sample_mask.shape)} final={tuple(final_sample_mask.shape)} "
            f"incomplete={tuple(incomplete.shape)}"
        )
    reinstate = incomplete.bool() & (sample_mask > 0)
    return torch.where(reinstate, sample_mask, final_sample_mask)
