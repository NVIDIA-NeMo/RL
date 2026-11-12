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

"""TrainingEngine subclass whose loss method is NeMo RL's clipped policy-gradient loss."""

import torch
from megatron.bridge.engine import ModelOutput, PreparedMicrobatch, TrainingEngine

from nemo_rl.algorithms.loss import ClippedPGLossFn

# Metrics the loss returns that are not sums over tokens, so they cannot be added across
# microbatches and ranks: its own total, a per-rank sample count, a mean with a separate
# denominator, and extrema.
# TODO: reduce the ratio extrema with min and max instead of dropping them.
_NOT_ADDITIVE = frozenset(
    {
        "loss",
        "num_valid_samples",
        "positive_nll_loss",
        "probs_ratio_min",
        "probs_ratio_max",
        "probs_ratio_clamped_min",
        "probs_ratio_clamped_max",
    }
)


def _with_first_column(values: torch.Tensor) -> torch.Tensor:
    """Prepend a zero column: the loss drops column 0 of every per-token field."""
    return torch.cat([values.new_zeros(values.shape[0], 1), values], dim=1)


class NeMoRLTrainingEngine(TrainingEngine):
    """TrainingEngine with NeMo RL's loss as a loss method."""

    def clipped_pg_loss(
        self,
        batch: PreparedMicrobatch,
        output: ModelOutput,
        *,
        loss_fn: ClippedPGLossFn,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Token-level clipped policy-gradient loss, summed over this rank's active positions.

        The engine's fields are already target-aligned, while the loss expects batch fields
        whose column 0 is the first token, so a zero column is prepended to each. Global
        counts of 1 turn the loss's means into sums; the engine divides the accumulated
        gradient by the global token count once per update.
        """
        active = batch.active_mask
        data = {
            "token_mask": _with_first_column(active.to(torch.float32)),
            "sample_mask": active.new_ones(active.shape[0], dtype=torch.float32),
            "advantages": _with_first_column(batch.advantages),
            "prev_logprobs": _with_first_column(batch.old_logprobs),
            "generation_logprobs": _with_first_column(batch.rollout_logprobs),
            "reference_policy_logprobs": _with_first_column(batch.reference_logprobs),
        }
        one = torch.ones((), device=active.device)
        loss, metrics = loss_fn(
            next_token_logprobs=output.logprobs.to(torch.float32),
            data=data,
            global_valid_seqs=one,
            global_valid_toks=one,
        )
        sums = {
            name: torch.as_tensor(value, dtype=torch.float64, device=active.device)
            for name, value in metrics.items()
            if name not in _NOT_ADDITIVE
        }
        return loss, sums
