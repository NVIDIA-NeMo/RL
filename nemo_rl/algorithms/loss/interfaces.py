# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class LossType(enum.Enum):
    TOKEN_LEVEL = "token_level"
    SEQUENCE_LEVEL = "sequence_level"


class MetricNormalizer(enum.Enum):
    """Global denominator a loss-returned metric was normalized by.

    Losses reduce most metrics with ``masked_mean(...,
    global_normalization_factor=...)`` where the factor is the global valid
    *token* count, the global valid *sequence* count, or absent entirely (raw
    counts, per-microbatch means, extrema). Split-API trainers run each
    microbatch with placeholder ``global_valid_*=1`` (collecting raw sums) and
    rescale once per optimizer step — to do that they must know, per metric,
    which denominator applies.

    Losses advertise the mapping via a ``metric_normalizations:
    dict[str, MetricNormalizer]`` instance attribute, built in ``__init__``
    from the same flags that pick the denominators, so it lives next to the
    metric definitions instead of in a consumer-side table. Metrics absent
    from the mapping fall back to the gradient normalization (the
    objective denominator) in split-API trainers. Survivor rescaling only
    changes advertised denominators and leaves unadvertised metrics unchanged.
    """

    TOKENS = "tokens"  # divided by global_valid_toks
    SEQUENCES = "sequences"  # divided by global_valid_seqs
    NONE = "none"  # not normalized: raw counts, local means, min/max


@dataclass(frozen=True)
class ObjectiveReduction:
    """Own the objective formula and its independently accumulated denominator."""

    loss_type: ClassVar[LossType]
    metric_normalizer: ClassVar[MetricNormalizer]
    requires_explicit_denominator: ClassVar[bool] = False
    token_normalized: ClassVar[bool] = False

    def per_response_weights(
        self, data: Mapping[str, torch.Tensor]
    ) -> torch.Tensor | None:
        """Return one weight per unpacked response ([B]), or None."""
        return None

    def local_counts(
        self, data: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Count sequence, token and objective mass before packing/CP splitting."""
        sample_mask = data["sample_mask"].float()
        if "token_mask" in data:
            valid_tokens_per_response = (
                data["token_mask"][:, 1:].sum(-1, dtype=torch.float32) * sample_mask
            )
        else:
            # Preserve legacy losses that count every input position without a mask.
            valid_tokens_per_response = sample_mask * data["input_ids"].shape[1]
        sequence_count = sample_mask.sum()
        token_count = valid_tokens_per_response.sum()
        contribution = self.denominator_contribution(
            data,
            valid_tokens_per_response=valid_tokens_per_response,
            sequence_count=sequence_count,
            token_count=token_count,
        )
        return sequence_count, token_count, contribution

    def denominator_contribution(
        self,
        data: Mapping[str, torch.Tensor],
        *,
        valid_tokens_per_response: torch.Tensor,
        sequence_count: torch.Tensor,
        token_count: torch.Tensor,
    ) -> torch.Tensor:
        """Return a scalar contribution to sum across chunks and DP ranks."""
        raise NotImplementedError

    def loss_divisor(
        self,
        *,
        global_valid_toks: torch.Tensor,
        global_valid_seqs: torch.Tensor,
        global_objective_denominator: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return this call's scalar divisor; streaming supplies one."""
        raise NotImplementedError

    def reduce(
        self,
        values: torch.Tensor,
        *,
        token_mask: torch.Tensor,
        sample_mask: torch.Tensor,
        mask: torch.Tensor,
        denominator: torch.Tensor,
        per_response_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        raise NotImplementedError


@dataclass(frozen=True)
class GlobalTokenMeanReduction(ObjectiveReduction):
    """Sum valid token losses and divide by total valid token mass."""

    loss_type: ClassVar[LossType] = LossType.TOKEN_LEVEL
    metric_normalizer: ClassVar[MetricNormalizer] = MetricNormalizer.TOKENS
    token_normalized: ClassVar[bool] = True

    def denominator_contribution(
        self,
        data: Mapping[str, torch.Tensor],
        *,
        valid_tokens_per_response: torch.Tensor,
        sequence_count: torch.Tensor,
        token_count: torch.Tensor,
    ) -> torch.Tensor:
        """Return a scalar contribution to sum across chunks and DP ranks."""
        return token_count

    def loss_divisor(
        self,
        *,
        global_valid_toks: torch.Tensor,
        global_valid_seqs: torch.Tensor,
        global_objective_denominator: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return this call's scalar divisor; streaming supplies one."""
        return global_valid_toks

    def reduce(
        self,
        values: torch.Tensor,
        *,
        token_mask: torch.Tensor,
        sample_mask: torch.Tensor,
        mask: torch.Tensor,
        denominator: torch.Tensor,
        per_response_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        # Keep the interface importable without the optional tokenizer dependencies.
        from nemo_rl.algorithms.utils import masked_mean

        return masked_mean(values, mask, global_normalization_factor=denominator)


@dataclass(frozen=True)
class SequenceMeanReduction(ObjectiveReduction):
    """Average tokens within each response, then average weighted responses."""

    loss_type: ClassVar[LossType] = LossType.SEQUENCE_LEVEL
    metric_normalizer: ClassVar[MetricNormalizer] = MetricNormalizer.SEQUENCES

    def denominator_contribution(
        self,
        data: Mapping[str, torch.Tensor],
        *,
        valid_tokens_per_response: torch.Tensor,
        sequence_count: torch.Tensor,
        token_count: torch.Tensor,
    ) -> torch.Tensor:
        """Return a scalar contribution to sum across chunks and DP ranks."""
        return sequence_count

    def loss_divisor(
        self,
        *,
        global_valid_toks: torch.Tensor,
        global_valid_seqs: torch.Tensor,
        global_objective_denominator: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return this call's scalar divisor; streaming supplies one."""
        return global_valid_seqs

    def reduce(
        self,
        values: torch.Tensor,
        *,
        token_mask: torch.Tensor,
        sample_mask: torch.Tensor,
        mask: torch.Tensor,
        denominator: torch.Tensor,
        per_response_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        # Keep the interface importable without the optional tokenizer dependencies.
        from nemo_rl.algorithms.utils import masked_mean

        # Keep sample weights outside the inner mean so fractional weights survive.
        return masked_mean(
            masked_mean(values, token_mask, dim=-1),
            sample_mask,
            global_normalization_factor=denominator,
        )


def make_objective_reduction(normalization: str) -> ObjectiveReduction:
    """Resolve the configured formula once, before training."""
    if normalization == "global_token_mean":
        return GlobalTokenMeanReduction()
    if normalization == "sequence_mean":
        return SequenceMeanReduction()
    raise ValueError(f"Unsupported loss normalization: {normalization}")


def get_objective_reduction(loss_fn: "LossFunction") -> ObjectiveReduction:
    """Use the explicit reducer, falling back to the legacy LossType contract."""
    reduction = getattr(loss_fn, "objective_reduction", None)
    if isinstance(reduction, ObjectiveReduction):
        return reduction

    legacy_loss_type = getattr(loss_fn, "loss_type", LossType.TOKEN_LEVEL)
    if legacy_loss_type == LossType.SEQUENCE_LEVEL:
        return SequenceMeanReduction()
    return GlobalTokenMeanReduction()


class LossInputType(enum.Enum):
    LOGIT = "logit"
    LOGPROB = "logprob"
    OPD_FULL = "opd_full"
    DISTILLATION = "distillation"
    DISTILLATION_CROSS_TOKENIZER = "distillation_cross_tokenizer"
    DRAFT = "draft"


class LossFunction(Protocol):
    """Signature for loss functions used in reinforcement learning algorithms.

    Loss functions compute a scalar loss value and associated metrics from
    model logprobs and other data contained in a BatchedDataDict.

    Losses may expose an ``objective_reduction: ObjectiveReduction`` describing
    streaming denominator counts. Losses without it retain their ``loss_type``
    token/sequence normalization.

    Losses may additionally expose a ``metric_normalizations:
    dict[str, MetricNormalizer]`` attribute advertising the global denominator
    each returned metric was normalized by (see ``MetricNormalizer``). It is
    optional: split-API trainers fall back to the objective denominator for
    metrics (or losses) that do not advertise. Survivor rescaling instead
    leaves metrics absent from the mapping unchanged.
    """

    loss_type: LossType
    input_type: LossInputType

    def __call__(
        self,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute loss and metrics from logprobs and other data.

        Args:
            data: Dictionary containing all relevant data for loss computation
                  such as rewards, values, actions, advantages, masks, and other
                  algorithm-specific information needed for the particular loss calculation.
            global_valid_seqs: torch.Tensor
                This tensor should contain the number of valid sequences in the microbatch.
                It's used for global normalization for losses/metrics that are computed at the sequence level
                and needs to be aggregated across all microbatches.
            global_valid_toks: torch.Tensor
                This tensor should contain the number of valid tokens in the microbatch.
                It's used for global normalization for losses/metrics that are computed at the token level
                and needs to be aggregated across all microbatches.
            **kwargs: Loss function input, which varies by input_type:
                - For LossInputType.LOGPROB: next_token_logprobs (torch.Tensor)
                - For LossInputType.LOGIT: logits (torch.Tensor)
                - For LossInputType.OPD_FULL: opd_full_divergence, and optionally
                  opd_full_entropy / opd_full_cross_entropy (torch.Tensor)
                - For LossInputType.DISTILLATION: student_topk_logprobs, teacher_topk_logprobs, H_all (torch.Tensor)
                - For LossInputType.DISTILLATION_CROSS_TOKENIZER: logits (torch.Tensor), teacher_full_logits_by_idx (dict[int, torch.Tensor])
                - For LossInputType.DRAFT: teacher_logits, student_logits, mask (torch.Tensor)

        Returns:
            tuple: (loss, metrics)
                - loss: A scalar tensor representing the loss value to be minimized during training
                - metrics: A dictionary of metrics related to the loss computation, which may include
                  component losses, statistics about gradients/rewards, and other diagnostic information
        """
        ...
