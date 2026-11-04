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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol

import torch

from nemo_rl.data_plane.schema import GROUP_TOKEN_MEAN_WEIGHT
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
    metric definitions instead of in a consumer-side table. Objective terms
    (``loss``, KL) advertise ``LOSS_DIVISOR`` so they are logged with the
    gradient's own denominator, whatever the reduction. Metrics absent
    from the mapping fall back to the gradient normalization (the
    objective denominator) in split-API trainers. Survivor rescaling
    requires every returned metric to be listed.
    """

    TOKENS = "tokens"  # divided by global_valid_toks
    SEQUENCES = "sequences"  # divided by global_valid_seqs
    NONE = "none"  # not normalized: raw counts, local means, min/max
    # Divide by the objective reduction's loss_divisor(), i.e. exactly the
    # divisor the gradient used.
    LOSS_DIVISOR = "loss_divisor"


@dataclass(frozen=True)
class ObjectiveReduction:
    """Own the objective formula and its independently accumulated denominator.

    Attributes:
        loss_type: Legacy export for ``LossFunction.loss_type`` readers
            (``process_global_batch``, the GSPO/CISPO asserts). New code
            branches on ``token_normalized`` or calls ``reduce``; a reduction
            can be ``TOKEN_LEVEL`` here and still not be token normalized.
        requires_explicit_denominator: True when the split worker must pass its
            finish-time count explicitly (``global_valid_*`` are not it).
        token_normalized: True when token-summed auxiliary losses (MTP, router
            aux losses) can share this reduction's denominator.
    """

    loss_type: ClassVar[LossType]
    requires_explicit_denominator: ClassVar[bool] = False
    token_normalized: ClassVar[bool] = False
    extra_fields: ClassVar[tuple[str, ...]] = ()

    def prepare_batch(
        self,
        *,
        sample_ids: Sequence[str],
        token_mask: torch.Tensor,
        sample_mask: torch.Tensor,
        group_indices: torch.Tensor,
        expected_group_size: int | None,
    ) -> dict[str, torch.Tensor]:
        """Prepare declared fields while final masks and complete groups coexist."""
        return {}

    def validate_setup(
        self,
        *,
        trainer: Literal["single_controller", "grpo", "ppo"],
        algorithm: Literal["grpo", "ppo"],
        megatron_enabled: bool,
        on_policy_distillation: bool,
    ) -> None:
        """Validate the execution path before model construction."""

    def validate_loss_options(
        self,
        *,
        sequence_level_importance_ratios: bool,
        opd_full: bool,
        positive_example_nll_weight: float,
        seq_logprob_error_in_loss: bool,
    ) -> None:
        """Validate loss options that could change the reduction's meaning."""

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
        """Reduce per-token values to this call's scalar share of the objective.

        Every loss term that is normalized like the gradient (actor, KL,
        score-centering correction, OPD-full divergence) goes through this
        method, so all of them share one formula and one denominator.

        Args:
            values: Per-token values ``[B, S - 1]``.
            token_mask: Response-token mask ``[B, S - 1]``, before sample masking.
            sample_mask: Per-response loss weights ``[B]``.
            mask: ``token_mask * sample_mask[:, None]``.
            denominator: Scalar from ``loss_divisor``: the global count on
                synchronous paths, or the split worker's placeholder when the
                finish-time count is applied later.
            per_response_weights: ``per_response_weights(data)`` output, or None.

        Returns:
            A scalar tensor; summing it over microbatches and data-parallel
            ranks yields the global objective.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class GlobalTokenMeanReduction(ObjectiveReduction):
    """Sum valid token losses and divide by total valid token mass."""

    loss_type: ClassVar[LossType] = LossType.TOKEN_LEVEL
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


def group_token_mean_weights(
    sample_ids: Sequence[str],
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    *,
    group_indices: torch.Tensor,
    expected_group_size: int | None = None,
) -> torch.Tensor:
    """Build float32 per-response weights ``1 / T_g`` for equal prompt-group token means.

    ``T_g`` is the valid generated-token mass of response ``i``'s prompt group:
    ``sum_j token_mask[j, 1:].sum() * sample_mask[j]`` over the group's rows.
    Nonnegative fractional masks contribute their full mass. A group with
    ``T_g == 0`` receives weight 0, so it adds no denominator mass.

    Args:
        sample_ids: One unique identifier per response row; used only to
            reject duplicated rows.
        token_mask: ``[B, S]`` loss mask aligned with ``input_ids``; position
            0 is never a predicted token and is excluded.
        sample_mask: ``[B]`` nonnegative per-response loss weights.
        group_indices: ``[B]`` ``torch.long`` prompt-group index per row,
            local to this batch (see ``group_index_column``).
        expected_group_size: When set, every nonempty group must have exactly
            this many rows; the advantage stage passes
            ``num_generations_per_prompt`` so a partial group fails loudly.
            ``None`` disables that check.

    Returns:
        ``[B]`` float32 detached weights; ``sum_i valid_toks_i * w_i`` equals
        the number of groups with positive token mass.

    Raises:
        ValueError: On shape/dtype mismatch, non-finite or negative masks,
            duplicate sample ids, or an incomplete group.
    """
    if token_mask.ndim != 2 or sample_mask.shape != (token_mask.shape[0],):
        raise ValueError("group token mean expects token_mask[B,S], sample_mask[B]")
    if len(sample_ids) != token_mask.shape[0] or len(set(sample_ids)) != len(
        sample_ids
    ):
        raise ValueError("group token mean requires one unique sample ID per response")
    if not torch.isfinite(token_mask).all() or not torch.isfinite(sample_mask).all():
        raise ValueError("group token mean masks must be finite")
    if (token_mask < 0).any() or (sample_mask < 0).any():
        raise ValueError("group token mean masks must be nonnegative")
    if group_indices.shape != sample_mask.shape or group_indices.dtype != torch.long:
        raise ValueError(
            "group token mean requires one integer group index per response"
        )
    group_sizes = torch.bincount(group_indices)
    if (
        expected_group_size is not None
        and (group_sizes[group_sizes > 0] != expected_group_size).any()
    ):
        raise ValueError(
            f"incomplete group: expected {expected_group_size} responses per group"
        )
    valid_lengths = token_mask[:, 1:].sum(-1, dtype=torch.float32)
    valid_lengths = valid_lengths * sample_mask.float()

    # Sum lengths by group, then look up the group total for each response.
    group_totals = valid_lengths.new_zeros(group_sizes.shape)
    group_totals.scatter_add_(0, group_indices, valid_lengths)
    row_totals = group_totals[group_indices]

    # Entirely masked groups keep zero weight.
    weights = torch.zeros_like(valid_lengths)
    nonempty = row_totals > 0
    weights[nonempty] = row_totals[nonempty].reciprocal()
    return weights.detach()


@dataclass(frozen=True)
class GroupTokenMeanReduction(GlobalTokenMeanReduction):
    """Average valid tokens within each prompt group, then average groups equally."""

    # metric_normalizer=TOKENS is inherited for the token diagnostics only;
    # ClippedPGLossFn drops the loss/kl entries because the group denominator applies.
    requires_explicit_denominator: ClassVar[bool] = True
    token_normalized: ClassVar[bool] = False
    extra_fields: ClassVar[tuple[str, ...]] = (GROUP_TOKEN_MEAN_WEIGHT,)

    def validate_setup(
        self,
        *,
        trainer: Literal["single_controller", "grpo", "ppo"],
        algorithm: Literal["grpo", "ppo"],
        megatron_enabled: bool,
        on_policy_distillation: bool,
    ) -> None:
        if (
            trainer != "single_controller"
            or algorithm != "grpo"
            or not megatron_enabled
            or on_policy_distillation
        ):
            raise ValueError(
                "group_token_mean requires SingleController GRPO with "
                "on_policy_distillation.enabled=false and policy.megatron_cfg.enabled=true."
            )

    def validate_loss_options(
        self,
        *,
        sequence_level_importance_ratios: bool,
        opd_full: bool,
        positive_example_nll_weight: float,
        seq_logprob_error_in_loss: bool,
    ) -> None:
        if sequence_level_importance_ratios:
            raise ValueError(
                "group_token_mean requires sequence_level_importance_ratios=False"
            )
        if opd_full or positive_example_nll_weight != 0:
            raise ValueError("group_token_mean does not support OPD-full or NLL")
        if seq_logprob_error_in_loss:
            raise ValueError(
                "group_token_mean requires masks finalized before training"
            )

    def prepare_batch(
        self,
        *,
        sample_ids: Sequence[str],
        token_mask: torch.Tensor,
        sample_mask: torch.Tensor,
        group_indices: torch.Tensor,
        expected_group_size: int | None,
    ) -> dict[str, torch.Tensor]:
        return {
            GROUP_TOKEN_MEAN_WEIGHT: group_token_mean_weights(
                sample_ids,
                token_mask,
                sample_mask,
                group_indices=group_indices,
                expected_group_size=expected_group_size,
            )
        }

    def per_response_weights(self, data: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return each response's inverse group-token mass, shape [B]."""
        weights = data[GROUP_TOKEN_MEAN_WEIGHT]
        if weights.ndim != 1 or weights.shape != data["sample_mask"].shape:
            raise ValueError(
                f"{GROUP_TOKEN_MEAN_WEIGHT} row weights must match sample_mask[B]"
            )
        return weights.float()

    def denominator_contribution(
        self,
        data: Mapping[str, torch.Tensor],
        *,
        valid_tokens_per_response: torch.Tensor,
        sequence_count: torch.Tensor,
        token_count: torch.Tensor,
    ) -> torch.Tensor:
        """Return this batch's scalar share of complete nonempty groups."""
        return (valid_tokens_per_response * self.per_response_weights(data)).sum()

    def loss_divisor(
        self,
        *,
        global_valid_toks: torch.Tensor,
        global_valid_seqs: torch.Tensor,
        global_objective_denominator: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return this call's scalar divisor; streaming supplies one."""
        if global_objective_denominator is None:
            raise ValueError(
                "group_token_mean requires the split training path or an explicit "
                "global_objective_denominator"
            )
        return global_objective_denominator

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
        assert per_response_weights is not None
        return super().reduce(
            values * per_response_weights.unsqueeze(-1),
            token_mask=token_mask,
            sample_mask=sample_mask,
            mask=mask,
            denominator=denominator,
            per_response_weights=None,
        )


def make_objective_reduction(normalization: str) -> ObjectiveReduction:
    """Resolve the configured formula once, before training."""
    if normalization == "global_token_mean":
        return GlobalTokenMeanReduction()
    if normalization == "sequence_mean":
        return SequenceMeanReduction()
    if normalization == "group_token_mean":
        return GroupTokenMeanReduction()
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
    metrics (or losses) that do not advertise. Survivor rescaling requires
    every returned metric to be listed.
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
