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

"""Helpers used by SingleControllerActor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.metric_utils import REWARD_KEY
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import (
    ROLLOUT_ENVIRONMENT_TAG,
    UNKNOWN_ROLLOUT_ENVIRONMENT,
)

# Reduction rules for all_mb_metrics. Mirror grpo.py / grpo_sync.py.
_MB_METRIC_MIN: frozenset[str] = frozenset(
    {"probs_ratio_min", "probs_ratio_clamped_min", "opd_full_reverse_kl_min"}
)
_MB_METRIC_MAX: frozenset[str] = frozenset(
    {
        "probs_ratio_max",
        "probs_ratio_clamped_max",
        "opd_full_reverse_kl_max",
        "opd_full_decomposition_error",
    }
)
_MB_METRIC_MEAN: frozenset[str] = frozenset(
    {
        "lr",
        "wd",
        "reward",
        "global_valid_seqs",
        "global_valid_toks",
        "mean_prompt_length",
    }
)


def aggregate_step_metrics(train_result: dict[str, Any]) -> dict[str, Any]:
    """Reduce per-microbatch metric lists into step-level scalars.

    Args:
        train_result: Output of TQPolicy.finish_train_step.

    Returns:
        Flat dict of step-level scalars ready for logging.
    """
    metrics: dict[str, Any] = {}
    loss = train_result.get("loss")
    if isinstance(loss, torch.Tensor):
        metrics["loss"] = loss.detach().mean().item()
    elif loss is not None:
        metrics["loss"] = float(loss)
    grad_norm = train_result.get("grad_norm")
    if isinstance(grad_norm, torch.Tensor):
        metrics["grad_norm"] = grad_norm.detach().mean().item()
    elif grad_norm is not None:
        metrics["grad_norm"] = float(grad_norm)
    draft_grad_norm = train_result.get("draft_grad_norm")
    if isinstance(draft_grad_norm, torch.Tensor):
        metrics["draft_grad_norm"] = draft_grad_norm.detach().mean().item()
    elif draft_grad_norm is not None:
        metrics["draft_grad_norm"] = float(draft_grad_norm)
    if "total_flops" in train_result:
        metrics["total_flops"] = float(train_result["total_flops"])
    if "num_ranks" in train_result:
        metrics["num_ranks"] = int(train_result["num_ranks"])

    # moe/mtp share the same reduction rules as all_mb_metrics in grpo.py.
    mb: dict[str, list[Any]] = {}
    if "moe_metrics" in train_result:
        mb.update({f"moe/{k}": v for k, v in train_result["moe_metrics"].items()})
    if "mtp_metrics" in train_result:
        mb.update({f"mtp/{k}": v for k, v in train_result["mtp_metrics"].items()})
    mb.update(train_result.get("all_mb_metrics", {}))

    for k, v in mb.items():
        if k in _MB_METRIC_MIN:
            valid = [x for x in v if not np.isinf(x)]
            metrics[k] = float(np.min(valid)) if valid else -1.0
        elif k in _MB_METRIC_MAX:
            valid = [x for x in v if not np.isinf(x)]
            metrics[k] = float(np.max(valid)) if valid else -1.0
        elif k in _MB_METRIC_MEAN:
            metrics[k] = float(np.mean(v))
        else:
            metrics[k] = float(np.sum(v))

    # Deferred-mode counterpart to rollout_reassembler's direct-mode
    # routed_experts_row_coverage/routed_experts_sentinel_token_fraction:
    # this step's per-reason count of rows whose route fragments failed to
    # reassemble at the policy worker (see TQWorkerMixin._route_fallback_counts).
    # Silent otherwise -- the only other trace is a logging.warning.
    for reason, count in train_result.get("route_fallback_counts", {}).items():
        metrics[f"routed_experts_deferred_fallback/{reason}"] = float(count)

    return metrics


@dataclass(frozen=True)
class RewardPartial:
    """One advantage-stage call's reward sums, already reduced.

    Rewards are per-row rather than per-token, so holding the tensors was never
    the bulk of the controller's memory. They are reduced here anyway because
    the advantage stage is moving into its own actor, and the whole point of
    that boundary is that nothing cohort-sized crosses it.
    """

    weighted_total: float
    weight: float

    @classmethod
    def from_rows(
        cls,
        rewards: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
    ) -> RewardPartial:
        """Reduce one call's rewards, weighting by row validity when given.

        An absent mask weights every row equally, so the merged result is the
        plain mean and callers that never had a mask keep their old numbers.
        """
        flat = rewards.flatten()
        if sample_mask is None:
            return cls(
                weighted_total=float(flat.sum(dtype=torch.float64)),
                weight=float(flat.numel()),
            )
        mask = sample_mask.flatten().to(flat.dtype)
        return cls(
            weighted_total=float((flat * mask).sum(dtype=torch.float64)),
            weight=float(mask.sum(dtype=torch.float64)),
        )


@dataclass(frozen=True)
class AdvantagePartial:
    """One advantage-stage call's token-masked advantage moments.

    Replaces keeping ``torch.masked_select(advantages, mask)`` itself, which is
    one float per trained token across the entire cohort, appended once per
    streaming chunk and then concatenated at step close -- so the step's peak
    was twice the accumulated size. Only the mean, min and max were ever read
    off it, and all three merge from these four numbers.
    """

    count: int
    total: float
    minimum: float
    maximum: float

    @classmethod
    def from_values(cls, values: torch.Tensor) -> AdvantagePartial:
        """Reduce one call's masked advantages; an empty selection counts zero."""
        if values.numel() == 0:
            return cls(count=0, total=0.0, minimum=0.0, maximum=0.0)
        return cls(
            count=int(values.numel()),
            total=float(values.sum(dtype=torch.float64)),
            minimum=float(values.min()),
            maximum=float(values.max()),
        )


def reduce_advantage_pump_metrics(
    reward_partials: list[RewardPartial],
    advantage_partials: list[AdvantagePartial],
    sequence_lengths: list[int],
    *,
    seq_logprob_error_metrics: list[dict[str, float]] | None = None,
    num_mask_sample_filtered: list[int] | None = None,
    environment_counts: list[dict[str, float]] | None = None,
    num_invalid_tool_calls: list[int] | None = None,
    num_malformed_thinking: list[int] | None = None,
    num_assistant_messages: list[int] | None = None,
    num_routed_experts_backfilled: list[int] | None = None,
) -> dict[str, float]:
    """Reduce per-step accumulators from _advantage_stage into step scalars.

    Args:
        reward_partials: One record per advantage_stage call. Already weighted
            by row validity when the caller had a sample mask (token-capture
            placeholders and mask_sample/overlong/seq-logprob-error rows carry
            0), so ``reward`` averages over trained rows only, matching what
            the advantage estimator's baseline already excludes.
        advantage_partials: One record per advantage_stage call, over that
            call's token-masked advantages.
        sequence_lengths: All input_lengths trained on this step.
        seq_logprob_error_metrics: Sequence-error metrics and their aggregation
            counts, one record per streaming chunk.
        num_mask_sample_filtered: Environment-flagged sample counts, one per
            streaming chunk.
        environment_counts: Per-environment training counts for selected chunks.
        num_invalid_tool_calls: Per-sample invalid tool-call counts.
        num_malformed_thinking: Per-sample malformed-thinking counts.
        num_assistant_messages: Per-sample assistant message counts (rate denominator).

    Returns:
        Step-level reward, advantage, token-count, optional sequence
        log-probability error metrics, the num_mask_sample_filtered count, and
        per-sample violation counts.

    """
    out: dict[str, float] = {}
    if reward_partials:
        weight = sum(partial.weight for partial in reward_partials)
        out[REWARD_KEY] = (
            sum(partial.weighted_total for partial in reward_partials) / weight
            if weight > 0
            else 0.0
        )
    if advantage_partials:
        # A call whose mask selected nothing carries no min or max to merge.
        populated = [partial for partial in advantage_partials if partial.count]
        if populated:
            count = sum(partial.count for partial in populated)
            out["advantages/mean"] = sum(partial.total for partial in populated) / count
            out["advantages/max"] = max(partial.maximum for partial in populated)
            out["advantages/min"] = min(partial.minimum for partial in populated)
        else:
            out["advantages/mean"] = 0.0
            out["advantages/max"] = 0.0
            out["advantages/min"] = 0.0
    if sequence_lengths:
        out["total_num_tokens"] = float(sum(sequence_lengths))
    if num_mask_sample_filtered is not None:
        out["num_mask_sample_filtered"] = float(sum(num_mask_sample_filtered))
    for counts in environment_counts or []:
        for key, value in counts.items():
            out[key] = out.get(key, 0.0) + value
    if seq_logprob_error_metrics:
        out.update(_reduce_seq_logprob_error_metrics(seq_logprob_error_metrics))
    n_asst = sum(num_assistant_messages or [])
    if n_asst:
        n_invalid = sum(num_invalid_tool_calls or [])
        n_malformed = sum(num_malformed_thinking or [])
        out["invalid_tool_call_rate"] = n_invalid / n_asst
        out["malformed_thinking_rate"] = n_malformed / n_asst
        out["num_invalid_tool_calls"] = float(n_invalid)
        out["num_malformed_thinking"] = float(n_malformed)
        out["num_assistant_messages"] = float(n_asst)
        # Router-replay partial loss: a message sentinel-filled inside an
        # otherwise-routed rollout, invisible to the replay_buffer's
        # field-entirely-absent guard (see backfill_missing_routed_experts).
        n_backfilled = sum(num_routed_experts_backfilled or [])
        out["routed_experts_backfilled_rate"] = n_backfilled / n_asst
        out["num_routed_experts_backfilled"] = float(n_backfilled)
    return out


def environment_sample_counts(
    tags: list[dict[str, Any]] | None,
    *,
    mask_sample: torch.Tensor,
    final_sample_mask: torch.Tensor,
    final_token_mask: torch.Tensor,
) -> dict[str, float]:
    """Count selected rows and trainable next-token targets by environment.

    Older replay checkpoints lack environment tags and are reported as unknown.
    Environment flags count independently of other, potentially overlapping filters.
    Valid samples sum the final sample weights, as in the policy loss; valid tokens
    sum the weighted next-token mask, excluding the first sequence position.

    Args:
        tags: One data-plane tag dict per selected sample, or None when the
            rows carry no tags at all.
        mask_sample: Bool tensor, True where the environment flagged the row.
        final_sample_mask: Per-sample loss weights after every filter.
        final_token_mask: Per-token loss mask already multiplied by
            final_sample_mask.

    Returns:
        Dict mapping "environment/<name>/<counter>" to its total for this
        chunk, for counters num_samples, num_mask_sample_filtered,
        num_valid_samples and num_valid_tokens.

    Raises:
        ValueError: If tags is given but its length does not match the
            number of selected samples.
    """
    size = mask_sample.numel()
    if tags is not None and len(tags) != size:
        raise ValueError("Environment tags must align with selected samples")
    environments = (
        [tag.get(ROLLOUT_ENVIRONMENT_TAG, UNKNOWN_ROLLOUT_ENVIRONMENT) for tag in tags]
        if tags is not None
        else [UNKNOWN_ROLLOUT_ENVIRONMENT] * size
    )
    valid_tokens = final_token_mask[:, 1:].sum(dim=-1).detach().cpu().tolist()
    valid_samples = final_sample_mask.detach().cpu().tolist()
    flagged = mask_sample.detach().cpu().tolist()
    counts: dict[str, float] = {}
    for environment, tokens, valid, masked in zip(
        environments, valid_tokens, valid_samples, flagged, strict=True
    ):
        prefix = f"environment/{environment}"
        for name, value in (
            ("num_samples", 1),
            ("num_mask_sample_filtered", int(masked)),
            ("num_valid_samples", valid),
            ("num_valid_tokens", tokens),
        ):
            key = f"{prefix}/{name}"
            counts[key] = counts.get(key, 0.0) + value
    return counts


def _reduce_seq_logprob_error_metrics(
    records: list[dict[str, float]],
) -> dict[str, float]:
    """Reduce sequence-error metrics across streaming chunks."""

    def reduce_range(
        *,
        count_key: str,
        max_key: str,
        mean_key: str,
        min_key: str,
    ) -> dict[str, float]:
        populated = [record for record in records if record[count_key] > 0]
        count = sum(record[count_key] for record in populated)
        if not count:
            return {max_key: 0.0, mean_key: 0.0, min_key: 0.0}
        return {
            max_key: max(record[max_key] for record in populated),
            mean_key: sum(record[mean_key] * record[count_key] for record in populated)
            / count,
            min_key: min(record[min_key] for record in populated),
        }

    reduced = reduce_range(
        count_key="_num_valid_seqs_before",
        max_key="max_seq_mult_prob_error",
        mean_key="mean_seq_mult_prob_error",
        min_key="min_seq_mult_prob_error",
    )
    reduced.update(
        reduce_range(
            count_key="_num_valid_seqs_after",
            max_key="max_seq_mult_prob_error_after_mask",
            mean_key="mean_seq_mult_prob_error_after_mask",
            min_key="min_seq_mult_prob_error_after_mask",
        )
    )

    masked_count = sum(record["num_masked_seqs_by_logprob_error"] for record in records)
    reduced["num_masked_seqs_by_logprob_error"] = int(masked_count)
    reduced["masked_correct_pct"] = (
        sum(
            record["masked_correct_pct"] * record["num_masked_seqs_by_logprob_error"]
            for record in records
        )
        / masked_count
        if masked_count
        else 0.0
    )
    return reduced


def apply_message_level_advantage_penalties(
    advantages: torch.Tensor,
    *,
    invalid_tool_call_mask: torch.Tensor,
    malformed_thinking_mask: torch.Tensor,
    invalid_tool_call_advantage: float | None,
    malformed_thinking_advantage: float | None,
) -> torch.Tensor:
    """Overwrite flagged token advantages while leaving valid tokens unchanged.

    Invalid-tool-call penalties take precedence when both masks select the same
    token, matching the legacy GRPO message-level implementation.

    Args:
        advantages: Per-token advantages of shape (batch, seq).
        invalid_tool_call_mask: Bool mask of the same shape as ``advantages``;
            True at tokens produced by an invalid tool call.
        malformed_thinking_mask: Bool mask of the same shape as ``advantages``;
            True at tokens produced by malformed thinking.
        invalid_tool_call_advantage: Value to overwrite flagged tokens with, or
            ``None`` to leave the invalid-tool-call branch disabled.
        malformed_thinking_advantage: Value to overwrite flagged tokens with, or
            ``None`` to leave the malformed-thinking branch disabled.

    Returns:
        New tensor with penalties applied. The original ``advantages`` object is
        returned unchanged when both advantages are ``None``.

    Raises:
        ValueError: If either mask shape does not match ``advantages``.
    """
    if invalid_tool_call_mask.shape != advantages.shape:
        raise ValueError(
            "invalid_tool_call_mask shape "
            f"{tuple(invalid_tool_call_mask.shape)} does not match advantages "
            f"{tuple(advantages.shape)}"
        )
    if malformed_thinking_mask.shape != advantages.shape:
        raise ValueError(
            "malformed_thinking_mask shape "
            f"{tuple(malformed_thinking_mask.shape)} does not match advantages "
            f"{tuple(advantages.shape)}"
        )

    result = advantages
    if malformed_thinking_advantage is not None:
        result = torch.where(
            malformed_thinking_mask.bool(),
            torch.as_tensor(
                malformed_thinking_advantage,
                dtype=advantages.dtype,
                device=advantages.device,
            ),
            result,
        )
    if invalid_tool_call_advantage is not None:
        result = torch.where(
            invalid_tool_call_mask.bool(),
            torch.as_tensor(
                invalid_tool_call_advantage,
                dtype=advantages.dtype,
                device=advantages.device,
            ),
            result,
        )
    return result


def tensor_field(data: TensorDict, field_name: str) -> torch.Tensor:
    """Read a tensor column from a TensorDict, depadding if nested.

    Args:
        data: TensorDict returned by the data plane.
        field_name: Column name to fetch.

    Returns:
        Dense tensor (nested columns are padded with zeros).
    """
    value = data[field_name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor field {field_name!r}; got {type(value)}")
    if value.is_nested:
        return torch.nested.to_padded_tensor(value, padding=0)
    return value


def squeeze_trailing_unit_dim(value: torch.Tensor) -> torch.Tensor:
    """Drop a trailing dim of size 1 if present.

    Args:
        value: Input tensor.

    Returns:
        Tensor without the trailing unit dim.
    """
    if value.dim() >= 2 and value.shape[-1] == 1:
        return value.squeeze(-1)
    return value


def fields_for_put(meta: KVBatchMeta, fields: dict[str, torch.Tensor]) -> TensorDict:
    """Pack tensors for DataPlane put, re-nesting jagged rows when needed.

    Args:
        meta: Batch meta whose sequence_lengths drive the nesting.
        fields: Field name to dense tensor.

    Returns:
        TensorDict shaped for dp_client.put_samples.
    """
    packed: dict[str, torch.Tensor] = {}
    if meta.sequence_lengths is None:
        for field_name, value in fields.items():
            packed[field_name] = value.detach().contiguous()
        # pyrefly: ignore[bad-argument-type]
        return TensorDict(packed, batch_size=[meta.size])

    lengths = torch.tensor(meta.sequence_lengths, dtype=torch.long)
    for field_name, value in fields.items():
        if value.dim() >= 2 and value.shape[1] == int(lengths.max().item()):
            rows = [
                value[i, : int(lengths[i].item())].detach().contiguous()
                for i in range(meta.size)
            ]
            packed[field_name] = torch.nested.as_nested_tensor(
                rows,
                layout=torch.jagged,
            )
        else:
            packed[field_name] = value.detach().contiguous()
    # pyrefly: ignore[bad-argument-type]
    return TensorDict(packed, batch_size=[meta.size])
