# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve original context-segment TMPE when its calls are separate rows."""

from dataclasses import dataclass
from typing import Any

import torch

from nemo_rl.algorithms.utils import compute_seq_logprob_errors
from nemo_rl.data_plane import KVBatchMeta

_CALL_TAGS = frozenset(
    {"context_segment_index", "context_segment_count", "call_index", "call_count"}
)


@dataclass(frozen=True)
class CallReplayMetrics:
    """Diagnostics only; neither masks nor probability tensors are modified."""

    metrics: dict[str, float]
    valid_context_count: int


def _group_rows(meta: KVBatchMeta) -> tuple[list[int], list[bool], int] | None:
    """Validate complete call contexts while allowing saved legacy contexts."""
    tags = meta.tags
    if tags is None or not any(_CALL_TAGS.intersection(tag) for tag in tags):
        return None
    if len(tags) != meta.size:
        raise ValueError("call replay metadata must be row-aligned")
    group_ids: list[int] = []
    is_call: list[bool] = []
    groups: dict[tuple[Any, ...], int] = {}
    calls: dict[tuple[str, str, int], tuple[int, set[int]]] = {}
    contexts: dict[tuple[str, str], tuple[int, set[int]]] = {}
    legacy_owners: set[tuple[str, str]] = set()
    for index, tag in enumerate(tags):
        if tag.get("is_execution_padding"):
            group_ids.append(-1)
            is_call.append(False)
            continue
        if not _CALL_TAGS.intersection(tag):
            key = ("legacy", index)
            if "dispatch_group_id" in tag and "logical_rollout_id" in tag:
                legacy_owners.add((tag["dispatch_group_id"], tag["logical_rollout_id"]))
            is_call.append(False)
        else:
            if not _CALL_TAGS.issubset(tag):
                raise ValueError("incomplete call replay metadata")
            group, owner = tag.get("dispatch_group_id"), tag.get("logical_rollout_id")
            context, context_count = (
                tag["context_segment_index"],
                tag["context_segment_count"],
            )
            call, call_count = tag["call_index"], tag["call_count"]
            if (
                not isinstance(group, str)
                or not group
                or not isinstance(owner, str)
                or not owner
                or type(context_count) is not int
                or context_count < 1
                or type(context) is not int
                or not 0 <= context < context_count
                or type(call_count) is not int
                or call_count < 0
                or type(call) is not int
                or not 0 <= call < max(1, call_count)
                or (call_count == 0 and not tag.get("uses_borrowed_input"))
            ):
                raise ValueError("invalid call replay context or action identity")
            identity = (group, owner, context)
            expected, seen = calls.setdefault(identity, (max(1, call_count), set()))
            if expected != max(1, call_count) or call in seen:
                raise ValueError("duplicate call or inconsistent call count")
            seen.add(call)
            expected_contexts, seen_contexts = contexts.setdefault(
                (group, owner), (context_count, set())
            )
            if expected_contexts != context_count:
                raise ValueError("inconsistent context count")
            seen_contexts.add(context)
            key = ("calls", *identity)
            is_call.append(True)
        group_ids.append(groups.setdefault(key, len(groups)))
    if any(seen != set(range(count)) for count, seen in calls.values()):
        raise ValueError("call replay context is missing selected actions")
    if any(seen != set(range(count)) for count, seen in contexts.values()):
        raise ValueError("call replay owner is missing context segments")
    if legacy_owners.intersection(contexts):
        raise ValueError("one owner mixes legacy and call replay representations")
    return group_ids, is_call, len(groups)


def _statistics(errors: torch.Tensor, name: str) -> dict[str, float]:
    return {
        f"max_{name}_mult_prob_error": errors.max().item() if errors.numel() else 0.0,
        f"mean_{name}_mult_prob_error": errors.mean().item() if errors.numel() else 0.0,
        f"min_{name}_mult_prob_error": errors.min().item() if errors.numel() else 0.0,
    }


def compute_call_replay_metrics(
    meta: KVBatchMeta,
    *,
    policy_logprobs: torch.Tensor,
    generation_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
) -> CallReplayMetrics | None:
    """Rejoin every original context's targets before sequence error statistics.

    Full token-aligned inputs include the first token. Multiplicative errors
    use the existing estimator; call means are weighted by their exact target
    counts before being combined. Call maxima remain separately visible.
    Old canonical rows restored from a checkpoint remain separate sequences;
    they are counted explicitly and never dropped or relabeled as call replay.
    """
    grouped = _group_rows(meta)
    if grouped is None:
        return None
    if (
        token_mask.ndim != 2
        or token_mask.shape[0] != meta.size
        or policy_logprobs.shape != token_mask.shape
        or generation_logprobs.shape != token_mask.shape
        or sample_mask.shape != (meta.size,)
    ):
        raise ValueError("call replay metrics require aligned probability tensors")
    row_groups, call_rows, num_groups = grouped
    errors, valid = compute_seq_logprob_errors(
        policy_logprobs=policy_logprobs[:, 1:],
        generation_logprobs=generation_logprobs[:, 1:],
        token_mask=token_mask[:, 1:],
        sample_mask=sample_mask,
    )
    counts = (token_mask[:, 1:] * sample_mask.unsqueeze(-1)).sum(-1).to(errors.dtype)
    group_ids = torch.tensor(row_groups, dtype=torch.long, device=errors.device)
    included = group_ids >= 0
    if valid[~included].any():
        raise ValueError("execution padding cannot own call replay targets")
    totals = torch.zeros(num_groups, dtype=errors.dtype, device=errors.device)
    denominators = torch.zeros_like(totals)
    totals.scatter_add_(0, group_ids[included], (errors * counts)[included])
    denominators.scatter_add_(0, group_ids[included], counts[included])
    valid_contexts = denominators > 0
    context_errors = totals[valid_contexts] / denominators[valid_contexts]
    calls = torch.tensor(call_rows, dtype=torch.bool, device=errors.device)
    call_errors = errors[calls & valid]
    metrics = _statistics(context_errors, "seq")
    metrics.update({f"{key}_after_mask": value for key, value in list(metrics.items())})
    metrics.update(_statistics(call_errors, "call"))
    metrics["_num_valid_calls"] = float(call_errors.numel())
    metrics["tmpe_legacy_context_row_count"] = float(
        (~calls & included & valid).sum().item()
    )
    return CallReplayMetrics(metrics, int(valid_contexts.sum().item()))
