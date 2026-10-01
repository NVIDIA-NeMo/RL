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
"""Multi-row (segment) support for the SingleController advantage stage.

With ``token_capture.segment_rows.enabled`` the finalizer publishes, next to
the canonical row of rollout ``i`` (sample id ``{group}_g{i}``), extra rows
``{group}_g{i}_t{j}`` for the other trainable chains of the same rollout
(pre-compaction segments, subagent sessions). Every row of a rollout shares
the rollout's reward, ``mask_sample`` and ``prompt_ids_for_adv``; the tokens,
``token_mask``, logprobs and ``truncated`` flag are per row.

This module is pure torch + stdlib so it can be unit-tested without the
trainer stack. It provides:

* :func:`resolve_row_identity` -- per-row ``rollout_local_idx`` /
  ``trace_in_rollout_idx`` / ``trace_kind`` from the batch meta tags (fallback:
  the sample-id grammar), plus a compact per-rollout key.
* :func:`rollout_vote_mask` -- the rollout's vote in the group baseline/std,
  derived from ALL of its rows (``amax`` of the per-row baseline mask) and
  read at the canonical rows, which is what the estimator sees.
* :func:`broadcast_rollout_advantages` -- copy the advantage computed on the
  canonical rows to every row of the same rollout.
* :func:`check_chunk_completeness` -- the train-pump invariant that a chunk
  holds whole groups (``groups x N`` rollouts) and whole rollouts (every row
  of a rollout, per the ``rows_in_rollout`` tag).
* :func:`accumulate_segment_stats` / :func:`reduce_segment_stats` -- the
  ``train/segments/*`` step metrics and the by-kind
  ``token_mult_prob_error{,_premask}/by_kind/<kind>`` diagnostics (port of
  v1's ``compute_multi_trace_diagnostics`` at rollout/row granularity).

Metric contract (all emitted under the ``train/`` prefix by the controller):

* ``segments/rows``, ``segments/canonical_rows``, ``segments/extra_rows``:
  row counts (canonical = ``trace_in_rollout_idx == 0``).
* ``segments/rollouts``: distinct rollouts; ``segments/rollouts_with_segments``:
  rollouts with >= 2 rows; ``segments/rows_per_rollout_mean`` / ``_max``.
* ``segments/rows_by_kind/<kind>``, ``segments/trained_rows_by_kind/<kind>``
  (rows with a non-zero final sample mask), ``segments/trainable_tokens_by_kind/<kind>``
  (``token_mask[:, 1:]`` ones on trained rows, i.e. what the loss trains on).
* ``segments/trainable_tokens_compacted_fraction``: trainable tokens that
  belong to a rollout with at least one ``compaction_segment`` row (its
  terminal row included, as in v1's per-rollout classification) divided by all
  trainable tokens; omitted when the step has no trainable token.
* ``segments/masked_row_frac_by_kind/<kind>``: among rows of that kind that
  carry tokens (``sample_mask > 0``), the fraction left with no gradient.
* ``token_mult_prob_error/by_kind/<kind>`` and
  ``token_mult_prob_error_premask/by_kind/<kind>``: token-weighted mean of
  ``exp(|generation_logprobs - prev_logprobs|)`` over ``token_mask[:, 1:]`` x
  the final sample mask (resp. the pre-gate sample mask), exactly the loss's
  ``token_mult_prob_error`` restricted to the kind. Buckets with no tokens or a
  non-finite value are omitted; nothing is emitted without ``prev_logprobs``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from nemo_rl.experience.sample_ids import try_parse_sample_id

# Known ``trace_kind`` values. Nothing below enumerates them: by-kind metrics
# bucket whatever kind string the finalizer stamped, so a new Gym chain kind
# (``compaction_summary`` arrived after ``subagent``) needs no change here.
TERMINAL_KIND = "terminal"
COMPACTION_SEGMENT_KIND = "compaction_segment"
COMPACTION_SUMMARY_KIND = "compaction_summary"
SUBAGENT_KIND = "subagent"
PLACEHOLDER_KIND = "placeholder"
_UNKNOWN_KIND = "unknown"

ROLLOUT_LOCAL_IDX_TAG = "rollout_local_idx"
TRACE_IN_ROLLOUT_IDX_TAG = "trace_in_rollout_idx"
TRACE_KIND_TAG = "trace_kind"
# Total rows the finalizer published for the row's rollout (canonical row
# included; 1 on placeholders). Lets a consumer assert it received the whole
# rollout, not only that it received no orphan.
ROWS_IN_ROLLOUT_TAG = "rows_in_rollout"


# ── row identity ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RowIdentity:
    """Per-row rollout membership for one advantage-stage chunk.

    Attributes:
        group_pos: ``(B,)`` long; position of the row's prompt group in
            first-seen order (the controller's ``_group_ids_from_meta`` order).
        rollout_local_idx: ``(B,)`` long; ``i`` of ``{group}_g{i}``.
        trace_in_rollout_idx: ``(B,)`` long; ``j`` (0 = canonical row).
        trace_kind: one string per row (``terminal`` / ``compaction_segment`` /
            ``subagent`` / ``placeholder``; ``terminal`` when untagged).
        rollout_key: ``(B,)`` long; compact id of ``(group_pos, rollout_local_idx)``
            in first-seen order, ``0..num_rollouts-1``.
        is_canonical: ``(B,)`` bool; ``trace_in_rollout_idx == 0``.
        canonical_index: ``(Rc,)`` long; row indices of the canonical rows in
            row order.
        row_to_canonical: ``(B,)`` long; for each row, the position in
            ``canonical_index`` of its rollout's canonical row, ``-1`` when the
            chunk holds no canonical row for that rollout.
        rows_in_rollout: ``(B,)`` long; the ``rows_in_rollout`` tag of each row
            (``-1`` when the row carries no such tag, e.g. rows published
            before the tag existed).
    """

    group_pos: torch.Tensor
    rollout_local_idx: torch.Tensor
    trace_in_rollout_idx: torch.Tensor
    trace_kind: list[str]
    rollout_key: torch.Tensor
    is_canonical: torch.Tensor
    canonical_index: torch.Tensor
    row_to_canonical: torch.Tensor
    rows_in_rollout: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.long)
    )

    @property
    def num_rows(self) -> int:
        return int(self.is_canonical.numel())

    @property
    def num_rollouts(self) -> int:
        return int(self.rollout_key.max().item()) + 1 if self.num_rows else 0

    @property
    def has_extra_rows(self) -> bool:
        """True when at least one row is not a canonical (trace 0) row."""
        return bool(self.num_rows) and not bool(self.is_canonical.all().item())

    @property
    def num_groups(self) -> int:
        """Distinct prompt groups in the chunk."""
        return int(self.group_pos.max().item()) + 1 if self.num_rows else 0

    @property
    def orphan_rows(self) -> int:
        """Rows whose rollout has no canonical row in this chunk."""
        return int((self.row_to_canonical < 0).sum().item())

    @property
    def rows_per_rollout(self) -> torch.Tensor:
        """``(num_rollouts,)`` long; rows the chunk holds for each rollout key."""
        return torch.bincount(self.rollout_key, minlength=self.num_rollouts)

    def rows_in_rollout_mismatches(self) -> list[str]:
        """Rollouts whose ``rows_in_rollout`` tag disagrees with the rows present.

        Returns one human-readable entry per offending rollout: a tag that
        differs between the rollout's rows, or a tag that does not equal the
        number of rows the chunk holds for that rollout. Rows without the tag
        (``-1``) are not checked, so untagged (older) rows never fail.
        """
        problems: list[str] = []
        if not self.num_rows:
            return problems
        counts = self.rows_per_rollout
        for rk in range(self.num_rollouts):
            rows = torch.nonzero(self.rollout_key == rk, as_tuple=False).reshape(-1)
            tagged = self.rows_in_rollout[rows]
            tagged = tagged[tagged >= 0]
            if tagged.numel() == 0:
                continue
            declared = sorted(set(tagged.tolist()))
            present = int(counts[rk].item())
            first = int(rows[0].item())
            where = (
                f"group_pos={int(self.group_pos[first])}, "
                f"rollout_local_idx={int(self.rollout_local_idx[first])}"
            )
            if len(declared) != 1:
                problems.append(
                    f"rollout ({where}) carries conflicting rows_in_rollout tags "
                    f"{declared}"
                )
            elif declared[0] != present:
                problems.append(
                    f"rollout ({where}) declares rows_in_rollout={declared[0]} but "
                    f"the chunk holds {present} row(s) for it"
                )
        return problems


def _tag_int(tag: Mapping[str, Any] | None, key: str) -> Optional[int]:
    if not tag:
        return None
    value = tag.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _tag_kind(tag: Mapping[str, Any] | None) -> Optional[str]:
    if not tag:
        return None
    value = tag.get(TRACE_KIND_TAG)
    if value is None:
        return None
    text = str(value)
    return text if text else None


def resolve_row_identity(
    sample_ids: Sequence[str],
    tags: Sequence[Mapping[str, Any] | None] | None = None,
) -> RowIdentity:
    """Derive rollout membership for every row of a chunk.

    The prompt group always comes from the sample id (``{group}_g{i}[_t{j}]``;
    an id outside that grammar is its own group and its own canonical rollout,
    the controller's historical lenient behaviour). ``rollout_local_idx``,
    ``trace_in_rollout_idx`` and ``trace_kind`` are read from ``tags`` when
    present and fall back to the parsed id (kind ``terminal``).

    Raises:
        ValueError: when ``tags`` is given with a different length, or when a
            rollout has more than one canonical row in the chunk.
    """
    n = len(sample_ids)
    if tags is not None and len(tags) != n:
        raise ValueError(
            f"resolve_row_identity: tags ({len(tags)}) must align 1:1 with "
            f"sample_ids ({n})"
        )
    group_pos_of: dict[str, int] = {}
    rollout_of: dict[tuple[int, int], int] = {}
    group_pos: list[int] = []
    rollout_idx: list[int] = []
    trace_idx: list[int] = []
    kinds: list[str] = []
    rollout_key: list[int] = []
    rows_in_rollout: list[int] = []
    for row, sample_id in enumerate(sample_ids):
        parsed = try_parse_sample_id(sample_id)
        if parsed is None:
            group_id, i, j = str(sample_id), 0, 0
        else:
            group_id, i, j = parsed
        tag = tags[row] if tags is not None else None
        tag_i = _tag_int(tag, ROLLOUT_LOCAL_IDX_TAG)
        tag_j = _tag_int(tag, TRACE_IN_ROLLOUT_IDX_TAG)
        if tag_i is not None:
            i = tag_i
        if tag_j is not None:
            j = tag_j
        kind = _tag_kind(tag) or TERMINAL_KIND
        tag_rows = _tag_int(tag, ROWS_IN_ROLLOUT_TAG)
        rows_in_rollout.append(tag_rows if tag_rows is not None and tag_rows >= 1 else -1)
        gp = group_pos_of.setdefault(group_id, len(group_pos_of))
        rk = rollout_of.setdefault((gp, i), len(rollout_of))
        group_pos.append(gp)
        rollout_idx.append(i)
        trace_idx.append(j)
        kinds.append(kind)
        rollout_key.append(rk)

    group_pos_t = torch.tensor(group_pos, dtype=torch.long)
    rollout_idx_t = torch.tensor(rollout_idx, dtype=torch.long)
    trace_idx_t = torch.tensor(trace_idx, dtype=torch.long)
    rollout_key_t = torch.tensor(rollout_key, dtype=torch.long)
    is_canonical = trace_idx_t == 0
    canonical_index = torch.nonzero(is_canonical, as_tuple=False).reshape(-1)

    num_rollouts = len(rollout_of)
    canonical_pos_of_rollout = torch.full((num_rollouts,), -1, dtype=torch.long)
    for pos, row in enumerate(canonical_index.tolist()):
        rk = rollout_key[row]
        if canonical_pos_of_rollout[rk] >= 0:
            raise ValueError(
                "resolve_row_identity: rollout "
                f"(group_pos={group_pos[row]}, rollout_local_idx={rollout_idx[row]}) "
                f"has more than one canonical row (sample_id={sample_ids[row]!r})"
            )
        canonical_pos_of_rollout[rk] = pos
    row_to_canonical = (
        canonical_pos_of_rollout[rollout_key_t]
        if num_rollouts
        else torch.zeros(0, dtype=torch.long)
    )
    return RowIdentity(
        group_pos=group_pos_t,
        rollout_local_idx=rollout_idx_t,
        trace_in_rollout_idx=trace_idx_t,
        trace_kind=kinds,
        rollout_key=rollout_key_t,
        is_canonical=is_canonical,
        canonical_index=canonical_index,
        row_to_canonical=row_to_canonical,
        rows_in_rollout=torch.tensor(rows_in_rollout, dtype=torch.long),
    )


def check_chunk_completeness(
    identity: RowIdentity, *, num_generations_per_prompt: int
) -> None:
    """Assert a chunk holds whole groups and whole rollouts.

    The finalizer publishes every rollout of a group (``N`` canonical rows,
    placeholders included) plus that rollout's extra rows in one put, and the
    sampler hands whole groups to the train pump, so a chunk must contain
    exactly ``num_groups x N`` distinct rollouts and, for every rollout that
    carries the ``rows_in_rollout`` tag, exactly that many rows. Anything
    else means a row-level split somewhere upstream, which would silently
    shrink the group baseline or train a rollout's rows under two different
    chunks; fail the step instead.

    Raises:
        RuntimeError: with the offending counts.
    """
    n = int(num_generations_per_prompt)
    if n < 1:
        raise ValueError(f"num_generations_per_prompt must be >= 1, got {n}")
    if not identity.num_rows:
        return
    expected = identity.num_groups * n
    if identity.num_rollouts != expected:
        raise RuntimeError(
            "segment rows: training chunk is not made of whole groups: "
            f"{identity.num_rollouts} distinct rollout(s) over "
            f"{identity.num_groups} group(s) x num_generations_per_prompt={n} "
            f"(expected {expected}); every group must publish exactly N "
            "canonical rows and the sampler must keep groups whole"
        )
    problems = identity.rows_in_rollout_mismatches()
    if problems:
        shown = "; ".join(problems[:5])
        more = f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""
        raise RuntimeError(
            "segment rows: training chunk does not hold every row of its "
            f"rollouts: {shown}{more}; a rollout's rows must be published and "
            "consumed together"
        )


# ── advantage broadcast ────────────────────────────────────────────────────


def rollout_vote_mask(
    baseline_mask: torch.Tensor,
    rollout_key: torch.Tensor,
    canonical_index: torch.Tensor,
) -> torch.Tensor:
    """The per-rollout baseline vote, read at the canonical rows.

    A rollout votes in its group's baseline/std iff at least one of its rows
    is baseline-valid (``amax`` of ``baseline_mask`` over the rollout's rows).
    This keeps the single-row invariant "a rollout that contributes gradient
    has its reward in the baseline" when the canonical row is masked (e.g. by
    the sequence-logprob gate) while a segment row of the same rollout still
    trains; the leave-one-out estimator otherwise applies the non-voting
    row's own leave-out to a sum it never entered and hands the trained
    segment a biased baseline. A rollout with every row masked still does not
    vote. The ``amax`` also carries the row weight (``sample_mask`` is a loss
    multiplier), so a rollout whose rows all share one weight votes with it.

    Args:
        baseline_mask: ``(B,)`` per-row baseline validity/weight.
        rollout_key: ``(B,)`` long from :class:`RowIdentity`.
        canonical_index: ``(Rc,)`` long from :class:`RowIdentity`.

    Returns:
        ``(Rc,)`` tensor aligned with ``canonical_index`` (the estimator's
        ``valid_mask``). When every rollout has exactly one row this equals
        ``baseline_mask.index_select(0, canonical_index)`` bit for bit.
    """
    values = baseline_mask.reshape(-1)
    if rollout_key.shape != values.shape:
        raise ValueError(
            "rollout_vote_mask: rollout_key "
            f"{tuple(rollout_key.shape)} must align with baseline_mask "
            f"{tuple(values.shape)}"
        )
    if values.numel() == 0:
        return values[:0]
    keys = rollout_key.to(device=values.device, dtype=torch.long)
    num_rollouts = int(keys.max().item()) + 1
    votes = torch.zeros(num_rollouts, dtype=values.dtype, device=values.device)
    votes = votes.scatter_reduce(0, keys, values, reduce="amax", include_self=False)
    index = canonical_index.to(device=values.device, dtype=torch.long)
    return votes[keys.index_select(0, index)]


def broadcast_rollout_advantages(
    advantages_canonical: torch.Tensor,
    row_to_canonical: torch.Tensor,
) -> torch.Tensor:
    """Expand per-canonical-row advantages to every row of the chunk.

    Args:
        advantages_canonical: ``(Rc, ...)`` estimator output computed on the
            canonical rows only (``rows = canonical_index``); for GRPO each row
            is constant along the sequence dimension.
        row_to_canonical: ``(B,)`` long from :class:`RowIdentity`; entry ``b``
            is the position in ``advantages_canonical`` of row ``b``'s rollout.

    Returns:
        ``(B, ...)`` tensor; row ``b`` is ``advantages_canonical[row_to_canonical[b]]``.

    Raises:
        ValueError: when a row has no canonical row in the chunk (``-1``) or an
            index is out of range -- both mean the finalizer/sampler contract
            (whole rollouts per chunk, canonical rows first) was broken.
    """
    if row_to_canonical.dim() != 1:
        raise ValueError(
            f"row_to_canonical must be 1-D, got shape {tuple(row_to_canonical.shape)}"
        )
    num_rows = int(row_to_canonical.numel())
    if num_rows == 0:
        return advantages_canonical[:0]
    orphan = int((row_to_canonical < 0).sum().item())
    if orphan:
        raise ValueError(
            f"broadcast_rollout_advantages: {orphan} row(s) have no canonical "
            "row in this chunk; segment rows must be published together with "
            "their rollout's canonical row"
        )
    num_canonical = int(advantages_canonical.shape[0])
    max_index = int(row_to_canonical.max().item())
    if max_index >= num_canonical:
        raise ValueError(
            f"broadcast_rollout_advantages: row_to_canonical refers to canonical "
            f"position {max_index} but only {num_canonical} canonical advantages "
            "were given"
        )
    index = row_to_canonical.to(device=advantages_canonical.device, dtype=torch.long)
    return advantages_canonical.index_select(0, index)


def subset_rows(
    tensors: Mapping[str, torch.Tensor], index: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Index every ``(B, ...)`` tensor of ``tensors`` by ``index`` along dim 0."""
    return {
        name: value.index_select(0, index.to(device=value.device, dtype=torch.long))
        for name, value in tensors.items()
    }


# ── segment stats ──────────────────────────────────────────────────────────

SEGMENT_STATS_KEYS: tuple[str, ...] = (
    "rollout_key",
    "is_canonical",
    "kinds",
    "has_tokens",
    "trained",
    "trainable_tokens",
    "err_sum_post",
    "weight_post",
    "err_sum_pre",
    "weight_pre",
)


def new_segment_stats_accumulator() -> dict[str, list]:
    """Fresh per-step accumulator; one entry per advantage-stage chunk."""
    return {key: [] for key in SEGMENT_STATS_KEYS}


def _row_vector(value: torch.Tensor, batch: int, dtype: torch.dtype) -> torch.Tensor:
    return value.detach().reshape(batch).to("cpu").to(dtype)


def _token_error_sums(
    lp_err: torch.Tensor, loss_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row ``sum(exp(|err|) * m)`` and ``sum(m)`` for ``m = loss_mask``.

    Mirrors ``ClippedPGLossFn``'s ``token_mult_prob_error`` (mask kept inside
    the exp exactly as the loss does; masked positions contribute nothing).
    Nonfinite errors on kept tokens propagate so a bucket that contains one
    is dropped by the reducer instead of being silently averaged away.
    """
    masked_err = torch.where(loss_mask > 0, lp_err * loss_mask, torch.zeros_like(lp_err))
    err_sum = (torch.exp(masked_err) * loss_mask).sum(dim=-1, dtype=torch.float64)
    weight = loss_mask.sum(dim=-1, dtype=torch.float64)
    return err_sum, weight


def accumulate_segment_stats(
    acc: dict[str, list],
    *,
    rollout_key: torch.Tensor,
    trace_kind: Sequence[str],
    final_sample_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    token_mask: torch.Tensor,
    generation_logprobs: Optional[torch.Tensor] = None,
    prev_logprobs: Optional[torch.Tensor] = None,
    pre_gate_sample_mask: Optional[torch.Tensor] = None,
    is_canonical: Optional[torch.Tensor] = None,
) -> None:
    """Append one advantage-stage chunk (everything reduced to per-row CPU scalars).

    Args:
        rollout_key: ``(B,)`` chunk-local rollout ids (``RowIdentity.rollout_key``);
            offset here so chunks can be concatenated.
        trace_kind: one kind string per row.
        final_sample_mask: ``(B,)`` mask the loss will use (after every filter
            including the sequence-logprob gate).
        sample_mask: ``(B,)`` mask as delivered by the finalizer (placeholder rows
            carry 0).
        token_mask: ``(B, S)`` generated-token mask.
        generation_logprobs / prev_logprobs: ``(B, S)``; when either is None the
            by-kind error buckets are skipped for this chunk.
        pre_gate_sample_mask: ``(B,)`` mask right before the sequence-logprob
            gate (env flag / overlong already applied); defaults to
            ``final_sample_mask`` (no gate).
        is_canonical: ``(B,)`` bool; defaults to the first row of each rollout.
    """
    batch = int(final_sample_mask.reshape(-1).numel())
    if batch == 0:
        return
    if len(trace_kind) != batch:
        raise ValueError(
            f"accumulate_segment_stats: trace_kind has {len(trace_kind)} entries "
            f"for {batch} rows"
        )
    rk = _row_vector(rollout_key, batch, torch.long)
    offset = int(acc["rollout_key"][-1].max()) + 1 if acc["rollout_key"] else 0
    if is_canonical is None:
        first_seen: set[int] = set()
        canon_list = []
        for key in rk.tolist():
            canon_list.append(key not in first_seen)
            first_seen.add(key)
        canonical = torch.tensor(canon_list, dtype=torch.bool)
    else:
        canonical = _row_vector(is_canonical, batch, torch.bool)
    final = _row_vector(final_sample_mask, batch, torch.float32)
    delivered = _row_vector(sample_mask, batch, torch.float32)
    pre_gate = (
        final.clone()
        if pre_gate_sample_mask is None
        else _row_vector(pre_gate_sample_mask, batch, torch.float32)
    )
    tok = token_mask.detach()
    if tok.dim() != 2:
        tok = tok.reshape(batch, -1)
    tok = tok[:, 1:].to("cpu").float()
    trained = final > 0
    trainable_tokens = tok.sum(dim=-1) * trained.float()

    acc["rollout_key"].append(rk + offset)
    acc["is_canonical"].append(canonical)
    acc["kinds"].append([str(k) if k else _UNKNOWN_KIND for k in trace_kind])
    acc["has_tokens"].append(delivered > 0)
    acc["trained"].append(trained)
    acc["trainable_tokens"].append(trainable_tokens)

    if generation_logprobs is None or prev_logprobs is None:
        nan = torch.full((batch,), float("nan"), dtype=torch.float64)
        acc["err_sum_post"].append(nan)
        acc["weight_post"].append(torch.zeros(batch, dtype=torch.float64))
        acc["err_sum_pre"].append(nan.clone())
        acc["weight_pre"].append(torch.zeros(batch, dtype=torch.float64))
        return
    gen = generation_logprobs.detach()[:, 1:].to("cpu").float()
    prev = prev_logprobs.detach()[:, 1:].to("cpu").float()
    lp_err = (gen - prev).abs()
    post_sum, post_w = _token_error_sums(lp_err, tok * final.unsqueeze(-1))
    pre_sum, pre_w = _token_error_sums(lp_err, tok * pre_gate.unsqueeze(-1))
    acc["err_sum_post"].append(post_sum)
    acc["weight_post"].append(post_w)
    acc["err_sum_pre"].append(pre_sum)
    acc["weight_pre"].append(pre_w)


def _emit_bucket_means(
    out: dict[str, float],
    prefix: str,
    kinds: list[str],
    err_sum: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    for kind in sorted(set(kinds)):
        rows = torch.tensor([i for i, k in enumerate(kinds) if k == kind], dtype=torch.long)
        w = float(weight[rows].sum().item())
        if w <= 0:
            continue
        mean = float(err_sum[rows].sum().item()) / w
        if math.isfinite(mean):
            out[f"{prefix}/by_kind/{kind}"] = mean


def reduce_segment_stats(acc: dict[str, list]) -> dict[str, float]:
    """Reduce a step's chunks into the ``segments/*`` and by-kind error metrics."""
    if not acc.get("rollout_key"):
        return {}
    rollout_key = torch.cat(acc["rollout_key"])
    canonical = torch.cat(acc["is_canonical"])
    kinds: list[str] = [k for chunk in acc["kinds"] for k in chunk]
    has_tokens = torch.cat(acc["has_tokens"])
    trained = torch.cat(acc["trained"])
    trainable_tokens = torch.cat(acc["trainable_tokens"])
    rows = int(rollout_key.numel())
    if rows == 0:
        return {}

    num_rollouts = int(rollout_key.max().item()) + 1
    rows_per_rollout = torch.bincount(rollout_key, minlength=num_rollouts)
    populated = rows_per_rollout[rows_per_rollout > 0]
    out: dict[str, float] = {
        "segments/rows": float(rows),
        "segments/canonical_rows": float(canonical.sum()),
        "segments/extra_rows": float((~canonical).sum()),
        "segments/rollouts": float(populated.numel()),
        "segments/rollouts_with_segments": float((populated >= 2).sum()),
        "segments/rows_per_rollout_mean": (
            float(populated.float().mean()) if populated.numel() else 0.0
        ),
        "segments/rows_per_rollout_max": (
            float(populated.max()) if populated.numel() else 0.0
        ),
    }

    kind_rows: dict[str, list[int]] = {}
    for row, kind in enumerate(kinds):
        kind_rows.setdefault(kind, []).append(row)
    for kind in sorted(kind_rows):
        idx = torch.tensor(kind_rows[kind], dtype=torch.long)
        out[f"segments/rows_by_kind/{kind}"] = float(len(kind_rows[kind]))
        out[f"segments/trained_rows_by_kind/{kind}"] = float(trained[idx].sum())
        out[f"segments/trainable_tokens_by_kind/{kind}"] = float(
            trainable_tokens[idx].sum()
        )
        with_tokens = has_tokens[idx]
        n_with = int(with_tokens.sum())
        if n_with:
            masked = with_tokens & ~trained[idx]
            out[f"segments/masked_row_frac_by_kind/{kind}"] = float(masked.sum()) / n_with

    total_trainable = float(trainable_tokens.sum())
    if total_trainable > 0:
        compacted_rollouts = torch.zeros(num_rollouts, dtype=torch.bool)
        for row, kind in enumerate(kinds):
            if kind == COMPACTION_SEGMENT_KIND:
                compacted_rollouts[int(rollout_key[row])] = True
        compacted_rows = compacted_rollouts[rollout_key]
        out["segments/trainable_tokens_compacted_fraction"] = (
            float(trainable_tokens[compacted_rows].sum()) / total_trainable
        )

    _emit_bucket_means(
        out,
        "token_mult_prob_error",
        kinds,
        torch.cat(acc["err_sum_post"]),
        torch.cat(acc["weight_post"]),
    )
    _emit_bucket_means(
        out,
        "token_mult_prob_error_premask",
        kinds,
        torch.cat(acc["err_sum_pre"]),
        torch.cat(acc["weight_pre"]),
    )
    return out
