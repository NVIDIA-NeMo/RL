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
"""Pure metric helpers for multi-trace (compaction / subagent) GRPO batches.

Two things live here so they can be unit-tested without Ray or a policy:

* :func:`compute_multi_trace_diagnostics` — per-trace diagnostics computed in
  the async GRPO loop once the seq-logprob-error gate has run: seq error and
  gate-mask rate bucketed by trace index within its rollout and by segment
  kind, the decomposition of ``multi_trace/masked_trace_fraction`` into its
  causes, per-kind token share, and |logprob error| by relative position;
  plus length-robust per-trace |logprob error| statistics (mean / max |Δ|,
  generated-token count, Jensen-consistent alternative gate rate) and
  trainable-token counts by segment kind, compaction status and per rollout.
* :func:`finalize_sum_count_metrics` — turns the exactly-summed ``X/sum`` and
  ``X/count`` rollout metrics (emitted per prompt group by
  ``rollouts._compaction_rollout_metrics`` and friends) into ``X/mean`` and,
  for rollout-level event counts, ``X/rate``.

No key is ever emitted with a NaN/inf value; buckets without data are omitted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

# Rollout-level event counters that get a `/rate` (= count / rollouts/count)
# in addition to the raw count. Keys are matched on prefix or exactly.
_RATE_COUNT_PREFIXES: tuple[str, ...] = ("termination/", "mask_sample/by_kind/")
_RATE_COUNT_KEYS: frozenset[str] = frozenset(
    {"format/think_tag_violation/count", "reward/masked_resolved/count"}
)
ROLLOUT_COUNT_KEY = "rollouts/count"

# Segment kinds (``rollouts._trace_kind``) that only occur in a rollout that
# compacted at least once. The compacted/uncompacted trainable-token roll-ups
# in :func:`compute_multi_trace_diagnostics` sum COMPACTED_KINDS vs the single
# ``uncompacted`` kind; ``subagent``, ``empty`` and ``unknown`` traces are in
# NEITHER roll-up (they still count towards ``trainable_tokens/total``). A
# rollout is classified "compacted" iff any of its traces has one of these kinds.
COMPACTED_KINDS: frozenset[str] = frozenset(
    {"pre_compaction", "compaction_summary", "post_compaction"}
)
_UNCOMPACTED_KIND = "uncompacted"


def trace_index_bucket(trace_in_rollout_idx: int) -> str:
    """Bucket a trace's index within its rollout as ``0``, ``1`` or ``2plus``."""
    idx = int(trace_in_rollout_idx)
    if idx <= 0:
        return "0"
    if idx == 1:
        return "1"
    return "2plus"


def _finite_mean(values: torch.Tensor) -> float | None:
    """Mean over finite entries, or None when there is nothing to average."""
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return None
    return float(finite.float().mean().item())


def _finite_max(values: torch.Tensor) -> float | None:
    """Max over finite entries, or None when there is nothing to reduce."""
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return None
    return float(finite.float().max().item())


def compute_multi_trace_diagnostics(
    *,
    seq_mult_prob_error: torch.Tensor,
    masked_by_seq_logprob_error: torch.Tensor,
    pre_seq_error_sample_loss_mask: torch.Tensor,
    mask_sample: torch.Tensor | Sequence[bool],
    is_empty_rollout: torch.Tensor | Sequence[bool],
    trace_in_rollout_idx: torch.Tensor | Sequence[int],
    trace_kinds: Sequence[str],
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    generation_logprobs: torch.Tensor,
    prev_logprobs: torch.Tensor,
    num_unpadded_traces: int,
    seq_logprob_error_threshold: float | None = None,
    trace_rollout_ids: torch.Tensor | Sequence[int] | None = None,
) -> dict[str, float]:
    """Per-trace diagnostics for a multi-trace training batch.

    All tensor arguments are indexed by trace row (the first ``num_unpadded_traces``
    rows are real traces; anything beyond is DP padding and ignored). ``[B, S]``
    tensors (``token_mask``, ``sample_mask`` is ``[B]``, ``generation_logprobs``,
    ``prev_logprobs``) are consumed on the ``[:, 1:]`` slice exactly like the loss.

    Args:
        seq_mult_prob_error: per-trace mean ``exp|logp_train - logp_gen|`` (0 for
            traces that had no valid token before the gate).
        masked_by_seq_logprob_error: per-trace bool, True if the seq-error gate
            zeroed the trace.
        pre_seq_error_sample_loss_mask: per-trace sample mask BEFORE the gate
            (>0 means the trace was going to train).
        mask_sample: per-trace env mask flag (True = env asked to exclude).
        is_empty_rollout: per-trace flag for the empty-rollout dummy trace.
        trace_in_rollout_idx: per-trace index within its rollout.
        trace_kinds: per-trace segment kind (``rollouts._trace_kind`` output;
            anything missing is bucketed as ``unknown``).
        token_mask: ``[B, S]`` generated-token mask.
        sample_mask: ``[B]`` sample mask AFTER the gate (what the loss uses).
        generation_logprobs / prev_logprobs: ``[B, S]`` inference-time and
            current-policy logprobs.
        num_unpadded_traces: number of real trace rows.
        seq_logprob_error_threshold: the live gate's threshold
            (``grpo.seq_logprob_error_threshold``). Only used, when set and > 1,
            to emit ``logprob_error/alt_gate_masked_fraction/*`` (see below).
        trace_rollout_ids: optional per-trace rollout id (the positional
            ``trace_rollout_ids`` of the async loop; rows beyond
            ``num_unpadded_traces`` are ignored). Enables the per-rollout
            ``multi_trace/rollouts/*``,
            ``multi_trace/trainable_tokens_per_rollout/*`` and
            ``multi_trace/traces_per_rollout/*`` keys.

    Length-robust log-prob error statistics. Emitted per bucket for both
    groupings (``by_trace_in_rollout_idx`` and ``by_segment_kind``) over the
    gate-eligible traces (pre-gate mask > 0 and >= 1 generated token). The token
    set is the trace's GENERATED tokens (``token_mask``), independent of the
    post-gate sample mask, so gated-out traces are measured too:

    * ``logprob_error/seq_mean_abs_err/{grouping}/{name}/mean`` — mean over
      traces of the per-trace mean |logp_gen - logp_train|.
    * ``logprob_error/seq_max_abs_err/{grouping}/{name}/{mean,max}`` — mean and
      max over traces of the per-trace max |logp_gen - logp_train|.
    * ``logprob_error/seq_gen_tokens/{grouping}/{name}/mean`` — mean generated
      token count per trace, to read the live gate's length dependence next to
      ``seq_masked_fraction``.
    * ``logprob_error/alt_gate_masked_fraction/{grouping}/{name}`` — only when
      ``seq_logprob_error_threshold`` is set and > 1: fraction of eligible traces
      with mean |Δ| > ln(threshold). This is the Jensen-consistent, length-robust
      counterpart of the live gate: the live gate masks when the arithmetic mean
      of exp|Δ| exceeds the threshold, and exp(mean|Δ|) <= mean exp|Δ|, so this
      alternative masks a SUBSET of what the live gate masks (a single outlier
      token can no longer mask a long trace). Diagnostic only — the live gate
      is unchanged.

    Trainable token counts. ``valid = token_mask * (sample_mask > 0)`` on the
    ``[:, 1:]`` slice is exactly the token set the loss trains on after the gate:

    * ``multi_trace/trainable_tokens/total``,
      ``multi_trace/trainable_tokens/by_segment_kind/{kind}`` (absolute counts;
      0 for a kind that is present without trainable tokens),
      ``multi_trace/trainable_tokens/compacted_rollouts`` (sum over
      :data:`COMPACTED_KINDS`), ``.../uncompacted_rollouts`` (kind
      ``uncompacted``) and ``.../compacted_fraction`` (compacted / total, only
      when total > 0). ``subagent`` / ``empty`` / ``unknown`` tokens are in
      neither roll-up (but in the total).
    * With ``trace_rollout_ids``: a rollout is compacted iff any of its traces
      has a kind in :data:`COMPACTED_KINDS`, else uncompacted.
      ``multi_trace/rollouts/{compacted,uncompacted}_count`` (always emitted),
      ``multi_trace/trainable_tokens_per_rollout/{class}/mean`` (valid tokens
      summed over the rollout's traces, averaged over the rollouts of that
      class) and ``multi_trace/traces_per_rollout/{class}/mean`` (both only for
      a non-empty class).

    Rationale: the token-level loss is normalized by the global valid-token
    count, so a rollout's weight in the step is proportional to its trainable
    tokens. These counts show how much of each step's gradient comes from
    compacted rollouts and how much a per-rollout normalization would
    rebalance.

    Returns:
        Flat ``{metric_key: float}`` dict. Bucket keys are present only when the
        bucket has data. Never contains NaN or inf.
    """
    n = int(num_unpadded_traces)
    if n <= 0:
        return {}

    def _as_tensor(x, dtype):
        t = x if torch.is_tensor(x) else torch.as_tensor(list(x))
        return t[:n].detach().to("cpu").to(dtype)

    seq_err = _as_tensor(seq_mult_prob_error, torch.float32)
    seq_masked = _as_tensor(masked_by_seq_logprob_error, torch.bool)
    pre_gate_unmasked = _as_tensor(pre_seq_error_sample_loss_mask, torch.float32) > 0
    env_masked = _as_tensor(mask_sample, torch.bool)
    empty = _as_tensor(is_empty_rollout, torch.bool)
    idx = _as_tensor(trace_in_rollout_idx, torch.int64).tolist()
    kinds = [str(k) if k else "unknown" for k in list(trace_kinds)[:n]]
    kinds += ["unknown"] * (n - len(kinds))

    tok_mask = token_mask[:n, 1:].detach().to("cpu").float()
    smask = _as_tensor(sample_mask, torch.float32)
    valid = tok_mask * (smask > 0).float().unsqueeze(-1)  # [n, S-1], what the loss trains on
    gen_tokens = tok_mask.sum(dim=-1)  # [n] generated tokens, independent of the gate
    has_gen_tokens = gen_tokens > 0
    # A trace's seq error is only meaningful if it had generated tokens AND was
    # going to train before the gate (seq_mult_prob_error is 0 otherwise).
    gate_eligible = pre_gate_unmasked & has_gen_tokens

    # Per-token |logp_gen - logp_train| and its per-trace mean / max over the
    # GENERATED tokens. torch.where (not a multiply) so a NaN/inf at a
    # non-generated position cannot poison a trace's statistic; a NaN on a
    # generated token is dropped by the finite reductions below.
    lp_err = (
        (generation_logprobs[:n, 1:] - prev_logprobs[:n, 1:])
        .detach()
        .to("cpu")
        .float()
        .abs()
    )
    tok_bool = tok_mask > 0
    trace_mean_abs_err = torch.where(tok_bool, lp_err, torch.zeros_like(lp_err)).sum(
        dim=-1
    ) / gen_tokens.clamp(min=1.0)
    # -inf for traces without generated tokens; those are never gate-eligible.
    if lp_err.shape[-1] > 0:
        trace_max_abs_err = torch.where(
            tok_bool, lp_err, torch.full_like(lp_err, float("-inf"))
        ).amax(dim=-1)
    else:
        trace_max_abs_err = torch.full((n,), float("-inf"))
    alt_gate_log_threshold: float | None = None
    if seq_logprob_error_threshold is not None and float(seq_logprob_error_threshold) > 1.0:
        alt_gate_log_threshold = math.log(float(seq_logprob_error_threshold))

    metrics: dict[str, float] = {
        "multi_trace/env_masked_trace_fraction": float(env_masked.float().mean().item()),
        "multi_trace/empty_rollout_trace_fraction": float(empty.float().mean().item()),
        "multi_trace/seq_logprob_masked_trace_fraction": float(
            seq_masked.float().mean().item()
        ),
    }

    by_idx: dict[str, list[int]] = {}
    by_kind: dict[str, list[int]] = {}
    for row in range(n):
        by_idx.setdefault(trace_index_bucket(idx[row]), []).append(row)
        by_kind.setdefault(kinds[row], []).append(row)

    def _emit_seq_error_buckets(prefix: str, groups: dict[str, list[int]]) -> None:
        for name, rows in groups.items():
            rows_t = torch.tensor(rows, dtype=torch.int64)
            eligible = rows_t[gate_eligible[rows_t]]
            if eligible.numel() == 0:
                continue
            mean_err = _finite_mean(seq_err[eligible])
            if mean_err is not None:
                metrics[
                    f"logprob_error/seq_mult_prob_error/{prefix}/{name}/mean"
                ] = mean_err
            metrics[f"logprob_error/seq_masked_fraction/{prefix}/{name}"] = float(
                seq_masked[eligible].float().mean().item()
            )
            # Length-robust counterparts (see docstring).
            mean_abs = _finite_mean(trace_mean_abs_err[eligible])
            if mean_abs is not None:
                metrics[f"logprob_error/seq_mean_abs_err/{prefix}/{name}/mean"] = (
                    mean_abs
                )
            max_abs_mean = _finite_mean(trace_max_abs_err[eligible])
            if max_abs_mean is not None:
                metrics[f"logprob_error/seq_max_abs_err/{prefix}/{name}/mean"] = (
                    max_abs_mean
                )
            max_abs_max = _finite_max(trace_max_abs_err[eligible])
            if max_abs_max is not None:
                metrics[f"logprob_error/seq_max_abs_err/{prefix}/{name}/max"] = (
                    max_abs_max
                )
            metrics[f"logprob_error/seq_gen_tokens/{prefix}/{name}/mean"] = float(
                gen_tokens[eligible].mean().item()
            )
            if alt_gate_log_threshold is not None:
                # Same keep-condition shape as the live gate (`err <= thr`
                # keeps), so a NaN statistic counts as masked in both.
                alt_masked = ~(trace_mean_abs_err[eligible] <= alt_gate_log_threshold)
                metrics[f"logprob_error/alt_gate_masked_fraction/{prefix}/{name}"] = (
                    float(alt_masked.float().mean().item())
                )

    _emit_seq_error_buckets("by_trace_in_rollout_idx", by_idx)
    _emit_seq_error_buckets("by_segment_kind", by_kind)

    # Trainable tokens = what the loss trains on after the gate, by segment
    # kind and rolled up by compaction status (subagent/empty/unknown are in
    # neither roll-up but count towards the total).
    total_valid = float(valid.sum().item())
    metrics["multi_trace/trainable_tokens/total"] = total_valid
    compacted_tokens = 0.0
    uncompacted_tokens = 0.0
    for kind, rows in by_kind.items():
        rows_t = torch.tensor(rows, dtype=torch.int64)
        kind_tokens = float(valid[rows_t].sum().item())
        metrics[f"multi_trace/trace_kind_count/{kind}"] = float(len(rows))
        metrics[f"multi_trace/trainable_tokens/by_segment_kind/{kind}"] = kind_tokens
        if total_valid > 0:
            metrics[f"multi_trace/trace_kind_valid_token_share/{kind}"] = (
                kind_tokens / total_valid
            )
        if kind in COMPACTED_KINDS:
            compacted_tokens += kind_tokens
        elif kind == _UNCOMPACTED_KIND:
            uncompacted_tokens += kind_tokens
    metrics["multi_trace/trainable_tokens/compacted_rollouts"] = compacted_tokens
    metrics["multi_trace/trainable_tokens/uncompacted_rollouts"] = uncompacted_tokens
    if total_valid > 0:
        metrics["multi_trace/trainable_tokens/compacted_fraction"] = (
            compacted_tokens / total_valid
        )

    # Per-rollout view (needs the positional rollout ids): a rollout is
    # compacted iff any of its traces has a kind in COMPACTED_KINDS.
    if trace_rollout_ids is not None:
        rollout_ids = _as_tensor(trace_rollout_ids, torch.int64).tolist()
        if len(rollout_ids) == n:
            valid_per_trace = valid.sum(dim=-1)  # [n]
            rollout_rows: dict[int, list[int]] = {}
            for row, rollout_id in enumerate(rollout_ids):
                rollout_rows.setdefault(rollout_id, []).append(row)
            tokens_per_rollout: dict[str, list[float]] = {
                "compacted": [],
                "uncompacted": [],
            }
            traces_per_rollout: dict[str, list[float]] = {
                "compacted": [],
                "uncompacted": [],
            }
            for rows in rollout_rows.values():
                cls = (
                    "compacted"
                    if any(kinds[row] in COMPACTED_KINDS for row in rows)
                    else "uncompacted"
                )
                rows_t = torch.tensor(rows, dtype=torch.int64)
                tokens_per_rollout[cls].append(
                    float(valid_per_trace[rows_t].sum().item())
                )
                traces_per_rollout[cls].append(float(len(rows)))
            for cls in ("compacted", "uncompacted"):
                count = len(traces_per_rollout[cls])
                metrics[f"multi_trace/rollouts/{cls}_count"] = float(count)
                if count > 0:
                    metrics[f"multi_trace/trainable_tokens_per_rollout/{cls}/mean"] = (
                        sum(tokens_per_rollout[cls]) / count
                    )
                    metrics[f"multi_trace/traces_per_rollout/{cls}/mean"] = (
                        sum(traces_per_rollout[cls]) / count
                    )

    # |logprob error| by relative position of the token within its trace's
    # valid tokens: position k/N for the k-th valid token -> quartile
    # ceil(4k/N) in {1..4}. Monotone-decreasing error vs position points at
    # within-trace staleness (weights refit mid-rollout); flat points at
    # numerics.
    if total_valid > 0:
        valid_bool = valid > 0
        counts = valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
        rel_pos = valid.cumsum(dim=-1) / counts
        quartile = torch.ceil(rel_pos * 4).clamp(min=1, max=4).to(torch.int64)
        for q in (1, 2, 3, 4):
            sel = valid_bool & (quartile == q)
            if not bool(sel.any()):
                continue
            mean_err = _finite_mean(lp_err[sel])
            if mean_err is not None:
                metrics[f"logprob_error/token_abs_err/by_position_quartile/q{q}"] = (
                    mean_err
                )

    return metrics


def finalize_sum_count_metrics(aggregated: dict) -> dict:
    """Derive exact means/rates from cross-group-summed ``X/sum`` / ``X/count`` keys.

    Input is the rollout-metrics dict AFTER cross-group aggregation (every
    ``*/sum`` and ``*/count`` key already summed over prompt groups). For each
    ``X/sum``: drop it and, when ``X/count > 0``, emit ``X/mean = sum / count``
    (the ``/count`` key is kept). For rollout-level event counters
    (``termination/*/count``, ``mask_sample/by_kind/*/count``,
    ``format/think_tag_violation/count``, ``reward/masked_resolved/count``) also
    emit ``X/rate = count / rollouts/count`` when ``rollouts/count > 0``.

    Keys not matching those patterns pass through untouched. Never emits NaN.
    """
    out = dict(aggregated)
    for key, value in aggregated.items():
        if not key.endswith("/sum"):
            continue
        base = key[: -len("/sum")]
        count = aggregated.get(f"{base}/count", 0)
        out.pop(key, None)
        if isinstance(count, (int, float)) and count > 0 and isinstance(
            value, (int, float)
        ):
            out[f"{base}/mean"] = value / count

    num_rollouts = aggregated.get(ROLLOUT_COUNT_KEY, 0)
    if isinstance(num_rollouts, (int, float)) and num_rollouts > 0:
        for key, value in aggregated.items():
            if not key.endswith("/count") or not isinstance(value, (int, float)):
                continue
            if key in _RATE_COUNT_KEYS or key.startswith(_RATE_COUNT_PREFIXES):
                out[key[: -len("/count")] + "/rate"] = value / num_rollouts
    return out
