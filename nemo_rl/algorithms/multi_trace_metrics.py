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
  causes, per-kind token share, and |logprob error| by relative position.
* :func:`finalize_sum_count_metrics` — turns the exactly-summed ``X/sum`` and
  ``X/count`` rollout metrics (emitted per prompt group by
  ``rollouts._compaction_rollout_metrics`` and friends) into ``X/mean`` and,
  for rollout-level event counts, ``X/rate``.

No key is ever emitted with a NaN/inf value; buckets without data are omitted.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

# Rollout-level event counters that get a `/rate` (= count / rollouts/count)
# in addition to the raw count. Keys are matched on prefix or exactly.
_RATE_COUNT_PREFIXES: tuple[str, ...] = ("termination/", "mask_sample/by_kind/")
_RATE_COUNT_KEYS: frozenset[str] = frozenset(
    {"format/think_tag_violation/count", "reward/masked_resolved/count"}
)
ROLLOUT_COUNT_KEY = "rollouts/count"


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
    has_gen_tokens = tok_mask.sum(dim=-1) > 0
    # A trace's seq error is only meaningful if it had generated tokens AND was
    # going to train before the gate (seq_mult_prob_error is 0 otherwise).
    gate_eligible = pre_gate_unmasked & has_gen_tokens

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

    _emit_seq_error_buckets("by_trace_in_rollout_idx", by_idx)
    _emit_seq_error_buckets("by_segment_kind", by_kind)

    total_valid = float(valid.sum().item())
    for kind, rows in by_kind.items():
        metrics[f"multi_trace/trace_kind_count/{kind}"] = float(len(rows))
        if total_valid > 0:
            rows_t = torch.tensor(rows, dtype=torch.int64)
            metrics[f"multi_trace/trace_kind_valid_token_share/{kind}"] = (
                float(valid[rows_t].sum().item()) / total_valid
            )

    # |logprob error| by relative position of the token within its trace's
    # valid tokens: position k/N for the k-th valid token -> quartile
    # ceil(4k/N) in {1..4}. Monotone-decreasing error vs position points at
    # within-trace staleness (weights refit mid-rollout); flat points at
    # numerics.
    if total_valid > 0:
        lp_err = (
            (generation_logprobs[:n, 1:] - prev_logprobs[:n, 1:])
            .detach()
            .to("cpu")
            .float()
            .abs()
        )
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
