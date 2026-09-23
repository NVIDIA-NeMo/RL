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
"""Per-step rollout distribution metrics for the SingleController.

Everything here is derived from tensors the advantage stage already holds for
each streaming chunk (``token_mask``, ``sample_mask``, rewards, prompt ids,
the ``truncated`` flag and ``input_lengths``), reduced once per training step.
The inputs are a few hundred rows, so the cost is a handful of CPU ops.

Metrics (logged under the ``train/`` prefix):

* ``gen_tokens/{mean,p50,p75,p90,max}``: tokens the policy generated per
  sample, i.e. the count of ``token_mask`` ones (assistant tokens; prompt and
  tool output are 0). ``gen_tokens/mean_pass`` / ``mean_fail`` split by
  outcome (reward >= 0.5) to expose length drift by outcome.
* ``turns/{mean,p50,p75,p90,max}``: assistant turns per sample, counted as
  contiguous runs of ``token_mask`` ones (each generated message is a run,
  separated by prompt / tool-output tokens). ``turns/mean_pass`` / ``mean_fail``.
* ``seq_len/{mean,p50,p90,max}``: full sequence length (prompt + all turns +
  tool output) and ``seq_len/frac_above_90pct_ctx``: fraction of samples using
  more than 90 % of ``max_total_sequence_length`` (context pressure).
* ``groups/count``: GRPO groups with >= 2 valid samples;
  ``groups/mixed_frac`` / ``mixed_count``: groups whose rewards are not all
  equal (for 0/1 rewards: neither all-pass nor all-fail), the only groups that
  produce non-zero advantages; ``groups/all_pass_frac``, ``groups/all_fail_frac``
  (every valid reward >= 0.5, resp. < 0.5); ``groups/reward_std_mean``: mean
  within-group reward std; ``groups/zero_advantage_sample_frac``: valid samples
  sitting in a non-mixed group (they contribute no gradient).
* ``reward/std`` over valid samples and ``reward/pass_frac`` (reward >= 0.5).
* ``truncated_frac``: valid samples the generation cut at the length limit.

Rows with ``sample_mask == 0`` (token-capture placeholders, environment
mask_sample, overlong filter, sequence-logprob-error masking) are excluded
from every statistic, matching what the advantage estimator trains on.
"""

from __future__ import annotations

from typing import Optional

import torch

ROLLOUT_STATS_KEYS: tuple[str, ...] = (
    "prompt_ids",
    "rewards",
    "sample_masks",
    "gen_tokens",
    "turns",
    "truncated",
    "seq_lens",
)

_PERCENTILES = (0.5, 0.75, 0.9)
_PASS_THRESHOLD = 0.5


def new_rollout_stats_accumulator() -> dict[str, list[torch.Tensor]]:
    """Fresh per-step accumulator: one CPU tensor per advantage-stage chunk."""
    return {key: [] for key in ROLLOUT_STATS_KEYS}


def per_sample_rollout_stats(token_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-row generated-token and assistant-turn counts from the training mask.

    Args:
        token_mask: ``(B, S)`` mask, non-zero on the tokens the policy generated.

    Returns:
        ``gen_tokens`` and ``turns`` as float CPU tensors of shape ``(B,)``. A
        turn is a maximal run of consecutive generated tokens; a run that
        starts at position 0 counts as well.
    """
    mask = token_mask.detach().bool()
    if mask.ndim != 2:
        mask = mask.reshape(mask.shape[0], -1)
    gen_tokens = mask.sum(dim=-1)
    if mask.shape[1] == 0:
        turns = torch.zeros(mask.shape[0], dtype=torch.long, device=mask.device)
    else:
        # A run starts where the mask is on and the previous position is off;
        # position 0 starts a run if it is on.
        turns = (mask[:, 1:] & ~mask[:, :-1]).sum(dim=-1) + mask[:, 0].long()
    return {
        "gen_tokens": gen_tokens.float().cpu(),
        "turns": turns.float().cpu(),
    }


def accumulate_rollout_stats(
    acc: dict[str, list[torch.Tensor]],
    *,
    prompt_ids: torch.Tensor,
    rewards: torch.Tensor,
    sample_mask: torch.Tensor,
    token_mask: torch.Tensor,
    truncated: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
) -> None:
    """Append one advantage-stage chunk to ``acc`` (all tensors moved to CPU).

    ``prompt_ids`` may be ``(B,)`` group ids or ``(B, P)`` prompt token ids; the
    reducer groups rows by identical values along the trailing dimensions.
    ``truncated`` / ``seq_lens`` are optional; the reducer skips the metrics
    that depend on them when any chunk lacks them.
    """
    batch = rewards.shape[0]
    stats = per_sample_rollout_stats(token_mask)
    acc["prompt_ids"].append(prompt_ids.detach().reshape(batch, -1).cpu())
    acc["rewards"].append(rewards.detach().float().reshape(batch).cpu())
    acc["sample_masks"].append(sample_mask.detach().float().reshape(batch).cpu())
    acc["gen_tokens"].append(stats["gen_tokens"])
    acc["turns"].append(stats["turns"])
    if truncated is not None:
        acc["truncated"].append(truncated.detach().float().reshape(batch).cpu())
    if seq_lens is not None:
        acc["seq_lens"].append(seq_lens.detach().float().reshape(batch).cpu())


def _distribution(out: dict[str, float], name: str, values: torch.Tensor) -> None:
    if values.numel() == 0:
        return
    values = values.float()
    quantiles = torch.quantile(values, torch.tensor(_PERCENTILES, dtype=values.dtype))
    out[f"{name}/mean"] = float(values.mean())
    out[f"{name}/p50"] = float(quantiles[0])
    out[f"{name}/p75"] = float(quantiles[1])
    out[f"{name}/p90"] = float(quantiles[2])
    out[f"{name}/max"] = float(values.max())


def _mean_or_zero(values: torch.Tensor) -> float:
    return float(values.float().mean()) if values.numel() else 0.0


def reduce_rollout_stats(
    acc: dict[str, list[torch.Tensor]],
    *,
    max_seq_len: Optional[int] = None,
) -> dict[str, float]:
    """Reduce a step's accumulated chunks into scalar metrics.

    Returns an empty dict when nothing valid was accumulated, so callers can
    ``update`` a metrics dict unconditionally.
    """
    if not acc.get("rewards"):
        return {}
    num_chunks = len(acc["rewards"])
    rewards = torch.cat(acc["rewards"])
    valid = torch.cat(acc["sample_masks"]) > 0
    if int(valid.sum()) == 0:
        return {}

    out: dict[str, float] = {}
    rewards = rewards[valid]
    passed = rewards >= _PASS_THRESHOLD
    gen_tokens = torch.cat(acc["gen_tokens"])[valid]
    turns = torch.cat(acc["turns"])[valid]

    _distribution(out, "gen_tokens", gen_tokens)
    out["gen_tokens/mean_pass"] = _mean_or_zero(gen_tokens[passed])
    out["gen_tokens/mean_fail"] = _mean_or_zero(gen_tokens[~passed])
    _distribution(out, "turns", turns)
    out["turns/mean_pass"] = _mean_or_zero(turns[passed])
    out["turns/mean_fail"] = _mean_or_zero(turns[~passed])

    if len(acc["seq_lens"]) == num_chunks:
        seq_lens = torch.cat(acc["seq_lens"])[valid]
        quantiles = torch.quantile(seq_lens, torch.tensor([0.5, 0.9]))
        out["seq_len/mean"] = float(seq_lens.mean())
        out["seq_len/p50"] = float(quantiles[0])
        out["seq_len/p90"] = float(quantiles[1])
        out["seq_len/max"] = float(seq_lens.max())
        if max_seq_len:
            out["seq_len/frac_above_90pct_ctx"] = float(
                (seq_lens > 0.9 * max_seq_len).float().mean()
            )
    if len(acc["truncated"]) == num_chunks:
        out["truncated_frac"] = float(torch.cat(acc["truncated"])[valid].mean())

    out["reward/std"] = (
        float(rewards.std(unbiased=False)) if rewards.numel() > 1 else 0.0
    )
    out["reward/pass_frac"] = float(passed.float().mean())

    # Group statistics: rows sharing a prompt id form one GRPO group.
    prompt_ids = torch.cat(acc["prompt_ids"])[valid]
    _, group_index = torch.unique(prompt_ids, dim=0, return_inverse=True)
    group_index = group_index.reshape(-1)
    num_groups = int(group_index.max()) + 1
    counts = torch.bincount(group_index, minlength=num_groups).float()
    sums = torch.bincount(group_index, weights=rewards, minlength=num_groups)
    sums_sq = torch.bincount(
        group_index, weights=rewards * rewards, minlength=num_groups
    )
    means = sums / counts
    variances = (sums_sq / counts - means * means).clamp_min(0.0)
    stds = variances.sqrt()
    group_min = torch.full((num_groups,), float("inf")).scatter_reduce(
        0, group_index, rewards, reduce="amin"
    )
    group_max = torch.full((num_groups,), float("-inf")).scatter_reduce(
        0, group_index, rewards, reduce="amax"
    )
    multi = counts >= 2
    mixed = multi & (stds > 1e-6)
    all_pass = multi & (group_min >= _PASS_THRESHOLD)
    all_fail = multi & (group_max < _PASS_THRESHOLD)
    num_multi = int(multi.sum())
    out["groups/count"] = float(num_multi)
    out["groups/mixed_count"] = float(mixed.sum())
    if num_multi:
        out["groups/mixed_frac"] = float(mixed.sum()) / num_multi
        out["groups/all_pass_frac"] = float(all_pass.sum()) / num_multi
        out["groups/all_fail_frac"] = float(all_fail.sum()) / num_multi
        out["groups/reward_std_mean"] = float(stds[multi].mean())
    else:
        out["groups/mixed_frac"] = 0.0
        out["groups/all_pass_frac"] = 0.0
        out["groups/all_fail_frac"] = 0.0
        out["groups/reward_std_mean"] = 0.0
    # A sample only carries a gradient if its group has reward spread.
    out["groups/zero_advantage_sample_frac"] = float(
        (~mixed[group_index]).float().mean()
    )
    return out
