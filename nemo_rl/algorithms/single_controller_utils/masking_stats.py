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
"""Per-step breakdown of why rows were masked from the loss (``train/masking/*``).

Accumulated once per advantage-stage chunk from the masks that stage already
holds, reduced once per step. Counts are rows (samples); ``*_frac`` values are
over the rows the finalizer delivered with tokens (``sample_mask > 0``), so
placeholder rows are reported but do not dilute the fractions.

* ``masking/rows``: rows in the step's batches (incl. placeholders).
* ``masking/placeholder_rows``: finalizer placeholders (rejected captures,
  ``sample_mask`` 0); never trained, never in the baseline.
* ``masking/env_flag_rows`` / ``_frac``: rows flagged ``mask_sample`` by the
  environment or by ``env.mask_sample_rules``.
* ``masking/truncated_rows`` / ``_frac``: rows whose sequence hit the length
  limit, whether or not ``grpo.overlong_filtering`` masks them;
  ``masking/truncated_masked_rows``: those actually masked for it (and not
  already env-flagged).
* ``masking/seq_logprob_error_rows``: rows the sequence-logprob-error check
  dropped after the other masks.
* ``masking/masked_rows`` / ``masked_frac``: rows with tokens that ended up
  with no gradient, for any reason; ``masking/trained_rows`` / ``trained_frac``:
  rows that train.
* ``masking/baseline_rows``: rows whose reward enters the group baseline/std;
  ``masking/reinstated_rows``: incomplete rows kept in the baseline by
  ``grpo.masked_sample_rewards_in_baseline``.
* ``masking/reward_mean_{trained,env_flag,truncated,masked}``: mean reward of
  each population, omitted when the population is empty, so the effect of
  masking on the reward signal is visible.
* ``masking/groups``, ``masking/groups_lt2_trained_frac``: GRPO groups in the
  step and the fraction left with fewer than two trained rows (no gradient).
"""

from __future__ import annotations

import torch

MASKING_STATS_KEYS: tuple[str, ...] = (
    "group_ids",
    "rewards",
    "sample_mask",
    "env_flag",
    "truncated",
    "truncated_masked",
    "final_mask",
    "baseline_mask",
)


def new_masking_stats_accumulator() -> dict[str, list[torch.Tensor]]:
    return {key: [] for key in MASKING_STATS_KEYS}


def accumulate_masking_stats(
    acc: dict[str, list[torch.Tensor]],
    *,
    prompt_ids: torch.Tensor,
    rewards: torch.Tensor,
    sample_mask: torch.Tensor,
    mask_sample: torch.Tensor,
    truncated: torch.Tensor,
    overlong_filtering: bool,
    final_sample_mask: torch.Tensor,
    baseline_mask: torch.Tensor,
) -> None:
    """Append one chunk (all tensors moved to CPU, rows keyed by prompt group)."""
    batch = rewards.shape[0]
    if batch == 0:
        return
    _, local_groups = torch.unique(
        prompt_ids.detach().reshape(batch, -1), dim=0, return_inverse=True
    )
    offset = int(acc["group_ids"][-1].max()) + 1 if acc["group_ids"] else 0
    env_flag = mask_sample.detach().bool().reshape(batch)
    trunc = truncated.detach().bool().reshape(batch)
    acc["group_ids"].append(local_groups.reshape(batch).long().cpu() + offset)
    acc["rewards"].append(rewards.detach().float().reshape(batch).cpu())
    acc["sample_mask"].append(sample_mask.detach().float().reshape(batch).cpu())
    acc["env_flag"].append(env_flag.cpu())
    acc["truncated"].append(trunc.cpu())
    acc["truncated_masked"].append(
        (trunc & ~env_flag).cpu()
        if overlong_filtering
        else torch.zeros(batch, dtype=torch.bool)
    )
    acc["final_mask"].append(final_sample_mask.detach().float().reshape(batch).cpu())
    acc["baseline_mask"].append(baseline_mask.detach().float().reshape(batch).cpu())


def _set_mean(out: dict[str, float], key: str, values: torch.Tensor) -> None:
    if values.numel():
        out[key] = float(values.float().mean())


def reduce_masking_stats(acc: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    if not acc.get("rewards"):
        return {}
    rewards = torch.cat(acc["rewards"])
    has_tokens = torch.cat(acc["sample_mask"]) > 0
    env_flag = torch.cat(acc["env_flag"]) & has_tokens
    truncated = torch.cat(acc["truncated"]) & has_tokens
    truncated_masked = torch.cat(acc["truncated_masked"]) & has_tokens
    trained = torch.cat(acc["final_mask"]) > 0
    in_baseline = torch.cat(acc["baseline_mask"]) > 0
    masked = has_tokens & ~trained
    # Dropped by the seq-logprob check = masked but neither env-flagged nor truncation-masked.
    seq_logprob_dropped = masked & ~env_flag & ~truncated_masked
    reinstated = in_baseline & ~trained

    rows = int(rewards.numel())
    with_tokens = int(has_tokens.sum())
    frac = lambda n: (n / with_tokens) if with_tokens else 0.0  # noqa: E731

    out: dict[str, float] = {
        "masking/rows": float(rows),
        "masking/placeholder_rows": float(rows - with_tokens),
        "masking/env_flag_rows": float(env_flag.sum()),
        "masking/env_flag_frac": frac(int(env_flag.sum())),
        "masking/truncated_rows": float(truncated.sum()),
        "masking/truncated_frac": frac(int(truncated.sum())),
        "masking/truncated_masked_rows": float(truncated_masked.sum()),
        "masking/seq_logprob_error_rows": float(seq_logprob_dropped.sum()),
        "masking/masked_rows": float(masked.sum()),
        "masking/masked_frac": frac(int(masked.sum())),
        "masking/trained_rows": float(trained.sum()),
        "masking/trained_frac": frac(int(trained.sum())),
        "masking/baseline_rows": float(in_baseline.sum()),
        "masking/reinstated_rows": float(reinstated.sum()),
    }
    _set_mean(out, "masking/reward_mean_trained", rewards[trained])
    _set_mean(out, "masking/reward_mean_env_flag", rewards[env_flag])
    _set_mean(out, "masking/reward_mean_truncated", rewards[truncated])
    _set_mean(out, "masking/reward_mean_masked", rewards[masked])

    group_ids = torch.cat(acc["group_ids"])
    num_groups = int(group_ids.max()) + 1
    trained_per_group = torch.bincount(group_ids[trained], minlength=num_groups)
    out["masking/groups"] = float(num_groups)
    out["masking/groups_lt2_trained_frac"] = (
        float((trained_per_group < 2).sum()) / num_groups if num_groups else 0.0
    )
    return out
