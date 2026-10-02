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
"""Per-harness and per-agent rollout metrics on the token-capture path.

Every rollout of a NeMo-Gym prompt group is produced by one Gym agent entry
(``agent_ref.name``), so a group can be labelled with that agent and with its
harness: the agent implementation Gym's resolved config runs for the entry
(``responses_api_agents.<implementation>``, e.g. ``claude_code_sandboxed_agent``),
shared by every entry built on it. On the token-capture path the per-group
``rollout_metrics`` that the batch path logs per agent never reach the
controller's logger, so the finalizer adds each labelled group's raw counts and
sums under :data:`GROUP_STATS_PREFIX` to its group metrics instead. The
controller sums those over the groups a training step consumes and
:func:`reduce_group_label_stats` turns them into rates and means, so a step's
per-label metrics cover exactly the rows the step trains on, like ``reward``.

Metrics, per label scope (``harness`` or ``agent``) and label name ``L``:

* ``{scope}/L/groups`` and ``rollouts``: prompt groups and rollouts in the step.
* ``reward`` and ``pass_frac``: mean reward and the fraction with reward >= 0.5
  over trained rollouts (verified by the finalizer and not masked by a
  ``mask_sample`` rule), matching the step's ``reward`` and
  ``reward/pass_frac``.
* ``reward_all_rollouts``: mean reward over every rollout, including the ones
  capture could not verify or a rule masked: the label's task success rate.
* ``trained_frac``, ``placeholder_frac``, ``mask_sample_frac``: shares of the
  rollouts that train, that capture verification turned into placeholders, and
  that a ``mask_sample`` rule masked (they add up to 1).
* ``gen_tokens_mean`` (with ``_pass`` / ``_fail``), ``turns_mean``,
  ``seq_len_mean``, ``truncated_frac``, ``calls_per_rollout``: over trained
  rollouts. Generated tokens and turns use the definitions of
  ``rollout_stats`` (``token_mask`` ones, and runs of them). Calls count every
  captured model call of the rollout, sub-agent calls included.
* ``mixed_group_frac``, ``all_pass_group_frac``, ``all_fail_group_frac``: over
  groups with at least two trained rollouts, as ``groups/*``.
* ``result/<field>``: mean of each numeric field of the Gym result (for
  example ``harness_finished``) over the rollouts that reported it, for the
  scopes the caller asks for (the harness scope by default).
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from typing import Any

GROUP_STATS_PREFIX = "group_stats/"
HARNESS_SCOPE = "harness"
AGENT_SCOPE = "agent"
# The threshold of rollout_stats' reward/pass_frac.
PASS_THRESHOLD = 0.5
# Same tolerance as rollout_stats uses for a group with reward spread.
_SPREAD_EPS = 1e-6
_RESULT_SUM = "result_sum/"
_RESULT_COUNT = "result_count/"


def label_segment(value: str) -> str:
    """Make a label usable as one metric-key segment ("/" separates segments)."""
    return str(value).replace("/", "_")


def numeric_result_fields(result: Mapping[str, Any]) -> dict[str, float]:
    """Top-level numeric (and boolean) fields of one Gym result, as floats."""
    fields: dict[str, float] = {}
    for key, value in result.items():
        if isinstance(value, bool):
            fields[str(key)] = float(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            fields[str(key)] = float(value)
    return fields


def generated_runs(token_mask: Sequence[float]) -> tuple[int, int]:
    """Generated-token count and turn count (runs of generated tokens) of one row."""
    tokens = 0
    turns = 0
    previous = False
    for value in token_mask:
        current = bool(value)
        tokens += current
        if current and not previous:
            turns += 1
        previous = current
    return tokens, turns


def group_label_stats(
    labels: Sequence[tuple[str, str]],
    *,
    rewards: Sequence[float],
    valid: Sequence[bool],
    mask_sample: Sequence[bool],
    gen_tokens: Sequence[float],
    turns: Sequence[float],
    seq_lens: Sequence[float],
    truncated: Sequence[bool],
    calls: Sequence[float],
    result_stats: Sequence[tuple[str, float, int]] = (),
    result_scopes: Collection[str] = (HARNESS_SCOPE,),
) -> dict[str, float]:
    """Raw counts and sums of one prompt group, under each ``(scope, name)`` label.

    All per-rollout sequences are parallel. ``result_stats`` holds
    ``(field, sum, count)`` of the group's numeric Gym result fields and is
    reported for the labels whose scope is in ``result_scopes``. Returns ``{}``
    for an unlabelled or empty group.
    """
    count = len(rewards)
    if not labels or count == 0:
        return {}
    for name, values in (
        ("valid", valid),
        ("mask_sample", mask_sample),
        ("gen_tokens", gen_tokens),
        ("turns", turns),
        ("seq_lens", seq_lens),
        ("truncated", truncated),
        ("calls", calls),
    ):
        if len(values) != count:
            raise ValueError(f"{name} has {len(values)} entries for {count} rollouts")

    trained = [bool(v) and not bool(m) for v, m in zip(valid, mask_sample)]
    trained_rewards = [float(r) for r, t in zip(rewards, trained) if t]
    passed = [reward >= PASS_THRESHOLD for reward in trained_rewards]

    def trained_sum(values: Sequence[float], *, outcome: bool | None = None) -> float:
        selected = [float(v) for v, t in zip(values, trained) if t]
        if outcome is not None:
            selected = [v for v, p in zip(selected, passed) if p == outcome]
        return math.fsum(selected)

    stats: dict[str, float] = {
        "groups": 1.0,
        "rollouts": float(count),
        "trained": float(len(trained_rewards)),
        "placeholders": float(sum(1 for v in valid if not v)),
        "mask_sample": float(sum(1 for v, m in zip(valid, mask_sample) if v and m)),
        "all_reward_sum": math.fsum(float(r) for r in rewards),
        "reward_sum": math.fsum(trained_rewards),
        "pass_count": float(sum(passed)),
        "gen_tokens_sum": trained_sum(gen_tokens),
        "gen_tokens_pass_sum": trained_sum(gen_tokens, outcome=True),
        "gen_tokens_fail_sum": trained_sum(gen_tokens, outcome=False),
        "turns_sum": trained_sum(turns),
        "seq_len_sum": trained_sum(seq_lens),
        "truncated": trained_sum([float(bool(t)) for t in truncated]),
        "calls_sum": trained_sum(calls),
        "multi": 0.0,
        "mixed": 0.0,
        "all_pass": 0.0,
        "all_fail": 0.0,
    }
    if len(trained_rewards) >= 2:
        mean = math.fsum(trained_rewards) / len(trained_rewards)
        variance = math.fsum((r - mean) ** 2 for r in trained_rewards) / len(
            trained_rewards
        )
        stats["multi"] = 1.0
        stats["mixed"] = float(math.sqrt(variance) > _SPREAD_EPS)
        stats["all_pass"] = float(all(passed))
        stats["all_fail"] = float(not any(passed))

    out: dict[str, float] = {}
    for scope, name in labels:
        prefix = f"{GROUP_STATS_PREFIX}{label_segment(scope)}/{label_segment(name)}/"
        for stat, value in stats.items():
            out[prefix + stat] = value
        if scope in result_scopes:
            for field, total, field_count in result_stats:
                out[f"{prefix}{_RESULT_SUM}{field}"] = float(total)
                out[f"{prefix}{_RESULT_COUNT}{field}"] = float(field_count)
    return out


def reduce_group_label_stats(sums: Mapping[str, float]) -> dict[str, float]:
    """Turn group stats summed over a step's groups into per-label metrics.

    Keys outside :data:`GROUP_STATS_PREFIX` are ignored. A rate is omitted
    (not logged as 0) when its denominator is 0, e.g. ``reward`` for a label
    none of whose rollouts trained in the step.
    """
    by_label: dict[tuple[str, str], dict[str, float]] = {}
    for key, value in sums.items():
        if not key.startswith(GROUP_STATS_PREFIX):
            continue
        parts = key[len(GROUP_STATS_PREFIX) :].split("/", 2)
        if len(parts) != 3:
            continue
        scope, name, stat = parts
        label = by_label.setdefault((scope, name), {})
        label[stat] = label.get(stat, 0.0) + float(value)

    out: dict[str, float] = {}
    for (scope, name), stats in sorted(by_label.items()):
        prefix = f"{scope}/{name}/"
        rollouts = stats.get("rollouts", 0.0)
        trained = stats.get("trained", 0.0)
        passed = stats.get("pass_count", 0.0)
        multi = stats.get("multi", 0.0)
        out[prefix + "groups"] = stats.get("groups", 0.0)
        out[prefix + "rollouts"] = rollouts
        if rollouts > 0:
            out[prefix + "reward_all_rollouts"] = (
                stats.get("all_reward_sum", 0.0) / rollouts
            )
            out[prefix + "trained_frac"] = trained / rollouts
            out[prefix + "placeholder_frac"] = stats.get("placeholders", 0.0) / rollouts
            out[prefix + "mask_sample_frac"] = stats.get("mask_sample", 0.0) / rollouts
        if trained > 0:
            out[prefix + "reward"] = stats.get("reward_sum", 0.0) / trained
            out[prefix + "pass_frac"] = passed / trained
            out[prefix + "gen_tokens_mean"] = stats.get("gen_tokens_sum", 0.0) / trained
            out[prefix + "turns_mean"] = stats.get("turns_sum", 0.0) / trained
            out[prefix + "seq_len_mean"] = stats.get("seq_len_sum", 0.0) / trained
            out[prefix + "truncated_frac"] = stats.get("truncated", 0.0) / trained
            out[prefix + "calls_per_rollout"] = stats.get("calls_sum", 0.0) / trained
        if passed > 0:
            out[prefix + "gen_tokens_mean_pass"] = (
                stats.get("gen_tokens_pass_sum", 0.0) / passed
            )
        if trained - passed > 0:
            out[prefix + "gen_tokens_mean_fail"] = stats.get(
                "gen_tokens_fail_sum", 0.0
            ) / (trained - passed)
        if multi > 0:
            out[prefix + "mixed_group_frac"] = stats.get("mixed", 0.0) / multi
            out[prefix + "all_pass_group_frac"] = stats.get("all_pass", 0.0) / multi
            out[prefix + "all_fail_group_frac"] = stats.get("all_fail", 0.0) / multi
        for stat, total in stats.items():
            if not stat.startswith(_RESULT_SUM):
                continue
            field = stat[len(_RESULT_SUM) :]
            field_count = stats.get(_RESULT_COUNT + field, 0.0)
            if field_count > 0:
                out[f"{prefix}result/{field}"] = total / field_count
    return out
