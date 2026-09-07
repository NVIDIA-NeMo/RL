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

"""Wall-clock time-efficiency reward for NeMo-Gym agentic rollouts.

Charges each rollout for the time its agent loop ran::

    reward_i = reward_i - lambda_time * (openhands_run_time_i / 60)

``openhands_run_time`` (seconds) is emitted by the Gym ``swe_agents`` server
for every rollout. It spans the agent container from launch to exit, so it
excludes final evaluation and Ray queueing but *includes* apptainer spin-up,
which the policy cannot influence. With the default ``lambda_time = 1/60`` a
60-minute rollout costs exactly 1.0: one hour of compute is priced at one
solved task.

Because the deduction is a small continuous term added to a binary reward, it
should be paired with a tight ``grpo.advantage_clip_low/high`` (e.g. -3/3) when
``grpo.normalize_rewards`` is on; otherwise a group whose rewards differ only
by wall time is normalized by a tiny spread and its advantages explode.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class TimeEfficiencyConfig(BaseModel, extra="allow"):
    """User-facing ``grpo.time_efficiency`` block.

    Attributes:
        enabled: Master switch. When ``False`` rewards are untouched and no
            metrics are emitted.
        lambda_time: Deduction per minute of agent wall time. ``1/60`` makes a
            60-minute rollout cost exactly 1.0.
        apply_to: ``"all"`` deducts from every rollout, including failures, so
            failed rollouts also receive a gradient from their wall time; a
            slow correct rollout can then score below a fast incorrect one.
            ``"correct"`` deducts only from rollouts that are resolved and
            still carry a positive reward after the reward-zeroing penalties.
        floor: Lower clamp on the post-deduction reward so one pathologically
            slow rollout cannot dominate its group. ``None`` disables.
    """

    enabled: bool = False
    lambda_time: float = 1.0 / 60.0
    apply_to: Literal["all", "correct"] = "all"
    floor: float | None = None


def rollout_minutes(result: dict[str, Any]) -> float:
    """Return the agent-loop wall time of one rollout in minutes.

    A missing, ``None``, negative or non-numeric ``openhands_run_time`` counts
    as 0 so an absent timing never silently penalizes a rollout.
    """
    seconds = result["full_result"].get("openhands_run_time")
    try:
        return max(float(seconds), 0.0) / 60.0
    except (TypeError, ValueError):
        return 0.0


def apply_time_efficiency_reward(
    results: list[dict[str, Any]],
    config: TimeEfficiencyConfig | None,
) -> dict[str, float]:
    """Deduct the wall-time price from each ``result["full_result"]["reward"]`` in place.

    Runs last in the NeMo-Gym postprocess, after effort shaping, the
    reward-zeroing penalties and the length penalties: those assume a binary
    env reward, so the continuous deduction only composes with them when it is
    applied after them.

    Args:
        results: The NeMo-Gym rollout results to adjust: one prompt group on
            the async collector path, the whole batch in
            ``run_nemo_gym_rollout_sync``.
        config: The ``grpo.time_efficiency`` block. ``None`` or
            ``enabled=False`` leaves ``results`` untouched.

    Returns:
        Metrics under the ``time_efficiency/`` prefix computed over ``results``,
        or an empty dict when the feature is disabled or ``results`` is empty.
    """
    if config is None or not config.enabled or not results:
        return {}

    minutes = [rollout_minutes(result) for result in results]
    deductions: list[float] = []
    for result, rollout_min in zip(results, minutes):
        full_result = result["full_result"]
        if config.apply_to == "correct" and not (
            full_result.get("resolved") and full_result.get("reward")
        ):
            deductions.append(0.0)
            continue
        base = float(full_result.get("reward") or 0.0)
        new = base - config.lambda_time * rollout_min
        if config.floor is not None:
            new = max(new, config.floor)
        deductions.append(base - new)
        full_result["reward"] = new

    # ``<name>/<stat>`` keys so aggregate_rollout_metrics maxes the ``/max``
    # entries across prompt groups instead of averaging them.
    return {
        "time_efficiency/minutes/mean": sum(minutes) / len(minutes),
        "time_efficiency/minutes/max": max(minutes),
        "time_efficiency/deduction/mean": sum(deductions) / len(deductions),
        "time_efficiency/deduction/max": max(deductions),
        # 1.0 when the deduction differs within the group (skipped rollouts
        # count as 0), i.e. the term can still produce a gradient after group
        # normalization. 1e-6 absorbs float noise from ``base - new``.
        "time_efficiency/group_has_signal": float(
            max(deductions) - min(deductions) > 1e-6
        ),
    }
