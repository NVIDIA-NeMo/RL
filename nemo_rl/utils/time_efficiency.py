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

Charges each rollout for the time its agent loop ran, optionally paying a
saturating bonus for the tool calls it made::

    reward_i = reward_i
               + lambda_call_bonus * min(calls_i / call_bonus_ref, 1)   # 0 by default
               - lambda_time * (openhands_run_time_i / 60)

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

from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


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
            Rollouts the env flags with ``mask_sample`` (``swe_agents`` flags
            timeouts, i.e. the group's longest rollouts) are exempt while
            ``env.should_mask_flagged_samples`` is on: they are dropped from
            the loss, so charging them could only drag their siblings' group
            baseline down. They still enter the baseline at their raw 0.0,
            which tilts it slightly the other way. With the flag off those
            rows train and are charged.
            ``"correct"`` deducts only from rollouts that are resolved and
            still carry a positive reward after the reward-zeroing penalties.
        lambda_call_bonus: Bonus paid for tool calls, saturating at
            ``call_bonus_ref`` calls: ``lambda_call_bonus * min(calls /
            call_bonus_ref, 1)``. Only well-formed ``function_call`` items
            count, so a malformed call earns nothing. ``0`` (default) disables
            the bonus and reduces this block to the plain time deduction. The
            bonus follows the same ``apply_to`` gate as the deduction; paying
            it on failures (``"all"``) rewards junk calls, so use ``"correct"``
            with it.
        call_bonus_ref: Number of calls at which the bonus saturates. Model
            and task specific: calibrate it to the untrained checkpoint's
            average calls per solved task (read ``time_efficiency/calls/mean``
            off step 1). Required when ``lambda_call_bonus > 0``.
        floor: Lower clamp on the adjusted reward so one pathologically slow
            rollout cannot dominate its group. ``None`` disables. With the
            bonus, a correct rollout that runs a 60-minute timeout without
            earning the bonus lands on exactly 0.0 and ties with the
            failures; a small positive floor keeps it above them.
    """

    enabled: bool = False
    lambda_time: float = 1.0 / 60.0
    apply_to: Literal["all", "correct"] = "all"
    lambda_call_bonus: Annotated[float, Field(ge=0.0)] = 0.0
    call_bonus_ref: float | None = None
    floor: float | None = None

    @model_validator(mode="after")
    def _check_call_bonus_ref(self) -> "TimeEfficiencyConfig":
        if self.lambda_call_bonus > 0 and not (
            self.call_bonus_ref is not None and self.call_bonus_ref > 0
        ):
            raise ValueError(
                "grpo.time_efficiency.call_bonus_ref must be a positive number of "
                "tool calls when lambda_call_bonus > 0"
            )
        return self


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


def rollout_calls(result: dict[str, Any]) -> int:
    """Return the number of well-formed tool calls the model emitted.

    Counts ``function_call`` items in the Responses output. Calls the
    environment rejected as malformed never become ``function_call`` items,
    so they earn no bonus.
    """
    output = (result["full_result"].get("response") or {}).get("output")
    if not isinstance(output, list):
        return 0
    return sum(
        1
        for item in output
        if isinstance(item, dict) and item.get("type") == "function_call"
    )


def apply_time_efficiency_reward(
    results: list[dict[str, Any]],
    config: TimeEfficiencyConfig | None,
    *,
    mask_sample: Sequence[bool] | None = None,
) -> dict[str, float]:
    """Adjust each ``result["full_result"]["reward"]`` in place for wall time and tool calls.

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
        mask_sample: Per-row env ``mask_sample`` flags, one per result, as
            produced by the caller's loss-mask extraction. Flagged rows are not
            charged (their deduction is 0). ``None`` charges every row; pass it
            when ``env.should_mask_flagged_samples`` is off, since those rows
            then train.

    Returns:
        Metrics under the ``time_efficiency/`` prefix computed over ``results``,
        or an empty dict when the feature is disabled or ``results`` is empty.
        ``deduction`` is the realized decrease of the reward (net of the call
        bonus and the floor; negative when the bonus wins), ``bonus`` the raw
        call bonus, ``calls`` the tool-call count, ``seconds_per_call`` the
        quantity a call bonus can be gamed on (a collapse alongside rising
        ``calls/mean`` means cheap-call spam), ``bonus_saturated_frac`` and
        ``floored_frac`` the calibration signals for ``call_bonus_ref`` and
        ``floor``. Skipped rows count toward the means with 0;
        ``group_has_signal`` is computed over the rows that train.

    Raises:
        ValueError: If ``mask_sample`` is given with a length other than
            ``len(results)``.
    """
    if config is None or not config.enabled or not results:
        return {}
    if mask_sample is not None and len(mask_sample) != len(results):
        raise ValueError(
            f"mask_sample has {len(mask_sample)} entries for {len(results)} results"
        )
    masked = (
        [bool(flag) for flag in mask_sample] if mask_sample else [False] * len(results)
    )

    minutes = [rollout_minutes(result) for result in results]
    calls = [rollout_calls(result) for result in results]
    deductions: list[float] = []
    bonuses: list[float] = []
    floored = 0
    for result, rollout_min, n_calls, is_masked in zip(results, minutes, calls, masked):
        full_result = result["full_result"]
        if is_masked or (
            config.apply_to == "correct"
            and not (full_result.get("resolved") and full_result.get("reward"))
        ):
            deductions.append(0.0)
            bonuses.append(0.0)
            continue
        base = float(full_result.get("reward") or 0.0)
        bonus = 0.0
        if config.lambda_call_bonus > 0:
            # call_bonus_ref is validated to be positive when the bonus is on.
            assert config.call_bonus_ref is not None
            bonus = config.lambda_call_bonus * min(n_calls / config.call_bonus_ref, 1.0)
        new = base + bonus - config.lambda_time * rollout_min
        if config.floor is not None and new < config.floor:
            new = config.floor
            floored += 1
        deductions.append(base - new)
        bonuses.append(bonus)
        full_result["reward"] = new

    # Loss-masked rows cannot learn from the term, so the signal metric only
    # looks at the rows that train.
    trainable_deductions = [
        d for d, is_masked in zip(deductions, masked) if not is_masked
    ]

    # ``<name>/<stat>`` keys so aggregate_rollout_metrics maxes the ``/max``
    # entries across prompt groups instead of averaging them.
    n = len(results)
    return {
        "time_efficiency/minutes/mean": sum(minutes) / n,
        "time_efficiency/minutes/max": max(minutes),
        "time_efficiency/deduction/mean": sum(deductions) / n,
        "time_efficiency/deduction/max": max(deductions),
        "time_efficiency/calls/mean": sum(calls) / n,
        "time_efficiency/bonus/mean": sum(bonuses) / n,
        "time_efficiency/seconds_per_call": sum(minutes) * 60.0 / max(sum(calls), 1),
        "time_efficiency/bonus_saturated_frac": (
            sum(1 for c in calls if c >= config.call_bonus_ref) / n
            if config.lambda_call_bonus > 0 and config.call_bonus_ref
            else 0.0
        ),
        "time_efficiency/floored_frac": floored / n,
        # 1.0 when the deduction differs among the trainable rows ("correct"
        # skips count as 0), i.e. the term can still produce a gradient after
        # group normalization. 1e-6 absorbs float noise from ``base - new``.
        "time_efficiency/group_has_signal": float(
            len(trainable_deductions) > 1
            and max(trainable_deductions) - min(trainable_deductions) > 1e-6
        ),
    }
