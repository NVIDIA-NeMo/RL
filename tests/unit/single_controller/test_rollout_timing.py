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

"""Tests for pooling NeMo-Gym rollout phase times across prompt groups."""

import pytest

from nemo_rl.algorithms.single_controller_utils.rollout_timing import (
    NemoGymRolloutTiming,
)


def _gym_metrics(
    await_seconds: float, postprocess_seconds: float, prefix: str = "timing/rollout"
) -> dict[str, float]:
    """Build one group's rollout metrics as run_rollouts would yield them."""
    return {
        f"{prefix}/await_results": await_seconds,
        f"{prefix}/postprocess_results": postprocess_seconds,
        f"{prefix}/postprocess_results_pct": 100.0
        * postprocess_seconds
        / (await_seconds + postprocess_seconds),
        f"{prefix}/total": await_seconds + postprocess_seconds,
        "mean_gen_tokens_per_sample": 512.0,
    }


def test_summarize_is_empty_before_any_group_lands() -> None:
    assert NemoGymRolloutTiming().summarize() == {}


def test_absolute_times_sum_and_share_is_taken_against_their_total() -> None:
    timing = NemoGymRolloutTiming()
    timing.add(_gym_metrics(await_seconds=90.0, postprocess_seconds=10.0))
    timing.add(_gym_metrics(await_seconds=70.0, postprocess_seconds=30.0))

    summary = timing.summarize()

    assert summary["await_results"] == pytest.approx(160.0)
    assert summary["postprocess_results"] == pytest.approx(40.0)
    assert summary["postprocess_results_pct"] == pytest.approx(20.0)
    assert summary["groups"] == 2.0


def test_share_weights_groups_by_time_not_by_count() -> None:
    """A long group must dominate a short one, so pooled totals drive the share."""
    timing = NemoGymRolloutTiming()
    timing.add(_gym_metrics(await_seconds=1.0, postprocess_seconds=1.0))
    timing.add(_gym_metrics(await_seconds=980.0, postprocess_seconds=20.0))

    summary = timing.summarize()

    # Averaging the per-group percentages would give 26%; the pooled share is
    # what reflects where the time actually went.
    assert summary["postprocess_results_pct"] == pytest.approx(2.1, abs=0.05)


def test_label_prefix_is_not_hardcoded() -> None:
    """run_rollouts takes its timer prefix from the caller, so match suffixes."""
    timing = NemoGymRolloutTiming()
    timing.add(_gym_metrics(await_seconds=8.0, postprocess_seconds=2.0, prefix="env"))

    summary = timing.summarize()

    assert summary["groups"] == 1.0
    assert summary["postprocess_results_pct"] == pytest.approx(20.0)


@pytest.mark.parametrize(
    "rollout_metrics",
    [
        None,
        {},
        # The native async rollout path: real metrics, no Gym phase labels.
        {"timing/rollout/total": 3.0, "mean_gen_tokens_per_sample": 128.0},
    ],
)
def test_metrics_without_gym_phase_labels_are_ignored(rollout_metrics) -> None:
    timing = NemoGymRolloutTiming()
    timing.add(rollout_metrics)

    assert timing.summarize() == {}


def test_share_is_zero_rather_than_undefined_when_both_phases_are_zero() -> None:
    timing = NemoGymRolloutTiming()
    timing.add(
        {
            "timing/rollout/await_results": 0.0,
            "timing/rollout/postprocess_results": 0.0,
        }
    )

    summary = timing.summarize()

    assert summary["postprocess_results_pct"] == 0.0
    assert summary["groups"] == 1.0


def test_reset_clears_the_window_so_the_next_step_starts_clean() -> None:
    timing = NemoGymRolloutTiming()
    timing.add(_gym_metrics(await_seconds=9.0, postprocess_seconds=1.0))
    timing.reset()

    assert timing.summarize() == {}

    timing.add(_gym_metrics(await_seconds=1.0, postprocess_seconds=3.0))
    summary = timing.summarize()

    assert summary["await_results"] == pytest.approx(1.0)
    assert summary["postprocess_results"] == pytest.approx(3.0)
    assert summary["groups"] == 1.0
