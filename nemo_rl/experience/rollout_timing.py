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

"""Pool NeMo-Gym rollout phase times across the prompt groups of a run.

``NemoGym.run_rollouts`` already times the two phases of its result loop --
``await_results``, blocked waiting for the next Gym task, and
``postprocess_results``, the CPU-bound decode-and-tensorize step it runs inline
on the Gym actor's event loop -- and the numbers ride
``PromptGroupRecord.rollout_metrics`` back to the caller. Pooling them here lets
SingleController report what share of that loop the inline postprocess costs,
which is what decides whether moving it off the actor's event loop is worth
doing.

Kept free of heavy imports so the aggregation can be unit tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

# Suffixes of the two labels ``NemoGym.run_rollouts`` times. The prefix is the
# caller's ``timer_prefix``, so match the suffix rather than the whole key.
_AWAIT_SUFFIX = "/await_results"
_POSTPROCESS_SUFFIX = "/postprocess_results"


def _sum_with_suffix(metrics: Mapping[str, Any], suffix: str) -> Optional[float]:
    """Total the numeric entries whose key ends with ``suffix``.

    Args:
        metrics: One prompt group's rollout metrics.
        suffix: Timer label suffix to match, including its leading separator.

    Returns:
        Summed seconds, or None when no key matched.
    """
    total: Optional[float] = None
    for key, value in metrics.items():
        if key.endswith(suffix) and isinstance(value, (int, float)):
            total = (total or 0.0) + float(value)
    return total


@dataclass
class NemoGymRolloutTiming:
    """Running totals of NeMo-Gym rollout phase times over committed groups.

    Totals are cumulative over the run rather than per step, matching
    RolloutStats, the other counter the manager exposes: nothing then has to
    agree on who clears them, and a share taken over the whole run is the
    number this exists to answer. Both updating and summarizing are O(1), so an
    asyncio caller can report from the event loop it shares with the pumps
    without stalling them.
    """

    groups: int = 0
    await_seconds: float = 0.0
    postprocess_seconds: float = 0.0

    def add(self, rollout_metrics: Optional[Mapping[str, Any]]) -> None:
        """Fold one prompt group's rollout metrics into the totals.

        Metrics carrying neither label are ignored: that is what the native
        async rollout path produces, and it has no Gym postprocess to measure.

        Args:
            rollout_metrics: A group's ``PromptGroupRecord.rollout_metrics``.
        """
        if not rollout_metrics:
            return

        await_seconds = _sum_with_suffix(rollout_metrics, _AWAIT_SUFFIX)
        postprocess_seconds = _sum_with_suffix(rollout_metrics, _POSTPROCESS_SUFFIX)
        if await_seconds is None and postprocess_seconds is None:
            return

        self.groups += 1
        self.await_seconds += await_seconds or 0.0
        self.postprocess_seconds += postprocess_seconds or 0.0

    def summarize(self) -> dict[str, float]:
        """Reduce the totals to the scalars worth printing and logging.

        The absolute times sum across groups because within a group the
        generator awaits and postprocesses its results sequentially, so the two
        phases partition the time spent inside ``run_rollouts``. The share is
        taken against that sum rather than against wall time, which groups
        running concurrently would otherwise make meaningless.

        Returns:
            Scalar statistics, empty until a NeMo-Gym group has landed.
        """
        if self.groups == 0:
            return {}

        loop_seconds = self.await_seconds + self.postprocess_seconds
        return {
            "await_results": self.await_seconds,
            "postprocess_results": self.postprocess_seconds,
            "postprocess_results_pct": (
                100.0 * self.postprocess_seconds / loop_seconds
                if loop_seconds > 0
                else 0.0
            ),
            "groups": float(self.groups),
        }
