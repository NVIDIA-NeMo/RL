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

"""Weighted task mixing for multi-dataset RL training.

A training step draws prompts from several datasets at configured proportions.
This module owns the vocabulary for that mixture and the arithmetic that turns
relative weights into exact per-task counts.
"""

from dataclasses import dataclass
from typing import Any, TypeAlias

#: Unique task identifier; equals the dataset's ``task_name``.
TaskName: TypeAlias = str

#: Normalized sampling weight per task. Sums to 1.0 across non-evaluation
#: tasks; evaluation-only tasks are present with weight 0.0.
TaskWeights: TypeAlias = dict[TaskName, float]

#: Number of prompt groups each task contributes to one training step.
TaskQuota: TypeAlias = dict[TaskName, int]

#: Per-task shortfall against a :data:`TaskQuota`. Zero means satisfied.
TaskDeficits: TypeAlias = dict[TaskName, int]

#: Per-task ``StatefulDataLoader.state_dict()`` payloads, keyed by task.
TaskDataloaderState: TypeAlias = dict[TaskName, dict[str, Any]]

#: Task label used when a trajectory carries no task attribution, so parallel
#: task lists stay populated. No quota key ever matches this, which turns an
#: unlabeled weighted run into a loud error instead of a silent stall.
UNWEIGHTED_TASK_NAME: TaskName = "_unweighted"


@dataclass
class TaskWeightSpec:
    """One weighted training task, as declared by a ``data.train`` entry."""

    task_name: TaskName
    weight: float | None  # None means "not declared in config"
    evaluation_only: bool


def normalize_weights(specs: list[TaskWeightSpec]) -> TaskWeights:
    """Normalize weights across non-evaluation tasks.

    Returns an empty mapping when no task declares a weight, which signals the
    legacy unweighted path. Otherwise every non-evaluation task must declare
    one -- there is no implicit default, since v1 TypedDict config defaults live
    in the exemplar YAML rather than at the call site.

    Args:
        specs: One entry per ``data.train`` dataset.

    Returns:
        Task name to normalized weight. Evaluation-only tasks map to 0.0 and are
        excluded from the denominator.

    Raises:
        ValueError: If weights are declared on only some non-evaluation tasks,
            if any weight is negative, or if the non-evaluation total is not
            positive.
    """
    if all(spec.weight is None for spec in specs):
        return {}

    missing = [s.task_name for s in specs if not s.evaluation_only and s.weight is None]
    if missing:
        raise ValueError(
            f"Some training datasets declare `weight` but these do not: {missing}. "
            "Weights are all-or-nothing; set an explicit weight on every "
            "non-evaluation_only entry in data.train."
        )

    negative = [s.task_name for s in specs if s.weight is not None and s.weight < 0]
    if negative:
        raise ValueError(
            f"Dataset weights must be non-negative; got negative for {negative}"
        )

    total = sum(s.weight for s in specs if not s.evaluation_only)
    if total <= 0:
        raise ValueError(
            "Total weight of non-evaluation_only training datasets must be positive."
        )

    return {s.task_name: 0.0 if s.evaluation_only else s.weight / total for s in specs}


def distribute_counts(
    total_count: int,
    weights: list[float],
    distribute_remainder: bool = True,
) -> list[int]:
    """Split ``total_count`` across ``weights`` by largest-remainder apportionment.

    Integer parts are assigned first; any remainder goes to the entries with the
    largest fractional parts. This keeps the split exact and reproducible -- the
    same weights always produce the same counts.

    Args:
        total_count: Total number of items to distribute.
        weights: Normalized weights, one per entry.
        distribute_remainder: When False, only the integer parts are returned
            and the result may sum to less than ``total_count``.

    Returns:
        Per-entry counts, summing to ``total_count`` when ``distribute_remainder``.
    """
    exact_counts = [total_count * w for w in weights]
    base_counts = [int(count) for count in exact_counts]
    remaining = total_count - sum(base_counts)

    if distribute_remainder:
        fractional_parts = [count - int(count) for count in exact_counts]
        indices = sorted(
            range(len(weights)), key=lambda i: fractional_parts[i], reverse=True
        )
        for i in range(remaining):
            base_counts[indices[i]] += 1

    return base_counts


def compute_quota(total_count: int, weights: TaskWeights) -> TaskQuota:
    """Per-task prompt quota for one training step.

    This is the single source of truth for both the per-task dataloader batch
    sizes (sync) and the replay-buffer release gate (async).

    Args:
        total_count: Prompt groups in one training step.
        weights: Normalized weights from :func:`normalize_weights`.

    Returns:
        Task name to prompt-group count. Zero-weight (evaluation-only) tasks are
        omitted rather than mapped to 0, so callers can iterate the quota to get
        exactly the training tasks.
    """
    task_names = [task for task, weight in weights.items() if weight > 0]
    counts = distribute_counts(total_count, [weights[task] for task in task_names])
    return dict(zip(task_names, counts, strict=True))
