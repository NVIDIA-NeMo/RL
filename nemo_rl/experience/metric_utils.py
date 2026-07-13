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

"""Shared aggregation helpers for rollout metrics."""

import math
import statistics
from collections.abc import Sequence

from wandb import Histogram


def calculate_single_metric(
    values: Sequence[float | int], batch_size: int, key_name: str
) -> dict:
    return {
        f"{key_name}/mean": sum(values) / batch_size,
        f"{key_name}/max": max(values),
        f"{key_name}/min": min(values),
        f"{key_name}/median": statistics.median(values),
        f"{key_name}/stddev": statistics.stdev(values) if len(values) > 1 else math.nan,
        f"{key_name}/histogram": Histogram(values),
    }


def pct(values: Sequence[float | int], p: float) -> float:
    """Percentile helper for buffer starvation diagnostics."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = min(int(len(sorted_v) * p / 100), len(sorted_v) - 1)
    return float(sorted_v[idx])
