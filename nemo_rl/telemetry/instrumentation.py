# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Instrumentation helpers that attach efficiency tags.

Algorithms should import ``managed_span`` / ``trace_fn`` from here (not raw
nemo-lens) so every leaf span gets ``rl.bucket`` when applicable.

Efficiency tagging
------------------

* Shared bucket tokens: ``productive`` | ``overhead`` | ``idle`` | ``wasted``.
* Umbrella groups (``job``, ``step``, ``rollout``, …) are timed but **not**
  tagged.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from enum import Enum
from typing import Any, Mapping, Optional

from nemo_rl.telemetry._fallbacks import (
    is_span_group_enabled,
    managed_span as _managed_span,
    safe_set_span_attributes,
    span_cm,
)
from nemo_rl.telemetry.span_groups import RLSpanGroup

# OTel / OneLogger-shared attribute key (flat sinks encode this in the name).
RL_BUCKET_ATTR = "rl.bucket"

__all__ = [
    "managed_span",
    "trace_fn",
    "span_cm",
    "is_span_group_enabled",
    "safe_set_span_attributes",
    "RL_BUCKET_ATTR",
    "Bucket",
    "UMBRELLA_GROUPS",
    "EFFICIENCY_CATEGORY_BUCKET",
    "bucket_for_span_group",
    "bucket_for_efficiency_category",
    "goodput_span_attributes",
]


class Bucket(str, Enum):
    """Shared goodput buckets."""

    PRODUCTIVE = "productive"
    OVERHEAD = "overhead"
    IDLE = "idle"
    WASTED = "wasted"


# Span groups that are umbrellas / lifecycle only — no rl.bucket tag.
UMBRELLA_GROUPS: frozenset[str] = frozenset(
    {
        RLSpanGroup.JOB,
        RLSpanGroup.STEP,
        RLSpanGroup.ROLLOUT,  # collect_rollouts umbrella (like Cosmos generate)
        RLSpanGroup.MODEL_INIT,
        RLSpanGroup.EVALUATE,  # eval pass; treat as umbrella unless timed as idle
    }
)

# Default classification for RLSpanGroup members that are leaf work.
# logprob / advantage / reference_policy count as overhead (prep), not the
# productive policy gradient update itself.
_DEFAULT_GROUP_BUCKET: Mapping[str, Bucket] = {
    RLSpanGroup.GENERATION: Bucket.PRODUCTIVE,
    RLSpanGroup.REWARD: Bucket.PRODUCTIVE,
    RLSpanGroup.POLICY_UPDATE: Bucket.PRODUCTIVE,
    RLSpanGroup.FORWARD_BACKWARD: Bucket.PRODUCTIVE,
    RLSpanGroup.OPTIMIZER: Bucket.PRODUCTIVE,
    RLSpanGroup.DATA_PROCESSING: Bucket.OVERHEAD,
    RLSpanGroup.CHECKPOINT: Bucket.OVERHEAD,
    RLSpanGroup.LOAD_CHECKPOINT: Bucket.OVERHEAD,
    RLSpanGroup.LOGPROB: Bucket.OVERHEAD,
    RLSpanGroup.ADVANTAGE: Bucket.OVERHEAD,
    RLSpanGroup.REFERENCE_POLICY: Bucket.OVERHEAD,
}

# Async efficiency category labels → bucket (when emitted as phase metrics).
# These are not RLSpanGroup members; listed for monitor / future metric tee.
EFFICIENCY_CATEGORY_BUCKET: Mapping[str, Bucket] = {
    "init/total": Bucket.OVERHEAD,
    "idle/buffer_starvation": Bucket.IDLE,
    "idle/refit_bubble": Bucket.IDLE,
    "idle/validation": Bucket.IDLE,
    "idle/buffer_full_backoff": Bucket.IDLE,
    "idle/generation_limit_pause": Bucket.IDLE,
    "idle/refit_event_wait": Bucket.IDLE,
    "wasted/failed_trajectory": Bucket.WASTED,
}


def bucket_for_span_group(group: str) -> Optional[Bucket]:
    """Return the goodput bucket for a span group, or None if umbrella / unknown.

    Unknown non-umbrella groups default to ``overhead`` so new leaves are not
    silently dropped from the denominator.
    """
    if group in UMBRELLA_GROUPS:
        return None
    if group in _DEFAULT_GROUP_BUCKET:
        return _DEFAULT_GROUP_BUCKET[group]
    return Bucket.OVERHEAD


def bucket_for_efficiency_category(category: str) -> Optional[Bucket]:
    """Return the bucket for an async efficiency category label, if known."""
    return EFFICIENCY_CATEGORY_BUCKET.get(category)


def goodput_span_attributes(group: str) -> dict[str, str]:
    """Attributes to merge into ``managed_span`` for *group*.

    Empty when the group is an umbrella (no ``rl.bucket``).
    """
    bucket = bucket_for_span_group(group)
    if bucket is None:
        return {}
    return {RL_BUCKET_ATTR: bucket.value}


@contextmanager
def managed_span(group: str, name: str, tracer=None, **attributes: Any):
    """Like lens ``managed_span``, but injects ``rl.bucket`` for leaf groups.

    Callers may override by passing ``rl.bucket=...`` explicitly. Umbrella
    groups (job / step / rollout / …) receive no bucket attribute.
    """
    attrs = dict(attributes)
    if RL_BUCKET_ATTR not in attrs:
        attrs.update(goodput_span_attributes(group))
    with _managed_span(group, name, tracer=tracer, **attrs) as span:
        yield span


def trace_fn(group: str, name: str, tracer=None):
    """Decorator that wraps a function in a bucket-tagged ``managed_span``."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with managed_span(group, name, tracer=tracer):
                return func(*args, **kwargs)

        return wrapper

    return decorator
