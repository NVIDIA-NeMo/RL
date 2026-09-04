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

"""Vocabulary for cross-DP admission: policy modes and the prose that pins
down what each recorded field actually means.

The semantics strings are part of the result contract — analysis code asserts
on them — so they live next to the modes they describe rather than inline in
the scheduler.
"""

from __future__ import annotations

from typing import Literal

CrossDpMode = Literal[
    "fcfs",
    "lfs",
    "predicted_lfs",
    "history_lfs",
    "oracle_probe_lfs",
    "exact_length_lpt",
]
PROBE_LFS_MODES = ("lfs", "oracle_probe_lfs")
ONLINE_LFS_MODES = ("lfs", "predicted_lfs", "oracle_probe_lfs")
DpSelectionMode = Literal["static_cost", "inflight_count"]

STATIC_ADMISSION_COST_SEMANTICS = (
    "Piecewise-static estimated total generation cost. It may be rebased "
    "when a same-group request completes, is never decremented for generated "
    "progress, and is not remaining work."
)
EXPLICIT_PROBE_SELECTION_SEMANTICS = "explicit_catalog_flag-v1"
IMPLICIT_PROBE_SELECTION_SEMANTICS = "implicit_first_pending-v1"
SCHEDULER_SELECTED_DP_PLACEMENT = "scheduler_selected"
PREFERRED_DP_PINNED_PLACEMENT = "preferred_dp_pinned"
PREFERRED_DP_PINNING_SEMANTICS = (
    "Exact-length diagnostic only: every request in the bounded first-turn "
    "session catalog has an immutable preferred DP. Later dynamically "
    "discovered requests are rejected because they have no routing manifest."
)
LFS_ADMISSION_FAIRNESS_POLICY = (
    "prose-inspired-idle-group-admission-age-v1"
)
LFS_ADMISSION_FAIRNESS_SEMANTICS = (
    "Exploratory admission-age safeguard inspired by the reference paper's "
    "prose-only underserved-group note. It is not that paper's Algorithm 2 "
    "and is not chunk-level shortest-attained-service scheduling. Every Nth "
    "ordinary admission opportunity, after no probe-tier candidate remains "
    "in the oldest session, selects the pending zero-inflight group with the "
    "oldest dispatcher admission; stable ties use catalog-group and front-"
    "request arrival order. A due opportunity with no eligible group falls "
    "back to ordinary LFS without carrying credit."
)
