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

"""In-flight bookkeeping for one admission session.

``Request`` is a single trajectory the dispatcher may hand to a DP rank;
``Session`` is the bounded first-turn catalog a rollout opens in one shot.
Both are plain records: every transition lives in :mod:`scheduler`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nemo_rl.models.generation.vllm.lfs.modes import (
    IMPLICIT_PROBE_SELECTION_SEMANTICS,
    SCHEDULER_SELECTED_DP_PLACEMENT,
)

@dataclass
class Request:
    session_id: str
    participant_id: str
    request_id: str
    group_id: str
    arrival_seq: int
    fallback_cost: int
    status: str = "pending"
    dp_idx: int | None = None
    preferred_dp_idx: int | None = None
    predicted_length: int | None = None
    assignment_sequence: int | None = None
    dp_assignment_ordinal: int | None = None
    session_dp_assignment_ordinal: int | None = None
    lease_started: bool = False
    unknown_admission: bool = False
    probe_admission: bool = False
    is_designated_probe: bool = False
    admission_fairness_selected: bool = False
    ordinary_admission_ordinal: int | None = None
    admission_selection_reason: str | None = None


@dataclass
class Session:
    session_id: str
    arrival_seq: int
    open_participants: set[str]
    request_ids: set[str] = field(default_factory=set)
    participant_requests: dict[str, set[str]] = field(default_factory=dict)
    estimates: dict[str, int] = field(default_factory=dict)
    probed_groups: set[str] = field(default_factory=set)
    unknown_admissions: dict[str, int] = field(default_factory=dict)
    failed_error: str | None = None
    completed_lengths: dict[str, list[int]] = field(default_factory=dict)
    dp_assignment_ordinals: list[int] = field(default_factory=list)
    # Benchmark-only truth, kept separate from estimates so the ordinary
    # probe/unknown exploration wave remains unchanged.
    oracle_estimates: dict[str, int] = field(default_factory=dict)
    probe_selection_semantics: str = IMPLICIT_PROBE_SELECTION_SEMANTICS
    designated_probe_request_ids: dict[str, str] = field(default_factory=dict)
    group_catalog_ordinals: dict[str, int] = field(default_factory=dict)
    last_group_admission_sequences: dict[str, int] = field(default_factory=dict)
    ordinary_admission_opportunities: int = 0
    admission_fairness_due_count: int = 0
    admission_fairness_selected_count: int = 0
    admission_fairness_override_count: int = 0
    admission_fairness_noop_count: int = 0
    admission_fairness_no_candidate_count: int = 0
    dp_placement_mode: str = SCHEDULER_SELECTED_DP_PLACEMENT
