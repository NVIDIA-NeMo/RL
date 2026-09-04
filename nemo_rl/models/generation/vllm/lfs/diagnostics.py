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

"""Read-only introspection for the cross-DP admission scheduler.

Everything here explains a decision after the fact: the snapshot the
dispatcher publishes, and the per-assignment counterfactual answering "what
would plain FCFS have picked instead?".  None of it feeds back into
admission, so it stays outside the scheduler's state machine.
"""

from __future__ import annotations

from typing import Any

from nemo_rl.models.generation.vllm.lfs.modes import (
    EXPLICIT_PROBE_SELECTION_SEMANTICS,
    LFS_ADMISSION_FAIRNESS_POLICY,
    LFS_ADMISSION_FAIRNESS_SEMANTICS,
    ONLINE_LFS_MODES,
    PREFERRED_DP_PINNING_SEMANTICS,
    PROBE_LFS_MODES,
    STATIC_ADMISSION_COST_SEMANTICS,
)
from nemo_rl.models.generation.vllm.lfs.state import Request

def build_snapshot(state) -> dict[str, Any]:
    return {
        "mode": state.mode,
        "dp_selection_mode": state.dp_selection_mode,
        "preferred_dp_pinning_semantics": (
            PREFERRED_DP_PINNING_SEMANTICS
        ),
        "lfs_admission_fairness": {
            "policy": LFS_ADMISSION_FAIRNESS_POLICY,
            "semantics": LFS_ADMISSION_FAIRNESS_SEMANTICS,
            "interval": state.lfs_admission_fairness_interval,
            "ordinary_admission_opportunity_count": (
                state.ordinary_admission_opportunities
            ),
            "due_count": state.admission_fairness_due_count,
            "selected_count": state.admission_fairness_selected_count,
            "override_count": state.admission_fairness_override_count,
            "noop_count": state.admission_fairness_noop_count,
            "no_candidate_count": (
                state.admission_fairness_no_candidate_count
            ),
        },
        "dp_size": state.dp_size,
        "max_num_seqs_per_dp": state.max_num_seqs_per_dp,
        "lookahead_per_dp": state.lookahead_per_dp,
        "admission_limit_per_dp": state.admission_limit_per_dp,
        "global_admission_limit": state.global_admission_limit,
        "max_total_inflight_observed": (
            state.max_total_inflight_observed
        ),
        "max_inflight_observed_by_dp": list(
            state.max_inflight_observed_by_dp
        ),
        "inflight_high_watermark_semantics": (
            "Exact dispatcher state after every occupancy-increasing "
            "assignment since actor construction; completions and "
            "rollbacks can only decrease occupancy."
        ),
        "fatal_error": state.fatal_error,
        "dp_inflight": [sorted(items) for items in state.dp_inflight],
        "dp_static_admission_cost": [
            sum(items.values()) for items in state.dp_inflight
        ],
        "dp_load_accounting_semantics": (
            STATIC_ADMISSION_COST_SEMANTICS
        ),
        "pending": state._pending_request_count,
        "pending_preferred_dp": (
            state._pending_preferred_dp_request_count
        ),
        "pending_counter_matches_request_states": (
            state._pending_request_count
            == sum(
                request.status == "pending"
                for request in state.requests.values()
            )
        ),
        "pending_preferred_dp_counter_matches_request_states": (
            state._pending_preferred_dp_request_count
            == sum(
                request.status == "pending"
                and request.preferred_dp_idx is not None
                for request in state.requests.values()
            )
        ),
        "inflight_group_index_matches_dp_inflight": (
            {
                request_id
                for inflight in state.dp_inflight
                for request_id in inflight
            }
            == {
                request_id
                for request_ids in state._inflight_by_group.values()
                for request_id in request_ids
            }
        ),
        "sessions": {
            session_id: {
                "open_participants": sorted(session.open_participants),
                "estimates": dict(session.estimates),
                "failed_error": session.failed_error,
                "dp_placement_mode": session.dp_placement_mode,
                "probe_selection_semantics": (
                    session.probe_selection_semantics
                ),
                "designated_probe_request_ids": dict(
                    session.designated_probe_request_ids
                ),
                "group_catalog_ordinals": dict(
                    session.group_catalog_ordinals
                ),
                "last_group_admission_sequences": dict(
                    session.last_group_admission_sequences
                ),
                "ordinary_admission_opportunity_count": (
                    session.ordinary_admission_opportunities
                ),
                "admission_fairness_due_count": (
                    session.admission_fairness_due_count
                ),
                "admission_fairness_selected_count": (
                    session.admission_fairness_selected_count
                ),
                "admission_fairness_override_count": (
                    session.admission_fairness_override_count
                ),
                "admission_fairness_noop_count": (
                    session.admission_fairness_noop_count
                ),
                "admission_fairness_no_candidate_count": (
                    session.admission_fairness_no_candidate_count
                ),
            }
            for session_id, session in state.sessions.items()
        },
        "assignment_history": list(state.assignment_history),
        "group_history": {
            group_id: {
                "mean": total / count,
                "sum": total,
                "count": count,
            }
            for group_id, (total, count) in state.group_history.items()
            if count > 0
        },
        "history_update_history": list(state.history_update_history),
        "estimate_rebase_history": list(state.estimate_rebase_history),
    }

def diagnostic_group_front(
    state, group_key: tuple[str, str]
) -> Request | None:
    request_ids = state._diagnostic_pending_by_group.get(group_key)
    if request_ids is None:
        return None
    while request_ids:
        request = state.requests.get(request_ids[0])
        if request is None or request.status != "pending":
            request_ids.popleft()
            continue
        session = state.sessions.get(request.session_id)
        if session is None or session.failed_error is not None:
            request_ids.popleft()
            continue
        return request
    return None

def diagnostic_policy_group_front(
    state, group_key: tuple[str, str]
) -> Request | None:
    """Reconstruct a policy front without consulting the policy queue.

    The diagnostic index intentionally preserves catalog order so it can
    reconstruct an FCFS counterfactual. Explicit probe designation is the
    one mode-specific within-group reorder, so reconstruct that choice
    from the immutable session manifest instead of reading
    ``_pending_by_group`` (the queue whose behavior is being checked).
    """

    catalog_front = diagnostic_group_front(state, group_key)
    if catalog_front is None:
        return None
    if state.mode not in PROBE_LFS_MODES:
        return catalog_front
    session = state.sessions[catalog_front.session_id]
    if (
        session.probe_selection_semantics
        != EXPLICIT_PROBE_SELECTION_SEMANTICS
        or catalog_front.group_id in session.probed_groups
    ):
        return catalog_front
    designated_request_id = session.designated_probe_request_ids.get(
        catalog_front.group_id
    )
    designated_request = state.requests.get(
        designated_request_id
    )
    if (
        designated_request is not None
        and designated_request.status == "pending"
        and designated_request.session_id == catalog_front.session_id
        and designated_request.group_id == catalog_front.group_id
    ):
        return designated_request
    # A cancelled or definitively failed designated request is replaced by
    # the first remaining catalog request, matching
    # _prioritize_designated_probes().
    return catalog_front

def pending_choice_diagnostics(state) -> dict[str, Any]:
    """Describe whether policy priority can still change admission order."""

    catalog_group_fronts: dict[tuple[str, str], Request] = {}
    policy_group_fronts: dict[tuple[str, str], Request] = {}
    for group_key in list(state._diagnostic_pending_by_group):
        catalog_request = diagnostic_group_front(state, group_key)
        if catalog_request is None:
            state._diagnostic_pending_by_group.pop(group_key, None)
            continue
        policy_request = diagnostic_policy_group_front(state, group_key)
        if policy_request is None:
            raise AssertionError(
                "policy diagnostic lost a group with a catalog front"
            )
        catalog_group_fronts[group_key] = catalog_request
        policy_group_fronts[group_key] = policy_request
    if not catalog_group_fronts:
        if state._pending_request_count != 0:
            raise AssertionError(
                "pending request counter is nonzero without any pending "
                "group front"
            )
        return {
            "pending_request_count_before_assignment": 0,
            "pending_group_count_before_assignment": 0,
            "pending_unprobed_group_count_before_assignment": 0,
            "pending_unknown_group_count_before_assignment": 0,
            "pending_finite_group_count_before_assignment": 0,
            "pending_finite_estimate_min": None,
            "pending_finite_estimate_max": None,
            "policy_active_priority_tier": None,
            "policy_eligible_session_ids": [],
            "policy_eligible_pending_groups": [],
            "policy_eligible_pending_finite_group_count": 0,
            "policy_eligible_pending_finite_estimate_min": None,
            "policy_eligible_pending_finite_estimate_max": None,
            "policy_priority_class_count_before_assignment": 0,
            "policy_expected_request_id": None,
            "policy_expected_group_id": None,
            "policy_expected_priority_key": None,
            "base_policy_expected_request_id": None,
            "base_policy_expected_group_id": None,
            "base_policy_expected_priority_key": None,
            "ordinary_admission_opportunity": False,
            "ordinary_admission_ordinal": None,
            "admission_fairness_interval": (
                state.lfs_admission_fairness_interval
            ),
            "admission_fairness_policy": (
                LFS_ADMISSION_FAIRNESS_POLICY
            ),
            "admission_fairness_semantics": (
                LFS_ADMISSION_FAIRNESS_SEMANTICS
            ),
            "admission_fairness_due": False,
            "admission_fairness_eligible_pending_groups": [],
            "admission_fairness_candidate_count": 0,
            "admission_fairness_expected_request_id": None,
            "admission_fairness_expected_group_id": None,
            "admission_fairness_expected_priority_key": None,
            "admission_fairness_selected": False,
            "admission_fairness_changed_base_choice": False,
            "admission_fairness_due_but_no_candidate": False,
            "admission_selection_reason": None,
            "fcfs_front_request_id": None,
        }

    fcfs_front = min(
        catalog_group_fronts.values(),
        key=lambda request: (
            state.sessions[request.session_id].arrival_seq,
            request.arrival_seq,
        ),
    )

    oldest_session_arrival_seq = min(
        state.sessions[request.session_id].arrival_seq
        for request in policy_group_fronts.values()
    )
    oldest_session_fronts = [
        request
        for request in policy_group_fronts.values()
        if state.sessions[request.session_id].arrival_seq
        == oldest_session_arrival_seq
    ]
    if state.mode in (
        "lfs",
        "predicted_lfs",
        "history_lfs",
        "oracle_probe_lfs",
    ):
        policy_active_priority_tier = min(
            state._lfs_key(request)[1]
            for request in oldest_session_fronts
        )
        policy_eligible_fronts = [
            request
            for request in oldest_session_fronts
            if state._lfs_key(request)[1]
            == policy_active_priority_tier
        ]
    else:
        policy_active_priority_tier = None
        policy_eligible_fronts = oldest_session_fronts
    base_policy_expected_front = (
        min(policy_eligible_fronts, key=state._lfs_key)
        if state.mode in ONLINE_LFS_MODES
        else None
    )
    ordinary_admission_opportunity = (
        state.mode in ONLINE_LFS_MODES
        and policy_active_priority_tier == 2
    )
    fairness_session = (
        state.sessions[oldest_session_fronts[0].session_id]
        if ordinary_admission_opportunity
        else None
    )
    ordinary_admission_ordinal = (
        fairness_session.ordinary_admission_opportunities + 1
        if fairness_session is not None
        else None
    )
    admission_fairness_due = bool(
        ordinary_admission_ordinal is not None
        and state.lfs_admission_fairness_interval > 0
        and ordinary_admission_ordinal
        % state.lfs_admission_fairness_interval
        == 0
    )
    admission_fairness_eligible_fronts = (
        [
            request
            for request in policy_eligible_fronts
            if not state._inflight_by_group.get(
                (request.session_id, request.group_id)
            )
        ]
        if admission_fairness_due
        else []
    )
    admission_fairness_expected_front = (
        min(
            admission_fairness_eligible_fronts,
            key=state._admission_fairness_key,
        )
        if admission_fairness_eligible_fronts
        else None
    )
    admission_fairness_selected = bool(
        admission_fairness_due
        and admission_fairness_expected_front is not None
    )
    policy_expected_front = (
        admission_fairness_expected_front
        if admission_fairness_selected
        else base_policy_expected_front
    )
    admission_fairness_changed_base_choice = bool(
        admission_fairness_selected
        and admission_fairness_expected_front is not None
        and base_policy_expected_front is not None
        and admission_fairness_expected_front.request_id
        != base_policy_expected_front.request_id
    )
    admission_fairness_due_but_no_candidate = bool(
        admission_fairness_due
        and admission_fairness_expected_front is None
    )
    if state.mode not in ONLINE_LFS_MODES:
        admission_selection_reason = None
    elif policy_active_priority_tier == 0:
        admission_selection_reason = "probe_priority"
    elif admission_fairness_due_but_no_candidate:
        admission_selection_reason = "fairness_no_candidate_fallback"
    elif admission_fairness_selected:
        admission_selection_reason = (
            "fairness_override"
            if admission_fairness_changed_base_choice
            else "fairness_noop"
        )
    else:
        admission_selection_reason = "base_lfs"

    unprobed_count = 0
    unknown_count = 0
    finite_estimates: list[int] = []
    priority_classes: set[tuple[int, float, int]] = set()
    for request in policy_group_fronts.values():
        session = state.sessions[request.session_id]
        estimate = session.estimates.get(request.group_id)
        if state.mode in ONLINE_LFS_MODES:
            if (
                estimate is None
                and request.group_id not in session.probed_groups
            ):
                unprobed_count += 1
                priority_classes.add((0, 0.0, 0))
            elif estimate is None:
                unknown_count += 1
                priority_classes.add(
                    (
                        2,
                        -float(request.fallback_cost),
                        session.unknown_admissions.get(
                            request.group_id, 0
                        ),
                    )
                )
            else:
                finite_estimates.append(int(estimate))
                priority_classes.add(
                    (
                        2,
                        -float(max(1, estimate)),
                        session.unknown_admissions.get(
                            request.group_id, 0
                        ),
                    )
                )
        elif estimate is not None:
            finite_estimates.append(int(estimate))

    eligible_finite_estimates = [
        int(estimate)
        for request in policy_eligible_fronts
        if (
            estimate := state.sessions[
                request.session_id
            ].estimates.get(request.group_id)
        )
        is not None
    ]
    policy_eligible_pending_groups = []
    for request in sorted(
        policy_eligible_fronts,
        key=lambda item: (
            state.sessions[item.session_id].arrival_seq,
            item.arrival_seq,
        ),
    ):
        request_session = state.sessions[request.session_id]
        request_estimate = request_session.estimates.get(request.group_id)
        effective_estimate = (
            request.fallback_cost
            if request_estimate is None
            else max(1, request_estimate)
        )
        policy_eligible_pending_groups.append(
            {
                "session_id": request.session_id,
                "session_arrival_seq": request_session.arrival_seq,
                "group_id": request.group_id,
                "request_id": request.request_id,
                "request_arrival_seq": request.arrival_seq,
                "fallback_cost": request.fallback_cost,
                "estimate": request_estimate,
                "effective_estimate": effective_estimate,
                "unknown_admission_count": (
                    request_session.unknown_admissions.get(
                        request.group_id, 0
                    )
                ),
                "group_catalog_ordinal": (
                    request_session.group_catalog_ordinals[
                        request.group_id
                    ]
                ),
                "last_group_admission_sequence": (
                    request_session.last_group_admission_sequences.get(
                        request.group_id, -1
                    )
                ),
                "group_inflight_count": len(
                    state._inflight_by_group.get(
                        (request.session_id, request.group_id), set()
                    )
                ),
                "policy_priority_key": list(state._lfs_key(request)),
            }
        )

    admission_fairness_eligible_pending_groups = []
    for request in sorted(
        admission_fairness_eligible_fronts,
        key=lambda item: (
            state.sessions[item.session_id].arrival_seq,
            state.sessions[item.session_id].group_catalog_ordinals[
                item.group_id
            ],
            item.arrival_seq,
        ),
    ):
        request_session = state.sessions[request.session_id]
        admission_fairness_eligible_pending_groups.append(
            {
                "session_id": request.session_id,
                "session_arrival_seq": request_session.arrival_seq,
                "group_id": request.group_id,
                "request_id": request.request_id,
                "request_arrival_seq": request.arrival_seq,
                "group_catalog_ordinal": (
                    request_session.group_catalog_ordinals[
                        request.group_id
                    ]
                ),
                "last_group_admission_sequence": (
                    request_session.last_group_admission_sequences.get(
                        request.group_id, -1
                    )
                ),
                "group_inflight_count": 0,
                "policy_priority_key": list(
                    state._lfs_key(request)
                ),
                "admission_fairness_priority_key": list(
                    state._admission_fairness_key(request)
                ),
            }
        )

    return {
        "pending_request_count_before_assignment": (
            state._pending_request_count
        ),
        "pending_group_count_before_assignment": len(
            policy_group_fronts
        ),
        "pending_unprobed_group_count_before_assignment": (
            unprobed_count
        ),
        "pending_unknown_group_count_before_assignment": unknown_count,
        "pending_finite_group_count_before_assignment": len(
            finite_estimates
        ),
        "pending_finite_estimate_min": (
            min(finite_estimates) if finite_estimates else None
        ),
        "pending_finite_estimate_max": (
            max(finite_estimates) if finite_estimates else None
        ),
        "policy_active_priority_tier": (
            policy_active_priority_tier
        ),
        "policy_eligible_session_ids": sorted(
            {request.session_id for request in policy_eligible_fronts}
        ),
        "policy_eligible_pending_groups": (
            policy_eligible_pending_groups
        ),
        "policy_eligible_pending_finite_group_count": len(
            eligible_finite_estimates
        ),
        "policy_eligible_pending_finite_estimate_min": (
            min(eligible_finite_estimates)
            if eligible_finite_estimates
            else None
        ),
        "policy_eligible_pending_finite_estimate_max": (
            max(eligible_finite_estimates)
            if eligible_finite_estimates
            else None
        ),
        "policy_priority_class_count_before_assignment": (
            len(priority_classes)
            if state.mode in ONLINE_LFS_MODES
            else None
        ),
        "policy_expected_request_id": (
            policy_expected_front.request_id
            if policy_expected_front is not None
            else None
        ),
        "policy_expected_group_id": (
            policy_expected_front.group_id
            if policy_expected_front is not None
            else None
        ),
        "policy_expected_priority_key": (
            list(state._lfs_key(policy_expected_front))
            if policy_expected_front is not None
            else None
        ),
        "base_policy_expected_request_id": (
            base_policy_expected_front.request_id
            if base_policy_expected_front is not None
            else None
        ),
        "base_policy_expected_group_id": (
            base_policy_expected_front.group_id
            if base_policy_expected_front is not None
            else None
        ),
        "base_policy_expected_priority_key": (
            list(state._lfs_key(base_policy_expected_front))
            if base_policy_expected_front is not None
            else None
        ),
        "ordinary_admission_opportunity": (
            ordinary_admission_opportunity
        ),
        "ordinary_admission_ordinal": ordinary_admission_ordinal,
        "admission_fairness_interval": (
            state.lfs_admission_fairness_interval
        ),
        "admission_fairness_policy": (
            LFS_ADMISSION_FAIRNESS_POLICY
        ),
        "admission_fairness_semantics": (
            LFS_ADMISSION_FAIRNESS_SEMANTICS
        ),
        "admission_fairness_due": admission_fairness_due,
        "admission_fairness_eligible_pending_groups": (
            admission_fairness_eligible_pending_groups
        ),
        "admission_fairness_candidate_count": len(
            admission_fairness_eligible_pending_groups
        ),
        "admission_fairness_expected_request_id": (
            admission_fairness_expected_front.request_id
            if admission_fairness_expected_front is not None
            else None
        ),
        "admission_fairness_expected_group_id": (
            admission_fairness_expected_front.group_id
            if admission_fairness_expected_front is not None
            else None
        ),
        "admission_fairness_expected_priority_key": (
            list(
                state._admission_fairness_key(
                    admission_fairness_expected_front
                )
            )
            if admission_fairness_expected_front is not None
            else None
        ),
        "admission_fairness_selected": (
            admission_fairness_selected
        ),
        "admission_fairness_changed_base_choice": (
            admission_fairness_changed_base_choice
        ),
        "admission_fairness_due_but_no_candidate": (
            admission_fairness_due_but_no_candidate
        ),
        "admission_selection_reason": admission_selection_reason,
        "fcfs_front_request_id": fcfs_front.request_id,
    }
