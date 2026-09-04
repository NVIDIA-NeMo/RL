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

"""Global probe/LFS admission for async vLLM data-parallel engines.

The vLLM schedulers only see one data-parallel engine.  This module owns the
small middleware queue immediately above them: it picks both the next request
and its DP rank, while each engine keeps its vanilla FCFS scheduler.

``CrossDpSchedulerState`` is deliberately free of Ray and asyncio so the
ordering and capacity invariants stay deterministically testable.  Its Ray
wrapper lives in :mod:`dispatcher`.
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Any

from nemo_rl.models.generation.vllm.lfs.diagnostics import build_snapshot, pending_choice_diagnostics
from nemo_rl.models.generation.vllm.lfs.modes import (
    EXPLICIT_PROBE_SELECTION_SEMANTICS,
    IMPLICIT_PROBE_SELECTION_SEMANTICS,
    LFS_ADMISSION_FAIRNESS_POLICY,
    ONLINE_LFS_MODES,
    PREFERRED_DP_PINNED_PLACEMENT,
    PROBE_LFS_MODES,
    SCHEDULER_SELECTED_DP_PLACEMENT,
    STATIC_ADMISSION_COST_SEMANTICS,
    CrossDpMode,
    DpSelectionMode,
)
from nemo_rl.models.generation.vllm.lfs.state import Request, Session
from nemo_rl.models.generation.vllm.lfs.validation import (
    inspect_catalog,
    resolve_global_admission_limit,
    validate_scheduler_config,
)

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









class CrossDpSchedulerState:
    """Pure state machine for bounded global FCFS or probe-based LFS.

    A rollout opens a session with its complete first-turn request catalog.
    Opening the session is what makes those requests schedulable; therefore a
    group-contiguous caller cannot accidentally fill the first wave before the
    other prompt groups are visible.
    """

    def __init__(
        self,
        dp_size: int,
        max_num_seqs_per_dp: int,
        mode: CrossDpMode,
        lookahead_per_dp: int = 0,
        global_admission_limit: int | None = None,
        dp_selection_mode: DpSelectionMode = "inflight_count",
        lfs_admission_fairness_interval: int = 0,
    ):
        validate_scheduler_config(
            dp_size=dp_size,
            max_num_seqs_per_dp=max_num_seqs_per_dp,
            mode=mode,
            lookahead_per_dp=lookahead_per_dp,
            dp_selection_mode=dp_selection_mode,
            lfs_admission_fairness_interval=lfs_admission_fairness_interval,
        )

        self.dp_size = dp_size
        self.max_num_seqs_per_dp = max_num_seqs_per_dp
        self.lookahead_per_dp = lookahead_per_dp
        self.admission_limit_per_dp = max_num_seqs_per_dp + lookahead_per_dp
        self.global_admission_limit = resolve_global_admission_limit(
            dp_size=dp_size,
            admission_limit_per_dp=self.admission_limit_per_dp,
            requested=global_admission_limit,
        )
        self.mode = mode
        self.dp_selection_mode = dp_selection_mode
        self.lfs_admission_fairness_interval = int(
            lfs_admission_fairness_interval
        )

        self.sessions: dict[str, Session] = {}
        self.requests: dict[str, Request] = {}
        self.dp_inflight: list[dict[str, int]] = [dict() for _ in range(dp_size)]
        self._inflight_by_group: dict[tuple[str, str], set[str]] = {}
        self._pending_request_count = 0
        self._pending_preferred_dp_request_count = 0
        self.assignment_history: list[dict[str, Any]] = []
        # Persistent across rollout sessions. history_lfs snapshots these means
        # when a new session opens; observations from the current session are
        # committed only when it closes, preventing current-step leakage/probes.
        self.group_history: dict[str, tuple[int, int]] = {}
        self.history_update_history: list[dict[str, Any]] = []
        self.estimate_rebase_history: list[dict[str, Any]] = []
        self._pending_fcfs: deque[str] = deque()
        self._pending_exact_length: list[tuple[int, int, str]] = []
        self._pending_by_group: dict[tuple[str, str], deque[str]] = {}
        # A scheduler-independent group index keeps diagnostics O(groups)
        # instead of rescanning every catalog request on every assignment.
        self._diagnostic_pending_by_group: dict[
            tuple[str, str], deque[str]
        ] = {}
        self._group_heap: list[
            tuple[tuple[int, int, float, int, int], int, tuple[str, str]]
        ] = []
        self._group_versions: dict[tuple[str, str], int] = {}
        self._new_assignment_events: deque[dict[str, Any]] = deque()
        self._cancel_tombstones: dict[str, None] = {}
        self._max_cancel_tombstones = 100000
        self._closed_session_ids: set[str] = set()
        self._closed_session_order: deque[str] = deque()
        self._max_closed_session_tombstones = 10000

        self._arrival_seq = 0
        self._assignment_seq = 0
        self._dp_assignment_ordinals = [0] * dp_size
        self._session_seq = 0
        self._dp_tiebreak = 0
        # Exact dispatcher-side high-water marks. Engine gauges are sampled
        # independently on each DP rank, so summing last-observed gauge values
        # cannot prove a global admission limit during a cross-DP handoff.
        # Every increase in dispatcher occupancy happens in _pump(), making
        # these cumulative watermarks a lossless capacity-safety witness.
        self.max_total_inflight_observed = 0
        self.max_inflight_observed_by_dp = [0] * dp_size
        self.ordinary_admission_opportunities = 0
        self.admission_fairness_due_count = 0
        self.admission_fairness_selected_count = 0
        self.admission_fairness_override_count = 0
        self.admission_fairness_noop_count = 0
        self.admission_fairness_no_candidate_count = 0
        self.fatal_error: str | None = None

    @property
    def total_capacity(self) -> int:
        return self.global_admission_limit

    def restore_group_history(self, history: dict[str, Any]) -> None:
        """Restore completed-session means before opening any new session."""
        if self.mode != "history_lfs":
            raise ValueError("group history can only be restored for history_lfs")
        if self.sessions or self.requests:
            raise RuntimeError("group history must be restored before opening a session")

        restored: dict[str, tuple[int, int]] = {}
        for group_id, value in history.items():
            if not isinstance(value, dict):
                raise ValueError(
                    f"group history entry {group_id!r} must be an object"
                )
            total = int(value["sum"])
            count = int(value["count"])
            if total < 0 or count <= 0:
                raise ValueError(
                    f"invalid group history entry {group_id!r}: "
                    f"sum={total}, count={count}"
                )
            restored[str(group_id)] = (total, count)
        self.group_history = restored

    def open_session(
        self,
        session_id: str,
        request_catalog: list[dict[str, Any]],
        participant_ids: list[str] | None = None,
    ) -> None:
        if self.fatal_error is not None:
            raise RuntimeError(self.fatal_error)
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if session_id in self.sessions:
            raise ValueError(f"session {session_id!r} already exists")
        if session_id in self._closed_session_ids:
            raise ValueError(f"session {session_id!r} was already closed")
        if participant_ids is None:
            participant_ids = [f"{session_id}:participant:0"]
        if not participant_ids or len(set(participant_ids)) != len(participant_ids):
            raise ValueError("participant_ids must be non-empty and unique")
        if not request_catalog:
            raise ValueError("request_catalog must contain at least one request")
        (
            dp_placement_mode,
            probe_selection_semantics,
            designated_probe_request_ids,
        ) = inspect_catalog(
            request_catalog, mode=self.mode, dp_size=self.dp_size
        )

        session = Session(
            session_id=session_id,
            arrival_seq=self._session_seq,
            open_participants=set(participant_ids),
            participant_requests={item: set() for item in participant_ids},
            dp_assignment_ordinals=[0] * self.dp_size,
            probe_selection_semantics=probe_selection_semantics,
            designated_probe_request_ids=designated_probe_request_ids,
            dp_placement_mode=dp_placement_mode,
        )
        if self.mode == "history_lfs":
            catalog_groups = {str(item["group_id"]) for item in request_catalog}
            session.estimates = {
                group_id: max(1, round(total / count))
                for group_id, (total, count) in self.group_history.items()
                if count > 0 and group_id in catalog_groups
            }
        elif self.mode == "oracle_probe_lfs":
            # Benchmark-only diagnostic: retain exact group maxima out of band.
            # They become visible to scheduling only after that group's probe
            # completes. Faster ordinary requests admitted by the unknown
            # round-robin fill cannot reveal the out-of-band value early.
            for item in request_catalog:
                group_id = str(item["group_id"])
                request_cost = int(item["oracle_cost"])
                if request_cost <= 0:
                    raise ValueError(
                        f"oracle_cost must be positive, got {request_cost}"
                    )
                session.oracle_estimates[group_id] = max(
                    request_cost, session.oracle_estimates.get(group_id, 0)
                )
        elif self.mode == "predicted_lfs":
            for item in request_catalog:
                group_id = str(item["group_id"])
                predicted_cost = int(item["predicted_cost"])
                if predicted_cost <= 0:
                    raise ValueError(
                        f"predicted_cost must be positive, got {predicted_cost}"
                    )
                previous = session.estimates.setdefault(group_id, predicted_cost)
                if previous != predicted_cost:
                    raise ValueError(
                        "predicted_cost must be identical within a prompt group: "
                        f"group={group_id!r}, {previous} != {predicted_cost}"
                    )
        self._session_seq += 1
        self.sessions[session_id] = session
        try:
            for item in request_catalog:
                self._add_request(
                    session=session,
                    participant_id=str(item.get("participant_id", participant_ids[0])),
                    request_id=str(item["request_id"]),
                    group_id=str(item["group_id"]),
                    fallback_cost=int(item["fallback_cost"]),
                    is_designated_probe=bool(
                        item.get("is_designated_probe", False)
                    ),
                    preferred_dp_idx=item.get("preferred_dp_idx"),
                )
            self._prioritize_designated_probes(session)
        except Exception:
            for request_id in session.request_ids:
                request = self.requests.pop(request_id, None)
                if request is not None and request.status == "pending":
                    self._decrement_pending_count(request)
            for group_key in [
                key
                for key in self._diagnostic_pending_by_group
                if key[0] == session_id
            ]:
                self._diagnostic_pending_by_group.pop(group_key, None)
                self._pending_by_group.pop(group_key, None)
                self._group_versions.pop(group_key, None)
            del self.sessions[session_id]
            raise

        self._pump()

    def acquire(
        self,
        session_id: str,
        participant_id: str,
        request_id: str,
        group_id: str,
        fallback_cost: int,
    ) -> dict[str, Any] | None:
        """Claim an assigned lease, or return ``None`` while it is pending."""
        self.prepare_acquire(
            session_id,
            participant_id,
            request_id,
            group_id,
            fallback_cost,
        )
        request = self.requests[request_id]
        if request.status == "assigned":
            return self._claim(request)
        if request.status == "pending":
            return None
        raise AssertionError(
            f"prepared request {request_id!r} has unexpected status "
            f"{request.status!r}"
        )

    def prepare_acquire(
        self,
        session_id: str,
        participant_id: str,
        request_id: str,
        group_id: str,
        fallback_cost: int,
    ) -> None:
        """Register and validate an acquire without claiming its lease.

        The Ray dispatcher uses this split phase so preassigned requests can be
        released to each DP worker strictly in assignment order.  Keeping the
        request in ``assigned`` (rather than ``started``) also lets participant
        close and client cancellation reclaim capacity until the target worker
        begins the causal vLLM frontend-submission path.
        """
        session = self._require_session(session_id)
        if self.fatal_error is not None:
            raise RuntimeError(self.fatal_error)
        if session.failed_error is not None:
            raise RuntimeError(session.failed_error)
        if participant_id not in session.open_participants:
            raise RuntimeError(
                f"participant {participant_id!r} is closed in session {session_id!r}"
            )

        request = self.requests.get(request_id)
        if request is None:
            # Later multi-turn requests are not known when the first-turn
            # catalog is opened.  They join the same prompt-group estimate.
            request = self._add_request(
                session=session,
                participant_id=participant_id,
                request_id=request_id,
                group_id=group_id,
                fallback_cost=fallback_cost,
            )
            self._pump()
        else:
            if request.session_id != session_id:
                raise ValueError(
                    f"request {request_id!r} belongs to session "
                    f"{request.session_id!r}, not {session_id!r}"
                )
            if request.participant_id != participant_id:
                raise ValueError(
                    f"request {request_id!r} participant mismatch: "
                    f"{request.participant_id!r} != {participant_id!r}"
                )
            if request.group_id != str(group_id):
                raise ValueError(
                    f"request {request_id!r} group mismatch: "
                    f"{request.group_id!r} != {group_id!r}"
                )

        if request.status in ("assigned", "pending"):
            return
        if request.status == "started":
            raise RuntimeError(f"request {request_id!r} already acquired its lease")
        raise RuntimeError(
            f"request {request_id!r} cannot acquire from status {request.status!r}"
        )

    def claim_if_assigned(self, request_id: str) -> dict[str, Any] | None:
        request = self.requests.get(request_id)
        if request is None or request.status != "assigned":
            return None
        return self._claim(request)

    def drain_new_assignment_events(self) -> list[dict[str, Any]]:
        events = list(self._new_assignment_events)
        self._new_assignment_events.clear()
        return events

    def complete(self, request_id: str, actual_length: int) -> None:
        request = self._require_request(request_id)
        if request.status == "completed":
            return
        if request.status != "started":
            raise RuntimeError(
                f"request {request_id!r} completed from invalid status "
                f"{request.status!r}"
            )
        if actual_length < 0:
            raise ValueError(f"actual_length must be non-negative, got {actual_length}")

        self._release_dp(request)
        request.status = "completed"
        session = self._require_session(request.session_id)
        if self.mode in ONLINE_LFS_MODES:
            previous_estimate = session.estimates.get(request.group_id)
            # Round-robin fill may place additional unknown members of a group
            # beside its probe.  A faster ordinary member must not reveal a
            # group estimate before the actual probe completes.  Once the
            # probe has established the first finite estimate, later
            # completions may continue to refine ordinary/predicted LFS.
            if (
                session.probe_selection_semantics
                == EXPLICIT_PROBE_SELECTION_SEMANTICS
            ):
                designated_request_id = (
                    session.designated_probe_request_ids.get(request.group_id)
                )
                designated_request = (
                    self.requests.get(designated_request_id)
                    if designated_request_id is not None
                    else None
                )
                is_observable_probe_completion = (
                    request.is_designated_probe
                    or (
                        # More than one unresolved group member can already be
                        # in flight when the designated probe definitively
                        # leaves.  The next such observation is the only
                        # available replacement even when it was originally
                        # admitted through the unknown round-robin tier.
                        request.unknown_admission
                        and designated_request is not None
                        and designated_request.status
                        in ("cancelled", "failed_terminal")
                    )
                )
            else:
                is_observable_probe_completion = request.probe_admission
            reveals_first_probe_estimate = (
                previous_estimate is not None
                or self.mode == "predicted_lfs"
                or is_observable_probe_completion
            )
            if reveals_first_probe_estimate:
                if self.mode == "oracle_probe_lfs":
                    session.estimates[request.group_id] = (
                        session.oracle_estimates[request.group_id]
                    )
                else:
                    session.estimates[request.group_id] = max(
                        actual_length,
                        session.estimates.get(request.group_id, 0),
                    )
                self._rebase_inflight_group_estimate(
                    session_id=request.session_id,
                    group_id=request.group_id,
                    estimate=session.estimates[request.group_id],
                    previous_estimate=previous_estimate,
                    completed_request_id=request_id,
                )
                self._refresh_group((request.session_id, request.group_id))
        elif self.mode == "history_lfs":
            session.completed_lengths.setdefault(request.group_id, []).append(
                actual_length
            )
        self._pump()
        self._maybe_drop_session(session)

    def cancel_unsubmitted(self, request_id: str) -> None:
        """Cancel a pending lease or one known not to have reached a worker."""
        request = self.requests.get(request_id)
        if request is None:
            self._cancel_tombstones[request_id] = None
            while len(self._cancel_tombstones) > self._max_cancel_tombstones:
                oldest = next(iter(self._cancel_tombstones))
                del self._cancel_tombstones[oldest]
            return
        if request.status in ("cancelled", "completed"):
            return
        if request.status in ("assigned", "started"):
            self._rollback_unknown_admission(request)
            self._release_dp(request)
        elif request.status == "pending":
            self._decrement_pending_count(request)
        else:
            raise RuntimeError(
                f"request {request_id!r} cannot be cancelled from "
                f"status {request.status!r}"
            )
        request.status = "cancelled"
        session = self._require_session(request.session_id)
        self._pump()
        self._maybe_drop_session(session)

    def fail_terminated(self, request_id: str, error: str) -> None:
        """Release a request after the worker RPC definitively terminated."""
        request = self._require_request(request_id)
        if request.status in ("failed_terminal", "completed"):
            return
        if request.status != "started":
            raise RuntimeError(
                f"submitted request {request_id!r} failed from invalid status "
                f"{request.status!r}"
            )
        session = self._require_session(request.session_id)
        self._rollback_unknown_admission(request)
        self._release_dp(request)
        request.status = "failed_terminal"
        self._pump()
        self._maybe_drop_session(session)

    def fail_unknown(self, request_id: str, error: str) -> None:
        """Fail-fast globally when the submitted RPC may still be running.

        We neither release nor reuse the uncertain DP slot.  More importantly,
        every subsequent open/acquire fails immediately instead of silently
        hanging or running with a permanently reduced capacity.
        """
        request = self._require_request(request_id)
        if request.status in ("failed_unknown", "completed"):
            return
        if request.status != "started":
            raise RuntimeError(
                f"submitted request {request_id!r} failed from invalid status "
                f"{request.status!r}"
            )
        request.status = "failed_unknown"
        self.fatal_error = (
            "cross-DP dispatcher remote state became unknown after request "
            f"{request_id!r}: {error}"
        )
        for other in self.requests.values():
            if other.status == "pending":
                self._decrement_pending_count(other)
                other.status = "cancelled"
            elif other.status == "assigned" and not other.lease_started:
                self._rollback_unknown_admission(other)
                self._release_dp(other)
                other.status = "cancelled"

    def close_participant(self, session_id: str, participant_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            if session_id in self._closed_session_ids:
                return
            raise KeyError(f"unknown cross-DP session {session_id!r}")
        if participant_id not in session.participant_requests:
            raise KeyError(
                f"unknown participant {participant_id!r} in cross-DP session "
                f"{session_id!r}"
            )
        if participant_id not in session.open_participants:
            return
        session.open_participants.remove(participant_id)

        # Release this participant's requests immediately. Waiting for the last
        # prompt thread can deadlock the remaining participants behind slots
        # reserved by a thread that already exited.
        for request_id in list(session.participant_requests[participant_id]):
            request = self.requests[request_id]
            if request.status == "pending":
                self._decrement_pending_count(request)
                request.status = "cancelled"
            elif request.status == "assigned" and not request.lease_started:
                self._rollback_unknown_admission(request)
                self._release_dp(request)
                request.status = "cancelled"

        if not session.open_participants:
            # Defensive cleanup for catalog entries whose owner never started.
            for request_id in list(session.request_ids):
                request = self.requests[request_id]
                if request.status == "pending":
                    self._decrement_pending_count(request)
                    request.status = "cancelled"
                elif request.status == "assigned" and not request.lease_started:
                    self._rollback_unknown_admission(request)
                    self._release_dp(request)
                    request.status = "cancelled"
        self._pump()
        self._maybe_drop_session(session)

    def snapshot(self) -> dict[str, Any]:
        """Publish the dispatcher-visible view of sessions, DPs and history."""
        return build_snapshot(self)

    def _add_request(
        self,
        session: Session,
        participant_id: str,
        request_id: str,
        group_id: str,
        fallback_cost: int,
        is_designated_probe: bool = False,
        preferred_dp_idx: int | None = None,
    ) -> Request:
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if request_id in self.requests:
            raise ValueError(f"request {request_id!r} already exists")
        if participant_id not in session.open_participants:
            raise ValueError(
                f"unknown participant {participant_id!r} for session "
                f"{session.session_id!r}"
            )
        if fallback_cost <= 0:
            raise ValueError(f"fallback_cost must be positive, got {fallback_cost}")
        if (
            session.dp_placement_mode == PREFERRED_DP_PINNED_PLACEMENT
            and preferred_dp_idx is None
        ):
            raise ValueError(
                "preferred-DP pinning is limited to the bounded first-turn "
                "request catalog; later requests without preferred_dp_idx "
                "cannot join a pinned session"
            )

        request = Request(
            session_id=session.session_id,
            participant_id=participant_id,
            request_id=request_id,
            group_id=str(group_id),
            arrival_seq=self._arrival_seq,
            fallback_cost=fallback_cost,
            is_designated_probe=is_designated_probe,
            preferred_dp_idx=preferred_dp_idx,
        )
        self._arrival_seq += 1
        self.requests[request_id] = request
        session.request_ids.add(request_id)
        session.participant_requests[participant_id].add(request_id)
        session.group_catalog_ordinals.setdefault(
            request.group_id, len(session.group_catalog_ordinals)
        )
        if request_id in self._cancel_tombstones:
            self._cancel_tombstones.pop(request_id)
            request.status = "cancelled"
        else:
            group_key = (session.session_id, request.group_id)
            self._pending_request_count += 1
            if request.preferred_dp_idx is not None:
                self._pending_preferred_dp_request_count += 1
            self._diagnostic_pending_by_group.setdefault(
                group_key, deque()
            ).append(request_id)
            if self.mode == "fcfs":
                self._pending_fcfs.append(request_id)
            elif self.mode == "exact_length_lpt":
                # Experiment-only exact-output-length LPT order. This is a
                # diagnostic reference for whole-request ordering, not a
                # makespan oracle: service time also depends on context and
                # batch state.
                heapq.heappush(
                    self._pending_exact_length,
                    (
                        -request.fallback_cost,
                        request.arrival_seq,
                        request_id,
                    ),
                )
            else:
                request_ids = self._pending_by_group.setdefault(
                    group_key, deque()
                )
                was_empty = not request_ids
                request_ids.append(request_id)
                if was_empty:
                    self._refresh_group(group_key)
        return request

    def _prioritize_designated_probes(self, session: Session) -> None:
        """Move explicit probes only within the LFS policy's group queues."""
        if (
            self.mode not in PROBE_LFS_MODES
            or session.probe_selection_semantics
            != EXPLICIT_PROBE_SELECTION_SEMANTICS
        ):
            return
        for group_id, request_id in session.designated_probe_request_ids.items():
            group_key = (session.session_id, group_id)
            pending = self._pending_by_group.get(group_key)
            if pending is None or request_id not in pending:
                # A pre-open cancellation tombstone can remove the designated
                # request. The first remaining request becomes an explicitly
                # observable replacement probe when capacity is assigned.
                self._refresh_group(group_key)
                continue
            pending.remove(request_id)
            pending.appendleft(request_id)
            self._refresh_group(group_key)

    def _pump(self) -> None:
        """Assign pending requests until capacity or the pending queue runs out."""
        if self.fatal_error is not None:
            return
        while self._admit_one():
            pass

    def _admit_one(self) -> bool:
        """Assign at most one request; return False when the loop must stop."""
        # _select_dp() advances the cyclic tie-break. Do not call it when there
        # is no request to assign: completion and participant-close paths both
        # pump after the last request, and those no-op pumps must not perturb
        # placement in the next session.
        if self._pending_request_count == 0:
            return False
        if (
            sum(len(inflight) for inflight in self.dp_inflight)
            >= self.global_admission_limit
        ):
            return False

        capacity = self._capacity_view()
        choice_diagnostics = pending_choice_diagnostics(self)
        selection = self._select_request_and_dp(capacity, choice_diagnostics)
        if selection is None:
            return False
        request, dp_idx, dp_placement_mode = selection

        session = self._require_session(request.session_id)
        tier, estimate, predicted_length = self._plan_admission(session, request)
        self._record_fairness(session, request, choice_diagnostics)
        self._commit_assignment(session, request, dp_idx, predicted_length)
        self._append_assignment_event(
            session,
            request,
            dp_idx=dp_idx,
            dp_placement_mode=dp_placement_mode,
            tier=tier,
            estimate=estimate,
            predicted_length=predicted_length,
            capacity=capacity,
            choice_diagnostics=choice_diagnostics,
        )
        return True

    def _capacity_view(self) -> dict[str, Any]:
        """Per-DP occupancy plus both one-step DP-selection counterfactuals.

        The counterfactuals are recorded, never acted on: they answer "would
        the other selector have chosen a different rank in this exact state?".
        """
        candidate_dp_indices = [
            candidate_dp_idx
            for candidate_dp_idx, inflight in enumerate(self.dp_inflight)
            if len(inflight) < self.admission_limit_per_dp
        ]
        static_cost_before_assignment = [
            sum(inflight.values()) for inflight in self.dp_inflight
        ]
        inflight_count_before_assignment = [
            len(inflight) for inflight in self.dp_inflight
        ]
        dp_tiebreak_before_assignment = self._dp_tiebreak
        selected_by_static_cost = (
            min(
                candidate_dp_indices,
                key=lambda candidate_dp_idx: (
                    static_cost_before_assignment[candidate_dp_idx],
                    inflight_count_before_assignment[candidate_dp_idx],
                    (
                        candidate_dp_idx
                        - dp_tiebreak_before_assignment
                    )
                    % self.dp_size,
                ),
            )
            if candidate_dp_indices
            else None
        )
        selected_by_inflight_count = (
            min(
                candidate_dp_indices,
                key=lambda candidate_dp_idx: (
                    len(self.dp_inflight[candidate_dp_idx]),
                    (
                        candidate_dp_idx
                        - dp_tiebreak_before_assignment
                    )
                    % self.dp_size,
                ),
            )
            if candidate_dp_indices
            else None
        )
        return {
            "candidate_dp_indices": candidate_dp_indices,
            "static_cost_before_assignment": static_cost_before_assignment,
            "inflight_count_before_assignment": inflight_count_before_assignment,
            "dp_tiebreak_before_assignment": dp_tiebreak_before_assignment,
            "selected_by_static_cost": selected_by_static_cost,
            "selected_by_inflight_count": selected_by_inflight_count,
        }

    def _select_request_and_dp(
        self, capacity: dict[str, Any], choice_diagnostics: dict[str, Any]
    ) -> tuple[Request, int, str] | None:
        """Pick the next request and its DP rank, or None to stop pumping."""
        candidate_dp_indices = capacity["candidate_dp_indices"]
        if (
            self.mode == "exact_length_lpt"
            and self._pending_preferred_dp_request_count > 0
        ):
            request = self._pop_exact_length_request_for_available_dp(
                candidate_dp_indices
            )
            if request is None:
                return
            if request.preferred_dp_idx is None:
                dp_idx = self._select_dp()
                if dp_idx is None:
                    raise AssertionError(
                        "an unpinned exact-length request was selected "
                        "without an available DP"
                    )
                dp_placement_mode = SCHEDULER_SELECTED_DP_PLACEMENT
            else:
                dp_idx = request.preferred_dp_idx
                if dp_idx not in candidate_dp_indices:
                    raise AssertionError(
                        "a pinned exact-length request was selected for "
                        f"unavailable DP {dp_idx}"
                    )
                self._dp_tiebreak = (dp_idx + 1) % self.dp_size
                dp_placement_mode = PREFERRED_DP_PINNED_PLACEMENT
        else:
            # Preserve the pre-pinning ordering and cyclic tie-break
            # behavior exactly when no pinned request is pending.
            dp_idx = self._select_dp()
            if dp_idx is None:
                return
            request = self._pop_request()
            if request is None:
                return
            dp_placement_mode = SCHEDULER_SELECTED_DP_PLACEMENT
        if (
            self.mode in ONLINE_LFS_MODES
            and choice_diagnostics["policy_expected_request_id"]
            != request.request_id
        ):
            raise AssertionError(
                "LFS heap selection disagrees with the independently "
                "reconstructed policy key: selected="
                f"{request.request_id!r}, expected="
                f"{choice_diagnostics['policy_expected_request_id']!r}"
            )

        return request, dp_idx, dp_placement_mode

    def _plan_admission(
        self, session: Session, request: Request
    ) -> tuple[str, int | None, int]:
        """Classify the admission tier and settle the static admission cost.

        Order matters: the tier reads ``probed_groups`` before the cost branch
        below inserts this group into it.
        """
        estimate = (
            session.estimates.get(request.group_id)
            if self.mode in (*ONLINE_LFS_MODES, "history_lfs")
            else None
        )
        estimate = (
            session.estimates.get(request.group_id)
            if self.mode in (*ONLINE_LFS_MODES, "history_lfs")
            else None
        )
        tier = "fcfs"
        if self.mode in ONLINE_LFS_MODES:
            if estimate is None and request.group_id not in session.probed_groups:
                tier = "probe"
            elif estimate is None:
                tier = "unknown_rr"
            else:
                tier = (
                    "oracle_probe_lfs"
                    if self.mode == "oracle_probe_lfs"
                    else self.mode
                )
        elif self.mode == "history_lfs":
            tier = "history_lfs" if estimate is not None else "cold_start_fcfs"
        elif self.mode == "exact_length_lpt":
            tier = "exact_length_lpt"

        if self.mode == "fcfs":
            predicted_length = request.fallback_cost
            request.probe_admission = False
            request.unknown_admission = False
        elif self.mode == "history_lfs":
            predicted_length = (
                request.fallback_cost if estimate is None else max(1, estimate)
            )
            request.probe_admission = False
            request.unknown_admission = False
        elif self.mode == "exact_length_lpt":
            predicted_length = request.fallback_cost
            request.probe_admission = False
            request.unknown_admission = False
        elif estimate is None:
            predicted_length = request.fallback_cost
            request.probe_admission = (
                request.group_id not in session.probed_groups
            )
            request.unknown_admission = True
            session.probed_groups.add(request.group_id)
            session.unknown_admissions[request.group_id] = (
                session.unknown_admissions.get(request.group_id, 0) + 1
            )
        else:
            predicted_length = max(1, estimate)
            request.probe_admission = False
            request.unknown_admission = False

        return tier, estimate, predicted_length

    def _record_fairness(
        self,
        session: Session,
        request: Request,
        choice_diagnostics: dict[str, Any],
    ) -> None:
        """Fold this admission opportunity into the fairness counters."""
        ordinary_admission_opportunity = bool(
            choice_diagnostics["ordinary_admission_opportunity"]
        )
        if ordinary_admission_opportunity:
            ordinary_admission_ordinal = int(
                choice_diagnostics["ordinary_admission_ordinal"]
            )
            if (
                ordinary_admission_ordinal
                != session.ordinary_admission_opportunities + 1
            ):
                raise AssertionError(
                    "ordinary admission ordinal disagrees with session "
                    "counter"
                )
            session.ordinary_admission_opportunities += 1
            self.ordinary_admission_opportunities += 1
            request.ordinary_admission_ordinal = (
                ordinary_admission_ordinal
            )
            if choice_diagnostics["admission_fairness_due"]:
                session.admission_fairness_due_count += 1
                self.admission_fairness_due_count += 1
                if choice_diagnostics["admission_fairness_selected"]:
                    session.admission_fairness_selected_count += 1
                    self.admission_fairness_selected_count += 1
                    if choice_diagnostics[
                        "admission_fairness_changed_base_choice"
                    ]:
                        session.admission_fairness_override_count += 1
                        self.admission_fairness_override_count += 1
                    else:
                        session.admission_fairness_noop_count += 1
                        self.admission_fairness_noop_count += 1
                else:
                    if not choice_diagnostics[
                        "admission_fairness_due_but_no_candidate"
                    ]:
                        raise AssertionError(
                            "due admission-fairness opportunity was "
                            "neither selected nor candidate-empty"
                        )
                    session.admission_fairness_no_candidate_count += 1
                    self.admission_fairness_no_candidate_count += 1
        else:
            request.ordinary_admission_ordinal = None

        request.admission_fairness_selected = bool(
            choice_diagnostics["admission_fairness_selected"]
        )
        request.admission_selection_reason = choice_diagnostics[
            "admission_selection_reason"
        ]
        if self.mode in ONLINE_LFS_MODES:
            session.last_group_admission_sequences[
                request.group_id
            ] = self._assignment_seq

        if self.mode in (*ONLINE_LFS_MODES, "history_lfs"):
            self._refresh_group((request.session_id, request.group_id))


    def _commit_assignment(
        self,
        session: Session,
        request: Request,
        dp_idx: int,
        predicted_length: int,
    ) -> None:
        """Move the request into ``assigned`` and charge it to its DP rank."""
        request.status = "assigned"
        request.dp_idx = dp_idx
        request.predicted_length = predicted_length
        request.assignment_sequence = self._assignment_seq
        request.dp_assignment_ordinal = self._dp_assignment_ordinals[dp_idx]
        request.session_dp_assignment_ordinal = (
            session.dp_assignment_ordinals[dp_idx]
        )
        self.dp_inflight[dp_idx][request.request_id] = predicted_length
        self._inflight_by_group.setdefault(
            (request.session_id, request.group_id), set()
        ).add(request.request_id)
        inflight_counts_after_assignment = [
            len(items) for items in self.dp_inflight
        ]
        self.max_total_inflight_observed = max(
            self.max_total_inflight_observed,
            sum(inflight_counts_after_assignment),
        )
        self.max_inflight_observed_by_dp = [
            max(previous, current)
            for previous, current in zip(
                self.max_inflight_observed_by_dp,
                inflight_counts_after_assignment,
                strict=True,
            )
        ]

    def _append_assignment_event(
        self,
        session: Session,
        request: Request,
        *,
        dp_idx: int,
        dp_placement_mode: str,
        tier: str,
        estimate: int | None,
        predicted_length: int,
        capacity: dict[str, Any],
        choice_diagnostics: dict[str, Any],
    ) -> None:
        """Record the full provenance of one assignment for downstream analysis."""
        candidate_dp_indices = capacity["candidate_dp_indices"]
        static_cost_before_assignment = capacity["static_cost_before_assignment"]
        inflight_count_before_assignment = capacity[
            "inflight_count_before_assignment"
        ]
        dp_tiebreak_before_assignment = capacity["dp_tiebreak_before_assignment"]
        selected_by_static_cost = capacity["selected_by_static_cost"]
        selected_by_inflight_count = capacity["selected_by_inflight_count"]
        inflight_counts_after_assignment = [
            len(items) for items in self.dp_inflight
        ]
        event = {
            "sequence": self._assignment_seq,
            "dp_assignment_ordinal": request.dp_assignment_ordinal,
            "session_dp_assignment_ordinal": (
                request.session_dp_assignment_ordinal
            ),
            "session_id": request.session_id,
            "request_id": request.request_id,
            "group_id": request.group_id,
            "dp_idx": dp_idx,
            "preferred_dp_idx": request.preferred_dp_idx,
            "dp_placement_mode": dp_placement_mode,
            "dp_selection_mode": self.dp_selection_mode,
            "tier": tier,
            "is_designated_probe": request.is_designated_probe,
            "probe_replacement": (
                tier == "probe"
                and request.group_id
                in session.designated_probe_request_ids
                and session.designated_probe_request_ids[
                    request.group_id
                ]
                != request.request_id
            ),
            "probe_selection_semantics": (
                session.probe_selection_semantics
            ),
            "estimate": estimate,
            "static_admission_cost": predicted_length,
            "predicted_length": predicted_length,
            "candidate_dp_count": len(candidate_dp_indices),
            "candidate_dp_indices": candidate_dp_indices,
            "dp_static_admission_cost_before_assignment": (
                static_cost_before_assignment
            ),
            "dp_inflight_count_before_assignment": (
                inflight_count_before_assignment
            ),
            "dp_tiebreak_before_assignment": (
                dp_tiebreak_before_assignment
            ),
            "dp_selected_by_static_admission_cost": (
                selected_by_static_cost
            ),
            "dp_selected_by_inflight_count": (
                selected_by_inflight_count
            ),
            "dp_selected_without_static_admission_cost": (
                selected_by_inflight_count
            ),
            "static_admission_cost_affected_dp_selection": (
                selected_by_static_cost != selected_by_inflight_count
            ),
            "dp_selector_applied": (
                dp_placement_mode == SCHEDULER_SELECTED_DP_PLACEMENT
            ),
            "preferred_dp_pin_honored": (
                dp_idx == request.preferred_dp_idx
                if dp_placement_mode
                == PREFERRED_DP_PINNED_PLACEMENT
                else None
            ),
            "dp_selection_matches_declared_mode": (
                dp_placement_mode
                == SCHEDULER_SELECTED_DP_PLACEMENT
                and dp_idx
                == (
                    selected_by_static_cost
                    if self.dp_selection_mode == "static_cost"
                    else selected_by_inflight_count
                )
            ),
            "dp_selection_counterfactual_semantics": (
                "Static-cost and inflight-count one-step counterfactuals "
                "hold the observed candidate set, in-flight state, and "
                "cyclic tie-break fixed; they do not model alternative "
                "scheduling trajectories."
            ),
            "dp_load_accounting_semantics": (
                STATIC_ADMISSION_COST_SEMANTICS
            ),
            **choice_diagnostics,
            "ordinary_admission_opportunity_count_after_assignment": (
                session.ordinary_admission_opportunities
            ),
            "admission_fairness_due_count_after_assignment": (
                session.admission_fairness_due_count
            ),
            "admission_fairness_selected_count_after_assignment": (
                session.admission_fairness_selected_count
            ),
            "admission_fairness_override_count_after_assignment": (
                session.admission_fairness_override_count
            ),
            "admission_fairness_noop_count_after_assignment": (
                session.admission_fairness_noop_count
            ),
            "admission_fairness_no_candidate_count_after_assignment": (
                session.admission_fairness_no_candidate_count
            ),
            "selected_differs_from_fcfs_front": (
                choice_diagnostics["fcfs_front_request_id"]
                != request.request_id
            ),
            "dp_inflight": inflight_counts_after_assignment,
        }
        self.assignment_history.append(event)
        self._new_assignment_events.append(event)
        self._assignment_seq += 1
        self._dp_assignment_ordinals[dp_idx] += 1
        session.dp_assignment_ordinals[dp_idx] += 1
        if len(self.assignment_history) > 12000:
            del self.assignment_history[:2000]

    def _pop_request(self) -> Request | None:
        if self.mode == "fcfs":
            while self._pending_fcfs:
                request_id = self._pending_fcfs.popleft()
                request = self.requests.get(request_id)
                if request is None or request.status != "pending":
                    continue
                session = self.sessions.get(request.session_id)
                if session is None or session.failed_error is not None:
                    continue
                self._decrement_pending_count(request)
                return request
            return None
        if self.mode == "exact_length_lpt":
            while self._pending_exact_length:
                _, _, request_id = heapq.heappop(self._pending_exact_length)
                request = self.requests.get(request_id)
                if request is None or request.status != "pending":
                    continue
                session = self.sessions.get(request.session_id)
                if session is None or session.failed_error is not None:
                    continue
                self._decrement_pending_count(request)
                return request
            return None

        fairness_group_key = self._fairness_group_key_if_due()
        if fairness_group_key is not None:
            request = self._group_front(fairness_group_key)
            if request is None:
                raise AssertionError(
                    "admission-fairness winner lost its pending group front"
                )
            self._pending_by_group[fairness_group_key].popleft()
            self._decrement_pending_count(request)
            return request

        while self._group_heap:
            key, version, group_key = heapq.heappop(self._group_heap)
            if self._group_versions.get(group_key) != version:
                continue
            request = self._group_front(group_key)
            if request is None:
                self._pending_by_group.pop(group_key, None)
                self._group_versions.pop(group_key, None)
                continue
            current_key = self._lfs_key(request)
            if current_key != key:
                self._refresh_group(group_key)
                continue
            self._pending_by_group[group_key].popleft()
            self._decrement_pending_count(request)
            return request
        return None

    def _pop_exact_length_request_for_available_dp(
        self, candidate_dp_indices: list[int]
    ) -> Request | None:
        """Pop the longest exact request whose pinned DP can accept it.

        A temporarily full pinned DP must not head-of-line block work assigned
        to another DP. Skipped heap entries are restored unchanged, preserving
        their exact-LPT priority as soon as their target DP has capacity.
        The scan is O(pending) per refill and is intentionally confined to the
        bounded, benchmark-only first-turn diagnostic described by
        ``PREFERRED_DP_PINNING_SEMANTICS``.
        """

        candidate_dp_set = set(candidate_dp_indices)
        skipped: list[tuple[int, int, str]] = []
        selected: Request | None = None
        while self._pending_exact_length:
            item = heapq.heappop(self._pending_exact_length)
            request = self.requests.get(item[2])
            if request is None or request.status != "pending":
                continue
            session = self.sessions.get(request.session_id)
            if session is None or session.failed_error is not None:
                continue
            if (
                request.preferred_dp_idx is not None
                and request.preferred_dp_idx not in candidate_dp_set
            ):
                skipped.append(item)
                continue
            self._decrement_pending_count(request)
            selected = request
            break
        for item in skipped:
            heapq.heappush(self._pending_exact_length, item)
        return selected

    def _fairness_group_key_if_due(self) -> tuple[str, str] | None:
        """Select from the policy queue without consulting diagnostic state."""

        if (
            self.lfs_admission_fairness_interval <= 0
            or self.mode not in ONLINE_LFS_MODES
        ):
            return None
        group_fronts: dict[tuple[str, str], Request] = {}
        for group_key in list(self._pending_by_group):
            request = self._group_front(group_key)
            if request is not None:
                group_fronts[group_key] = request
        if not group_fronts:
            return None

        oldest_session_arrival_seq = min(
            self.sessions[request.session_id].arrival_seq
            for request in group_fronts.values()
        )
        oldest_session_fronts = [
            request
            for request in group_fronts.values()
            if self.sessions[request.session_id].arrival_seq
            == oldest_session_arrival_seq
        ]
        active_tier = min(
            self._lfs_key(request)[1]
            for request in oldest_session_fronts
        )
        if active_tier != 2:
            return None
        session = self.sessions[oldest_session_fronts[0].session_id]
        next_ordinary_ordinal = (
            session.ordinary_admission_opportunities + 1
        )
        if (
            next_ordinary_ordinal
            % self.lfs_admission_fairness_interval
            != 0
        ):
            return None

        candidates = [
            request
            for request in oldest_session_fronts
            if self._lfs_key(request)[1] == 2
            and not self._inflight_by_group.get(
                (request.session_id, request.group_id)
            )
        ]
        if not candidates:
            return None
        winner = min(candidates, key=self._admission_fairness_key)
        return (winner.session_id, winner.group_id)

    def _admission_fairness_key(
        self, request: Request
    ) -> tuple[int, int, int]:
        session = self.sessions[request.session_id]
        return (
            session.last_group_admission_sequences.get(
                request.group_id, -1
            ),
            session.group_catalog_ordinals[request.group_id],
            request.arrival_seq,
        )

    def _decrement_pending_count(self, request: Request) -> None:
        if request.status != "pending":
            raise AssertionError(
                f"request {request.request_id!r} is not pending"
            )
        if self._pending_request_count <= 0:
            raise AssertionError("pending request count underflow")
        self._pending_request_count -= 1
        if request.preferred_dp_idx is not None:
            if self._pending_preferred_dp_request_count <= 0:
                raise AssertionError(
                    "pending preferred-DP request count underflow"
                )
            self._pending_preferred_dp_request_count -= 1

    def _lfs_key(
        self, request: Request
    ) -> tuple[int, int, float, int, int]:
        session = self.sessions[request.session_id]
        estimate = session.estimates.get(request.group_id)
        if self.mode == "history_lfs":
            # No current-session probe. Known prompts use the mean of completed
            # rollouts from earlier sessions; unseen prompts retain FCFS order.
            return (
                session.arrival_seq,
                0 if estimate is not None else 1,
                -float(estimate) if estimate is not None else 0.0,
                0,
                request.arrival_seq,
            )
        if estimate is None and request.group_id not in session.probed_groups:
            return (session.arrival_seq, 0, 0.0, 0, request.arrival_seq)

        # Algorithm 2 gives an unresolved group the numerical generation-length
        # upper bound, then orders all ordinary candidates by that estimate.
        # Keep the existing admission count as a stable round-robin tie-break
        # so G < capacity still fills across groups after the probe wave.
        effective_estimate = (
            request.fallback_cost
            if estimate is None
            else max(1, estimate)
        )
        return (
            session.arrival_seq,
            2,
            -float(effective_estimate),
            session.unknown_admissions.get(request.group_id, 0),
            request.arrival_seq,
        )

    def _group_front(self, group_key: tuple[str, str]) -> Request | None:
        request_ids = self._pending_by_group.get(group_key)
        if request_ids is None:
            return None
        while request_ids:
            request = self.requests.get(request_ids[0])
            if request is None or request.status != "pending":
                request_ids.popleft()
                continue
            session = self.sessions.get(request.session_id)
            if session is None or session.failed_error is not None:
                request_ids.popleft()
                continue
            return request
        return None

    def _refresh_group(self, group_key: tuple[str, str]) -> None:
        if self.mode not in (*ONLINE_LFS_MODES, "history_lfs"):
            return
        version = self._group_versions.get(group_key, 0) + 1
        self._group_versions[group_key] = version
        request = self._group_front(group_key)
        if request is not None:
            heapq.heappush(
                self._group_heap, (self._lfs_key(request), version, group_key)
            )

    def _select_dp(self) -> int | None:
        candidates = [
            dp_idx
            for dp_idx, inflight in enumerate(self.dp_inflight)
            if len(inflight) < self.admission_limit_per_dp
        ]
        if not candidates:
            return None

        def static_cost_key(dp_idx: int) -> tuple[int, int, int]:
            cyclic_distance = (dp_idx - self._dp_tiebreak) % self.dp_size
            return (
                sum(self.dp_inflight[dp_idx].values()),
                len(self.dp_inflight[dp_idx]),
                cyclic_distance,
            )

        def inflight_count_key(dp_idx: int) -> tuple[int, int]:
            cyclic_distance = (dp_idx - self._dp_tiebreak) % self.dp_size
            return (
                len(self.dp_inflight[dp_idx]),
                cyclic_distance,
            )

        if self.dp_selection_mode == "static_cost":
            selected = min(candidates, key=static_cost_key)
        else:
            # Causal control: length estimates and static admission costs are
            # deliberately absent from both the key and its tie-break.
            selected = min(candidates, key=inflight_count_key)
        self._dp_tiebreak = (selected + 1) % self.dp_size
        return selected

    def _claim(self, request: Request) -> dict[str, Any]:
        assert request.dp_idx is not None
        assert request.predicted_length is not None
        assert request.assignment_sequence is not None
        assert request.dp_assignment_ordinal is not None
        assert request.session_dp_assignment_ordinal is not None
        request.status = "started"
        request.lease_started = True
        return {
            "request_id": request.request_id,
            "dp_idx": request.dp_idx,
            "predicted_length": request.predicted_length,
            "assignment_sequence": request.assignment_sequence,
            "dp_assignment_ordinal": request.dp_assignment_ordinal,
            "session_dp_assignment_ordinal": (
                request.session_dp_assignment_ordinal
            ),
        }

    def _release_dp(self, request: Request) -> None:
        if request.dp_idx is None:
            return
        if request.request_id not in self.dp_inflight[request.dp_idx]:
            raise AssertionError(
                f"request {request.request_id!r} missing from DP "
                f"{request.dp_idx} inflight set"
            )
        del self.dp_inflight[request.dp_idx][request.request_id]
        group_key = (request.session_id, request.group_id)
        inflight_group = self._inflight_by_group.get(group_key)
        if (
            inflight_group is None
            or request.request_id not in inflight_group
        ):
            raise AssertionError(
                f"request {request.request_id!r} missing from in-flight "
                f"group index {group_key!r}"
            )
        inflight_group.remove(request.request_id)
        if not inflight_group:
            del self._inflight_by_group[group_key]
        request.dp_idx = None
        request.predicted_length = None

    def _rollback_unknown_admission(self, request: Request) -> None:
        if not request.unknown_admission:
            return
        session = self._require_session(request.session_id)
        if request.group_id not in session.estimates:
            count = session.unknown_admissions.get(request.group_id, 0)
            if count <= 0:
                raise AssertionError(
                    f"unknown admission count underflow for group {request.group_id!r}"
                )
            if count == 1:
                del session.unknown_admissions[request.group_id]
            else:
                session.unknown_admissions[request.group_id] = count - 1
            if count == 1:
                session.probed_groups.discard(request.group_id)
            self._refresh_group((request.session_id, request.group_id))
        request.unknown_admission = False
        request.probe_admission = False

    def _rebase_inflight_group_estimate(
        self,
        *,
        session_id: str,
        group_id: str,
        estimate: int,
        previous_estimate: int | None,
        completed_request_id: str,
    ) -> None:
        """Replace stale upper-bound costs after a group estimate arrives."""

        new_cost = max(1, int(estimate))
        group_key = (session_id, group_id)
        inflight_request_ids = sorted(
            self._inflight_by_group.get(group_key, set())
        )
        touched: list[dict[str, int | str | bool]] = []
        changed: list[dict[str, int | str | bool]] = []
        for request_id in inflight_request_ids:
            request = self._require_request(request_id)
            if (
                request.status not in ("assigned", "started")
                or request.dp_idx is None
            ):
                raise AssertionError(
                    f"in-flight group index contains non-live request "
                    f"{request_id!r} in status {request.status!r}"
                )
            old_cost = self.dp_inflight[request.dp_idx].get(
                request.request_id
            )
            if old_cost is None:
                raise AssertionError(
                    f"in-flight request {request.request_id!r} is missing "
                    f"from DP {request.dp_idx}"
                )
            self.dp_inflight[request.dp_idx][request.request_id] = new_cost
            request.predicted_length = new_cost
            item: dict[str, int | str | bool] = {
                "request_id": request.request_id,
                "dp_idx": request.dp_idx,
                "status_at_rebase": request.status,
                "old_static_admission_cost": old_cost,
                "new_static_admission_cost": new_cost,
                "cost_changed": old_cost != new_cost,
            }
            touched.append(item)
            if old_cost != new_cost:
                changed.append(item)
        if previous_estimate is None:
            estimate_transition = "first_finite_estimate"
        elif new_cost > int(previous_estimate):
            estimate_transition = "increased"
        elif new_cost < int(previous_estimate):
            estimate_transition = "decreased"
        else:
            estimate_transition = "unchanged"
        pending_request_ids = sorted(
            request_id
            for request_id in self._pending_by_group.get(
                group_key, ()
            )
            if (
                (request := self.requests.get(request_id)) is not None
                and request.status == "pending"
            )
        )
        completed_request = self._require_request(completed_request_id)
        session = self._require_session(session_id)
        self.estimate_rebase_history.append(
            {
                "session_id": session_id,
                "group_id": group_id,
                "estimate": new_cost,
                "previous_estimate": previous_estimate,
                "first_finite_estimate": previous_estimate is None,
                "estimate_transition": estimate_transition,
                "completed_request_id": completed_request_id,
                "completed_request_is_designated_probe": (
                    completed_request.is_designated_probe
                ),
                "completed_request_is_probe_admission": (
                    completed_request.probe_admission
                ),
                "completed_request_is_unknown_admission": (
                    completed_request.unknown_admission
                ),
                "probe_selection_semantics": (
                    session.probe_selection_semantics
                ),
                "designated_probe_request_id": (
                    session.designated_probe_request_ids.get(group_id)
                ),
                "designated_probe_request_status": (
                    self.requests[
                        session.designated_probe_request_ids[group_id]
                    ].status
                    if group_id in session.designated_probe_request_ids
                    else None
                ),
                "completed_request_was_admission_fairness_selected": (
                    completed_request.admission_fairness_selected
                ),
                "completed_request_ordinary_admission_ordinal": (
                    completed_request.ordinary_admission_ordinal
                ),
                "completed_request_admission_selection_reason": (
                    completed_request.admission_selection_reason
                ),
                "pending_request_ids": pending_request_ids,
                "inflight_request_ids": inflight_request_ids,
                "touched_request_count": len(touched),
                "changed_request_count": len(changed),
                "no_op_request_count": len(touched) - len(changed),
                "touched_requests": touched,
                "changed_requests": changed,
                "load_accounting_semantics": (
                    STATIC_ADMISSION_COST_SEMANTICS
                ),
            }
        )
        if len(self.estimate_rebase_history) > 12000:
            del self.estimate_rebase_history[:2000]

    def _maybe_drop_session(self, session: Session) -> None:
        if session.open_participants:
            return
        terminal = {"completed", "cancelled", "failed_terminal"}
        if any(
            self.requests[item].status not in terminal
            for item in session.request_ids
        ):
            return
        if self.mode == "history_lfs":
            for group_id, lengths in session.completed_lengths.items():
                if not lengths:
                    continue
                old_total, old_count = self.group_history.get(group_id, (0, 0))
                new_total = old_total + sum(lengths)
                new_count = old_count + len(lengths)
                self.group_history[group_id] = (new_total, new_count)
                self.history_update_history.append(
                    {
                        "session_id": session.session_id,
                        "group_id": group_id,
                        "prediction": (
                            old_total / old_count if old_count else None
                        ),
                        "current_group_mean": sum(lengths) / len(lengths),
                        "history_count_before": old_count,
                        "history_mean_after": new_total / new_count,
                    }
                )
            if len(self.history_update_history) > 12000:
                del self.history_update_history[:2000]
        for request_id in session.request_ids:
            del self.requests[request_id]
        for group_key in [
            key for key in self._pending_by_group if key[0] == session.session_id
        ]:
            self._pending_by_group.pop(group_key, None)
            self._group_versions.pop(group_key, None)
        for group_key in [
            key
            for key in self._diagnostic_pending_by_group
            if key[0] == session.session_id
        ]:
            self._diagnostic_pending_by_group.pop(group_key, None)
        if any(
            group_key[0] == session.session_id
            for group_key in self._inflight_by_group
        ):
            raise AssertionError(
                f"session {session.session_id!r} closed with indexed "
                "in-flight requests"
            )
        del self.sessions[session.session_id]
        self._closed_session_ids.add(session.session_id)
        self._closed_session_order.append(session.session_id)
        while len(self._closed_session_order) > self._max_closed_session_tombstones:
            expired = self._closed_session_order.popleft()
            self._closed_session_ids.discard(expired)

    def _require_session(self, session_id: str) -> Session:
        try:
            return self.sessions[session_id]
        except KeyError as error:
            raise KeyError(f"unknown cross-DP session {session_id!r}") from error

    def _require_request(self, request_id: str) -> Request:
        try:
            return self.requests[request_id]
        except KeyError as error:
            raise KeyError(f"unknown cross-DP request {request_id!r}") from error
