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

"""Ray concurrency boundary for the pure cross-DP scheduler state machine."""

import asyncio
import os
import time
from collections import deque
from typing import Any

import ray

from nemo_rl.models.generation.vllm.lfs.scheduler import (
    CrossDpMode,
    CrossDpSchedulerState,
    DpSelectionMode,
)


@ray.remote(num_cpus=0, max_concurrency=100000)
class CrossDpDispatcherActor:  # pragma: no cover - exercised in Ray integration tests
    """Share one scheduler safely across event loops, threads, and Ray actors."""

    def __init__(
        self,
        dp_size: int,
        max_num_seqs_per_dp: int,
        mode: CrossDpMode,
        trace: bool = False,
        initial_group_history: dict[str, Any] | None = None,
        lookahead_per_dp: int = 0,
        global_admission_limit: int | None = None,
        dp_selection_mode: DpSelectionMode = "inflight_count",
        lfs_admission_fairness_interval: int = 0,
    ) -> None:
        self.state = CrossDpSchedulerState(
            dp_size,
            max_num_seqs_per_dp,
            mode,
            lookahead_per_dp=lookahead_per_dp,
            global_admission_limit=global_admission_limit,
            dp_selection_mode=dp_selection_mode,
            lfs_admission_fairness_interval=(
                lfs_admission_fairness_interval
            ),
        )
        if initial_group_history:
            self.state.restore_group_history(initial_group_history)
        self.trace = trace
        self._waiters: dict[str, asyncio.Future] = {}
        # Leave ample async-actor call slots for complete/cancel/close RPCs.
        # Above this limit we fail explicitly instead of risking actor starvation.
        self._max_waiters = 90000
        self._completion_history: list[dict[str, Any]] = []
        # A scheduler assignment reserves capacity, but it is not yet an engine
        # submission. Release at most one lease at a time on each DP and wait
        # until that DP worker confirms that vLLM's add_request() completed
        # before releasing the next one. Creating a Ray worker proxy is not a
        # sufficient boundary: separately scheduled proxy calls can reach an
        # async worker out of order. Independent DP ranks still launch in
        # parallel because each DP owns a separate gate.
        self._launch_queues_by_dp: list[deque[str]] = [
            deque() for _ in range(dp_size)
        ]
        self._launch_outstanding_by_dp: list[str | None] = [None] * dp_size
        self._engine_frontend_ack_history: list[dict[str, Any]] = []

    async def open_session(
        self,
        session_id: str,
        request_catalog: list[dict[str, Any]],
        participant_ids: list[str],
    ) -> None:
        self.state.open_session(session_id, request_catalog, participant_ids)
        self._after_state_change()

    async def acquire(
        self,
        session_id: str,
        participant_id: str,
        request_id: str,
        group_id: str,
        fallback_cost: int,
    ) -> dict[str, Any]:
        self.state.prepare_acquire(
            session_id, participant_id, request_id, group_id, fallback_cost
        )
        self._after_state_change()

        if request_id in self._waiters:
            raise RuntimeError(f"request {request_id!r} already has a pending acquire")
        if len(self._waiters) >= self._max_waiters:
            self.state.cancel_unsubmitted(request_id)
            self._after_state_change()
            raise RuntimeError(
                "Cross-DP dispatcher waiter limit exceeded; split the rollout "
                f"or raise the actor limit (current={self._max_waiters})"
            )

        future = asyncio.get_running_loop().create_future()
        self._waiters[request_id] = future
        self._drain_launch_gates()
        try:
            return await future
        except asyncio.CancelledError:
            self.state.cancel_unsubmitted(request_id)
            self._clear_launch_outstanding(request_id)
            self._after_state_change()
            raise
        finally:
            self._waiters.pop(request_id, None)

    async def confirm_engine_frontend_submitted(
        self,
        request_id: str,
        assignment_sequence: int,
        dp_assignment_ordinal: int,
        session_dp_assignment_ordinal: int,
        client_reported_at_unix_s: float | None = None,
        client_reported_at_monotonic_s: float | None = None,
        client_hostname: str | None = None,
    ) -> None:
        """Acknowledge that vLLM accepted the leased request on its DP worker."""
        received_at_unix_s = time.time()
        received_at_monotonic_s = time.monotonic()
        request = self.state.requests.get(request_id)
        if request is None:
            raise KeyError(f"unknown cross-DP request {request_id!r}")
        if request.status != "started":
            raise RuntimeError(
                f"request {request_id!r} acknowledged engine submission from "
                f"invalid status {request.status!r}"
            )
        if request.assignment_sequence != int(assignment_sequence):
            raise RuntimeError(
                f"request {request_id!r} assignment sequence mismatch: "
                f"{request.assignment_sequence} != {assignment_sequence}"
            )
        if request.dp_assignment_ordinal != int(dp_assignment_ordinal):
            raise RuntimeError(
                f"request {request_id!r} DP assignment ordinal mismatch: "
                f"{request.dp_assignment_ordinal} != {dp_assignment_ordinal}"
            )
        if request.session_dp_assignment_ordinal != int(
            session_dp_assignment_ordinal
        ):
            raise RuntimeError(
                f"request {request_id!r} session DP assignment ordinal "
                f"mismatch: {request.session_dp_assignment_ordinal} != "
                f"{session_dp_assignment_ordinal}"
            )
        assert request.dp_idx is not None
        dp_idx = request.dp_idx
        if self._launch_outstanding_by_dp[dp_idx] != request_id:
            raise RuntimeError(
                f"request {request_id!r} is not the outstanding launch on "
                f"DP {dp_idx}: {self._launch_outstanding_by_dp[dp_idx]!r}"
            )

        self._record_engine_frontend_ack(
            request_id=request_id,
            dp_idx=dp_idx,
            assignment_sequence=int(assignment_sequence),
            dp_assignment_ordinal=int(dp_assignment_ordinal),
            session_dp_assignment_ordinal=int(session_dp_assignment_ordinal),
            source="engine_frontend_submitted",
            client_reported_at_unix_s=client_reported_at_unix_s,
            client_reported_at_monotonic_s=client_reported_at_monotonic_s,
            client_hostname=client_hostname,
            dispatcher_received_at_unix_s=received_at_unix_s,
            dispatcher_received_at_monotonic_s=received_at_monotonic_s,
        )
        self._launch_outstanding_by_dp[dp_idx] = None
        self._drain_launch_gates()

    async def complete(
        self,
        request_id: str,
        actual_length: int,
        client_reported_at_unix_s: float | None = None,
        client_reported_at_monotonic_s: float | None = None,
        client_hostname: str | None = None,
    ) -> None:
        received_at_unix_s = time.time()
        received_at_monotonic_s = time.monotonic()
        dispatcher_hostname = os.uname().nodename
        request = self.state.requests.get(request_id)
        session_id = request.session_id if request is not None else None
        dp_idx = request.dp_idx if request is not None else None
        self._ack_launch_from_terminal_transition(request_id, source="completion")
        self.state.complete(request_id, actual_length)
        events = self._after_state_change(
            trigger_completion={
                "request_id": request_id,
                "client_reported_at_unix_s": client_reported_at_unix_s,
                "client_reported_at_monotonic_s": (
                    client_reported_at_monotonic_s
                ),
                "client_hostname": client_hostname,
                "dispatcher_received_at_unix_s": received_at_unix_s,
                "dispatcher_received_at_monotonic_s": (
                    received_at_monotonic_s
                ),
            }
        )
        completed_at_unix_s = time.time()
        completed_at_monotonic_s = time.monotonic()
        self._completion_history.append(
            {
                "request_id": request_id,
                "session_id": session_id,
                "dp_idx": dp_idx,
                "actual_length": actual_length,
                "client_reported_at_unix_s": client_reported_at_unix_s,
                "client_reported_at_monotonic_s": (
                    client_reported_at_monotonic_s
                ),
                "client_hostname": client_hostname,
                "dispatcher_received_at_unix_s": received_at_unix_s,
                "dispatcher_received_at_monotonic_s": (
                    received_at_monotonic_s
                ),
                "dispatcher_completed_at_unix_s": completed_at_unix_s,
                "dispatcher_completed_at_monotonic_s": (
                    completed_at_monotonic_s
                ),
                "dispatcher_hostname": dispatcher_hostname,
                "client_to_dispatcher_rpc_s": (
                    received_at_monotonic_s
                    - client_reported_at_monotonic_s
                    if client_reported_at_monotonic_s is not None
                    and client_hostname == dispatcher_hostname
                    else None
                ),
                "client_to_dispatcher_rpc_wall_s": (
                    received_at_unix_s - client_reported_at_unix_s
                    if client_reported_at_unix_s is not None
                    else None
                ),
                "dispatcher_complete_service_s": (
                    completed_at_monotonic_s - received_at_monotonic_s
                ),
                "refill_assignment_sequences": [
                    int(event["sequence"]) for event in events
                ],
            }
        )
        if len(self._completion_history) > 12000:
            del self._completion_history[:2000]

    async def cancel_unsubmitted(self, request_id: str) -> None:
        self.state.cancel_unsubmitted(request_id)
        self._clear_launch_outstanding(request_id)
        self._after_state_change()
        self._reject_waiter(
            request_id, f"cross-DP request {request_id!r} was cancelled"
        )

    async def fail_terminated(self, request_id: str, error: str) -> None:
        self._ack_launch_from_terminal_transition(
            request_id, source="terminated_failure"
        )
        self.state.fail_terminated(request_id, error)
        self._after_state_change()

    async def fail_unknown(self, request_id: str, error: str) -> None:
        self.state.fail_unknown(request_id, error)
        self._clear_launch_outstanding(request_id)
        self._after_state_change()
        assert self.state.fatal_error is not None
        for waiting_request_id in list(self._waiters):
            self._reject_waiter(waiting_request_id, self.state.fatal_error)

    async def close_participant(
        self, session_id: str, participant_id: str
    ) -> None:
        session = self.state.sessions.get(session_id)
        request_ids = (
            list(session.participant_requests.get(participant_id, set()))
            if session is not None
            else []
        )
        # Once a lease has been returned, absence of the frontend ACK does not
        # prove that the request is still unsubmitted: add_request() may already
        # have returned on the worker while its ACK RPC is in flight. Releasing
        # such a slot could over-admit the engine. Participant close in this
        # window therefore makes remote state unknown and must fail fast while
        # retaining the scheduler capacity reservation.
        outstanding_request_ids = {
            request_id
            for request_id in self._launch_outstanding_by_dp
            if request_id is not None
        }
        unknown_request_ids = []
        for request_id in request_ids:
            if request_id not in outstanding_request_ids:
                continue
            request = self.state.requests.get(request_id)
            if request is None or request.status != "started":
                continue
            self.state.fail_unknown(
                request_id,
                "participant closed after lease return but before the "
                "engine-frontend acknowledgement was observed",
            )
            self._clear_launch_outstanding(request_id)
            unknown_request_ids.append(request_id)
        self.state.close_participant(session_id, participant_id)
        self._after_state_change()
        if unknown_request_ids:
            assert self.state.fatal_error is not None
            for waiting_request_id in list(self._waiters):
                self._reject_waiter(
                    waiting_request_id, self.state.fatal_error
                )
        for request_id in request_ids:
            request = self.state.requests.get(request_id)
            if request is None or request.status == "cancelled":
                self._reject_waiter(
                    request_id,
                    f"cross-DP request {request_id!r} was cancelled",
                )

    async def snapshot(self) -> dict[str, Any]:
        snapshot = self.state.snapshot()
        snapshot["completion_history"] = list(self._completion_history)
        snapshot["launch_queues_by_dp"] = [
            list(items) for items in self._launch_queues_by_dp
        ]
        snapshot["launch_outstanding_by_dp"] = list(
            self._launch_outstanding_by_dp
        )
        snapshot["engine_frontend_ack_history"] = list(
            self._engine_frontend_ack_history
        )
        # Keep the old key for result readers written before the acknowledgement
        # boundary moved from driver-side proxy creation to engine submission.
        # Events retain their truthful new source value.
        snapshot["worker_proxy_ack_history"] = list(
            self._engine_frontend_ack_history
        )
        return snapshot

    def _after_state_change(
        self,
        trigger_completion: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        events = self.state.drain_new_assignment_events()
        for event in events:
            # The state and actor share this event object.  Adding the timestamp
            # here therefore makes admission/drain timing visible in snapshots
            # without putting a clock dependency in the pure scheduler tests.
            event["assigned_at_monotonic_s"] = time.monotonic()
            # Ray may place this zero-CPU actor on a different host from the
            # benchmark driver.  A remote monotonic clock cannot safely be
            # subtracted from the driver's monotonic clock, so also expose a
            # common wall-clock timestamp for cross-process timeline alignment.
            event["assigned_at_unix_s"] = time.time()
            dispatcher_hostname = os.uname().nodename
            event["dispatcher_hostname"] = dispatcher_hostname
            if trigger_completion is not None:
                event["trigger_completion_request_id"] = trigger_completion[
                    "request_id"
                ]
                event["trigger_client_reported_at_unix_s"] = (
                    trigger_completion["client_reported_at_unix_s"]
                )
                event["trigger_client_reported_at_monotonic_s"] = (
                    trigger_completion["client_reported_at_monotonic_s"]
                )
                event["trigger_client_hostname"] = trigger_completion[
                    "client_hostname"
                ]
                event["trigger_dispatcher_received_at_unix_s"] = (
                    trigger_completion["dispatcher_received_at_unix_s"]
                )
                event["trigger_dispatcher_received_at_monotonic_s"] = (
                    trigger_completion["dispatcher_received_at_monotonic_s"]
                )
                if (
                    trigger_completion["client_reported_at_monotonic_s"]
                    is not None
                    and trigger_completion["client_hostname"]
                    == dispatcher_hostname
                ):
                    event["client_completion_to_refill_assignment_s"] = (
                        event["assigned_at_monotonic_s"]
                        - trigger_completion[
                            "client_reported_at_monotonic_s"
                        ]
                    )
                event["dispatcher_receive_to_refill_assignment_s"] = (
                    event["assigned_at_monotonic_s"]
                    - trigger_completion[
                        "dispatcher_received_at_monotonic_s"
                    ]
                )
            request_id = event["request_id"]
            dp_idx = int(event["dp_idx"])
            self._launch_queues_by_dp[dp_idx].append(request_id)
            if self.trace:
                self._trace_assignment(event)
        self._drain_launch_gates()
        return events

    def _drain_launch_gates(self) -> None:
        if self.state.fatal_error is not None:
            return
        terminal_statuses = {
            "cancelled",
            "completed",
            "failed_terminal",
            "failed_unknown",
        }
        for dp_idx, queue in enumerate(self._launch_queues_by_dp):
            if self._launch_outstanding_by_dp[dp_idx] is not None:
                continue
            while queue:
                request_id = queue[0]
                request = self.state.requests.get(request_id)
                if request is None or request.status in terminal_statuses:
                    queue.popleft()
                    continue
                if request.status == "pending":
                    raise AssertionError(
                        f"queued launch {request_id!r} on DP {dp_idx} reverted "
                        "to pending"
                    )
                if request.status == "started":
                    raise AssertionError(
                        f"queued launch {request_id!r} on DP {dp_idx} was "
                        "already claimed"
                    )
                if request.status != "assigned" or request.dp_idx != dp_idx:
                    raise AssertionError(
                        f"queued launch {request_id!r} has state "
                        f"status={request.status!r}, dp_idx={request.dp_idx!r}; "
                        f"expected assigned on DP {dp_idx}"
                    )

                future = self._waiters.get(request_id)
                if future is None or future.done():
                    break
                lease = self.state.claim_if_assigned(request_id)
                if lease is None:
                    raise AssertionError(
                        f"launch queue head {request_id!r} lost its assignment"
                    )
                queue.popleft()
                self._launch_outstanding_by_dp[dp_idx] = request_id
                future.set_result(lease)
                break

    def _clear_launch_outstanding(self, request_id: str) -> None:
        for dp_idx, outstanding in enumerate(self._launch_outstanding_by_dp):
            if outstanding == request_id:
                self._launch_outstanding_by_dp[dp_idx] = None
                return

    def _ack_launch_from_terminal_transition(
        self, request_id: str, *, source: str
    ) -> None:
        for dp_idx, outstanding in enumerate(self._launch_outstanding_by_dp):
            if outstanding != request_id:
                continue
            request = self.state.requests.get(request_id)
            if request is None:
                raise AssertionError(
                    f"outstanding launch {request_id!r} has no scheduler request"
                )
            assert request.assignment_sequence is not None
            assert request.dp_assignment_ordinal is not None
            assert request.session_dp_assignment_ordinal is not None
            now_unix_s = time.time()
            now_monotonic_s = time.monotonic()
            self._record_engine_frontend_ack(
                request_id=request_id,
                dp_idx=dp_idx,
                assignment_sequence=request.assignment_sequence,
                dp_assignment_ordinal=request.dp_assignment_ordinal,
                session_dp_assignment_ordinal=(
                    request.session_dp_assignment_ordinal
                ),
                source=source,
                client_reported_at_unix_s=None,
                client_reported_at_monotonic_s=None,
                client_hostname=None,
                dispatcher_received_at_unix_s=now_unix_s,
                dispatcher_received_at_monotonic_s=now_monotonic_s,
            )
            self._launch_outstanding_by_dp[dp_idx] = None
            self._drain_launch_gates()
            return

    def _record_engine_frontend_ack(
        self,
        *,
        request_id: str,
        dp_idx: int,
        assignment_sequence: int,
        dp_assignment_ordinal: int,
        session_dp_assignment_ordinal: int,
        source: str,
        client_reported_at_unix_s: float | None,
        client_reported_at_monotonic_s: float | None,
        client_hostname: str | None,
        dispatcher_received_at_unix_s: float,
        dispatcher_received_at_monotonic_s: float,
    ) -> None:
        request = self.state.requests.get(request_id)
        self._engine_frontend_ack_history.append(
            {
                "request_id": request_id,
                "session_id": (
                    request.session_id if request is not None else None
                ),
                "dp_idx": dp_idx,
                "assignment_sequence": assignment_sequence,
                "dp_assignment_ordinal": dp_assignment_ordinal,
                "session_dp_assignment_ordinal": (
                    session_dp_assignment_ordinal
                ),
                "source": source,
                "client_reported_at_unix_s": client_reported_at_unix_s,
                "client_reported_at_monotonic_s": (
                    client_reported_at_monotonic_s
                ),
                "client_hostname": client_hostname,
                "dispatcher_received_at_unix_s": dispatcher_received_at_unix_s,
                "dispatcher_received_at_monotonic_s": (
                    dispatcher_received_at_monotonic_s
                ),
                "dispatcher_hostname": os.uname().nodename,
                "client_to_dispatcher_rpc_s": (
                    dispatcher_received_at_monotonic_s
                    - client_reported_at_monotonic_s
                    if client_reported_at_monotonic_s is not None
                    and client_hostname == os.uname().nodename
                    else None
                ),
            }
        )
        if len(self._engine_frontend_ack_history) > 12000:
            del self._engine_frontend_ack_history[:2000]

    def _reject_waiter(self, request_id: str, error: str) -> None:
        future = self._waiters.get(request_id)
        if future is not None and not future.done():
            future.set_exception(RuntimeError(error))

    @staticmethod
    def _trace_assignment(event: dict[str, Any]) -> None:
        print(
            "[CROSS-DP-SCHED] "
            f"t={time.monotonic():.6f} seq={event['sequence']} "
            f"session={event['session_id']} request={event['request_id']} "
            f"group={event['group_id']} tier={event['tier']} "
            f"estimate={event['estimate']} dp={event['dp_idx']} "
            f"inflight={event['dp_inflight']}",
            flush=True,
        )
