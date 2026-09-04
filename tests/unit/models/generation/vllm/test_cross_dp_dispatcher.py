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

import asyncio
import os
import threading
import time

import pytest
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_rl.models.generation.vllm.lfs.dispatcher import (
    CrossDpDispatcherActor,
)


async def await_ref(ref):
    return await ref


async def acquire(actor, request_id: str, group_id: str):
    return await actor.acquire.remote("s", "p", request_id, group_id, 128)


def confirm_frontend(actor, lease: dict) -> None:
    ray.get(
        actor.confirm_engine_frontend_submitted.remote(
            lease["request_id"],
            lease["assignment_sequence"],
            lease["dp_assignment_ordinal"],
            lease["session_dp_assignment_ordinal"],
        )
    )


@pytest.fixture
def local_ray():
    owned = not ray.is_initialized()
    if owned:
        ray.init(num_cpus=1, include_dashboard=False)
    yield
    if owned:
        ray.shutdown()


def test_dispatcher_handle_works_across_repeated_asyncio_run(local_ray) -> None:
    # This test compares dispatcher and client CLOCK_MONOTONIC timestamps.
    # Keep the actor in the same clock domain even when pytest is attached to
    # a multi-node Ray cluster, matching VllmGeneration's production setup.
    actor = CrossDpDispatcherActor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
    ).remote(1, 1, "lfs", False)
    catalog = [
        {"request_id": "r0", "group_id": "a", "fallback_cost": 128},
        {"request_id": "r1", "group_id": "b", "fallback_cost": 128},
    ]
    ray.get(actor.open_session.remote("s", catalog, ["p"]))

    first = asyncio.run(acquire(actor, "r0", "a"))
    assert first["dp_idx"] == 0
    snapshot = ray.get(actor.snapshot.remote())
    assert snapshot["assignment_history"][0]["assigned_at_monotonic_s"] > 0
    assert snapshot["assignment_history"][0]["assigned_at_unix_s"] > 0
    assert snapshot["assignment_history"][0]["dispatcher_hostname"]
    reported_at_unix = time.time()
    reported_at_monotonic = time.monotonic()
    asyncio.run(
        await_ref(
            actor.complete.remote(
                "r0",
                10,
                reported_at_unix,
                reported_at_monotonic,
                os.uname().nodename,
            )
        )
    )
    snapshot = ray.get(actor.snapshot.remote())
    refill = snapshot["assignment_history"][1]
    assert refill["trigger_completion_request_id"] == "r0"
    assert refill["client_completion_to_refill_assignment_s"] >= 0
    completion = snapshot["completion_history"][0]
    assert completion["request_id"] == "r0"
    assert completion["refill_assignment_sequences"] == [refill["sequence"]]
    assert completion["client_to_dispatcher_rpc_s"] >= 0
    second = asyncio.run(acquire(actor, "r1", "b"))
    assert second["dp_idx"] == 0
    asyncio.run(await_ref(actor.complete.remote("r1", 20)))
    asyncio.run(await_ref(actor.close_participant.remote("s", "p")))


def test_dispatcher_propagates_inflight_count_dp_selection_mode(
    local_ray,
) -> None:
    actor = CrossDpDispatcherActor.remote(
        2,
        2,
        "fcfs",
        False,
        None,
        0,
        3,
        "inflight_count",
    )
    catalog = [
        {"request_id": "long", "group_id": "a", "fallback_cost": 1000},
        {"request_id": "short-0", "group_id": "b", "fallback_cost": 1},
        {"request_id": "short-1", "group_id": "c", "fallback_cost": 1},
    ]
    ray.get(actor.open_session.remote("s", catalog, ["p"]))

    snapshot = ray.get(actor.snapshot.remote())
    assert snapshot["dp_selection_mode"] == "inflight_count"
    assert [
        event["dp_idx"] for event in snapshot["assignment_history"]
    ] == [0, 1, 0]

    ray.get(actor.close_participant.remote("s", "p"))


def test_dispatcher_propagates_lfs_admission_fairness_interval(
    local_ray,
) -> None:
    actor = CrossDpDispatcherActor.remote(
        1,
        2,
        "lfs",
        False,
        None,
        0,
        2,
        "inflight_count",
        1,
    )
    catalog = [
        {"request_id": "r0", "group_id": "a", "fallback_cost": 128},
        {"request_id": "r1", "group_id": "a", "fallback_cost": 128},
        {"request_id": "r2", "group_id": "a", "fallback_cost": 128},
    ]
    ray.get(actor.open_session.remote("s", catalog, ["p"]))

    first = ray.get(actor.acquire.remote("s", "p", "r0", "a", 128))
    confirm_frontend(actor, first)
    second = ray.get(actor.acquire.remote("s", "p", "r1", "a", 128))
    confirm_frontend(actor, second)
    ray.get(actor.complete.remote(first["request_id"], 10))
    third = ray.get(actor.acquire.remote("s", "p", "r2", "a", 128))
    confirm_frontend(actor, third)
    snapshot = ray.get(actor.snapshot.remote())

    assert snapshot["lfs_admission_fairness"]["interval"] == 1
    assert snapshot["assignment_history"][2]["admission_fairness_due"]
    assert snapshot["assignment_history"][2][
        "admission_fairness_due_but_no_candidate"
    ]

    ray.get(actor.complete.remote(second["request_id"], 20))
    ray.get(actor.complete.remote(third["request_id"], 30))
    ray.get(actor.close_participant.remote("s", "p"))


@pytest.mark.timeout(30)
def test_pending_acquire_can_span_another_thread_and_event_loop(local_ray) -> None:
    actor = CrossDpDispatcherActor.remote(1, 1, "lfs", False)
    catalog = [
        {"request_id": "r0", "group_id": "a", "fallback_cost": 128},
        {"request_id": "r1", "group_id": "b", "fallback_cost": 128},
    ]
    ray.get(actor.open_session.remote("s", catalog, ["p"]))
    asyncio.run(acquire(actor, "r0", "a"))

    result: dict = {}

    def wait_for_second_lease() -> None:
        result.update(asyncio.run(acquire(actor, "r1", "b")))

    thread = threading.Thread(target=wait_for_second_lease)
    thread.start()
    asyncio.run(await_ref(actor.complete.remote("r0", 10)))
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert result["dp_idx"] == 0
    asyncio.run(await_ref(actor.complete.remote("r1", 20)))
    asyncio.run(await_ref(actor.close_participant.remote("s", "p")))


@pytest.mark.timeout(30)
def test_per_dp_launch_gate_ignores_client_acquire_arrival_order(local_ray) -> None:
    actor = CrossDpDispatcherActor.remote(
        1,
        2,
        "fcfs",
        False,
        lookahead_per_dp=1,
    )
    catalog = [
        {"request_id": f"r{index}", "group_id": "g", "fallback_cost": 128}
        for index in range(3)
    ]
    ray.get(actor.open_session.remote("s", catalog, ["p"]))

    third_ref = actor.acquire.remote("s", "p", "r2", "g", 128)
    second_ref = actor.acquire.remote("s", "p", "r1", "g", 128)
    first_ref = actor.acquire.remote("s", "p", "r0", "g", 128)

    first = ray.get(first_ref)
    assert first["request_id"] == "r0"
    ready, _ = ray.wait([second_ref, third_ref], num_returns=1, timeout=0.2)
    assert ready == []

    confirm_frontend(actor, first)
    second = ray.get(second_ref)
    assert second["request_id"] == "r1"
    ready, _ = ray.wait([third_ref], num_returns=1, timeout=0.2)
    assert ready == []

    confirm_frontend(actor, second)
    third = ray.get(third_ref)
    assert third["request_id"] == "r2"
    confirm_frontend(actor, third)

    for request_id in ("r0", "r1", "r2"):
        ray.get(actor.complete.remote(request_id, 1))
    ray.get(actor.close_participant.remote("s", "p"))
    snapshot = ray.get(actor.snapshot.remote())
    assert [
        event["request_id"] for event in snapshot["engine_frontend_ack_history"]
    ] == ["r0", "r1", "r2"]
    assert all(
        event["source"] == "engine_frontend_submitted"
        for event in snapshot["engine_frontend_ack_history"]
    )
    assert snapshot["launch_queues_by_dp"] == [[]]
    assert snapshot["launch_outstanding_by_dp"] == [None]


@pytest.mark.timeout(30)
def test_invalid_or_duplicate_frontend_ack_does_not_release_gate(
    local_ray,
) -> None:
    actor = CrossDpDispatcherActor.remote(
        1,
        1,
        "fcfs",
        False,
        lookahead_per_dp=1,
    )
    catalog = [
        {"request_id": f"r{index}", "group_id": "g", "fallback_cost": 128}
        for index in range(2)
    ]
    ray.get(actor.open_session.remote("s", catalog, ["p"]))

    first = ray.get(actor.acquire.remote("s", "p", "r0", "g", 128))
    second_ref = actor.acquire.remote("s", "p", "r1", "g", 128)

    with pytest.raises(RuntimeError, match="assignment sequence mismatch"):
        ray.get(
            actor.confirm_engine_frontend_submitted.remote(
                first["request_id"],
                first["assignment_sequence"] + 1,
                first["dp_assignment_ordinal"],
                first["session_dp_assignment_ordinal"],
            )
        )
    snapshot = ray.get(actor.snapshot.remote())
    assert snapshot["engine_frontend_ack_history"] == []
    assert snapshot["launch_outstanding_by_dp"] == ["r0"]
    ready, _ = ray.wait([second_ref], num_returns=1, timeout=0.05)
    assert ready == []

    confirm_frontend(actor, first)
    second = ray.get(second_ref)
    with pytest.raises(RuntimeError, match="not the outstanding launch"):
        confirm_frontend(actor, first)
    snapshot = ray.get(actor.snapshot.remote())
    assert [
        event["request_id"]
        for event in snapshot["engine_frontend_ack_history"]
    ] == ["r0"]
    assert snapshot["launch_outstanding_by_dp"] == ["r1"]

    confirm_frontend(actor, second)
    ray.get(actor.complete.remote("r0", 1))
    ray.get(actor.complete.remote("r1", 1))
    ray.get(actor.close_participant.remote("s", "p"))


@pytest.mark.timeout(30)
def test_explicit_unknown_failure_retains_capacity_and_rejects_waiters(
    local_ray,
) -> None:
    actor = CrossDpDispatcherActor.remote(
        1,
        1,
        "fcfs",
        False,
        lookahead_per_dp=1,
    )
    catalog = [
        {"request_id": f"r{index}", "group_id": "g", "fallback_cost": 128}
        for index in range(2)
    ]
    ray.get(actor.open_session.remote("s", catalog, ["p"]))

    first = ray.get(actor.acquire.remote("s", "p", "r0", "g", 128))
    second_ref = actor.acquire.remote("s", "p", "r1", "g", 128)
    ray.get(
        actor.fail_unknown.remote(
            first["request_id"],
            "Ray transport failed with remote engine state unknown",
        )
    )

    with pytest.raises(RuntimeError, match="remote state became unknown"):
        ray.get(second_ref)
    snapshot = ray.get(actor.snapshot.remote())
    assert snapshot["fatal_error"] is not None
    assert snapshot["dp_inflight"] == [["r0"]]
    assert snapshot["launch_outstanding_by_dp"] == [None]
    assert snapshot["engine_frontend_ack_history"] == []


@pytest.mark.timeout(30)
def test_worker_proxy_creation_cannot_release_next_frontend_call(
    local_ray,
) -> None:
    """Model the measured proxy/worker race at the real engine boundary."""
    actor = CrossDpDispatcherActor.remote(
        1,
        2,
        "fcfs",
        False,
        lookahead_per_dp=1,
    )
    catalog = [
        {"request_id": f"r{index}", "group_id": "g", "fallback_cost": 128}
        for index in range(2)
    ]
    ray.get(actor.open_session.remote("s", catalog, ["p"]))

    first = ray.get(actor.acquire.remote("s", "p", "r0", "g", 128))
    second_ref = actor.acquire.remote("s", "p", "r1", "g", 128)

    # A driver-side worker proxy for r0 now exists, but its remote method is
    # deliberately delayed. Under the old acknowledgement boundary this would
    # release r1, whose remote call could reach the worker first.
    proxy_created = threading.Event()
    allow_frontend_submission = threading.Event()
    frontend_invocations: list[str] = []

    def delayed_first_worker_frontend() -> None:
        proxy_created.set()
        assert allow_frontend_submission.wait(timeout=5)
        frontend_invocations.append(first["request_id"])
        confirm_frontend(actor, first)

    first_worker = threading.Thread(target=delayed_first_worker_frontend)
    first_worker.start()
    assert proxy_created.wait(timeout=5)

    ready, _ = ray.wait([second_ref], num_returns=1, timeout=0.05)
    assert ready == []

    allow_frontend_submission.set()
    second = ray.get(second_ref)
    frontend_invocations.append(second["request_id"])
    confirm_frontend(actor, second)
    first_worker.join(timeout=5)

    assert not first_worker.is_alive()
    assert frontend_invocations == ["r0", "r1"]
    ray.get(actor.complete.remote("r0", 1))
    ray.get(actor.complete.remote("r1", 1))
    ray.get(actor.close_participant.remote("s", "p"))


@pytest.mark.timeout(30)
def test_launch_gates_are_independent_across_dp_ranks(local_ray) -> None:
    actor = CrossDpDispatcherActor.remote(
        2,
        1,
        "fcfs",
        False,
        lookahead_per_dp=1,
    )
    catalog = [
        {"request_id": f"r{index}", "group_id": "g", "fallback_cost": 128}
        for index in range(4)
    ]
    ray.get(actor.open_session.remote("s", catalog, ["p"]))

    refs = {
        request_id: actor.acquire.remote(
            "s", "p", request_id, "g", 128
        )
        for request_id in ("r3", "r2", "r1", "r0")
    }
    first_dp0, first_dp1 = ray.get([refs["r0"], refs["r1"]])
    assert (first_dp0["dp_idx"], first_dp1["dp_idx"]) == (0, 1)
    ready, _ = ray.wait(
        [refs["r2"], refs["r3"]], num_returns=1, timeout=0.2
    )
    assert ready == []

    confirm_frontend(actor, first_dp0)
    second_dp0 = ray.get(refs["r2"])
    assert second_dp0["dp_idx"] == 0
    ready, _ = ray.wait([refs["r3"]], num_returns=1, timeout=0.2)
    assert ready == []

    confirm_frontend(actor, first_dp1)
    second_dp1 = ray.get(refs["r3"])
    assert second_dp1["dp_idx"] == 1
    confirm_frontend(actor, second_dp0)
    confirm_frontend(actor, second_dp1)

    for request_id in ("r0", "r1", "r2", "r3"):
        ray.get(actor.complete.remote(request_id, 1))
    ray.get(actor.close_participant.remote("s", "p"))


@pytest.mark.timeout(30)
def test_cancelling_unsubmitted_launch_releases_next_gate_entry(local_ray) -> None:
    actor = CrossDpDispatcherActor.remote(
        1,
        1,
        "fcfs",
        False,
        lookahead_per_dp=1,
    )
    catalog = [
        {"request_id": f"r{index}", "group_id": "g", "fallback_cost": 128}
        for index in range(2)
    ]
    ray.get(actor.open_session.remote("s", catalog, ["p"]))

    first = ray.get(actor.acquire.remote("s", "p", "r0", "g", 128))
    second_ref = actor.acquire.remote("s", "p", "r1", "g", 128)
    ready, _ = ray.wait([second_ref], num_returns=1, timeout=0.2)
    assert ready == []

    ray.get(actor.cancel_unsubmitted.remote("r0"))
    second = ray.get(second_ref)
    assert second["request_id"] == "r1"
    confirm_frontend(actor, second)
    ray.get(actor.complete.remote("r1", 1))
    ray.get(actor.close_participant.remote("s", "p"))


@pytest.mark.timeout(30)
def test_frontend_failure_before_ack_releases_gate_without_deadlock(
    local_ray,
) -> None:
    actor = CrossDpDispatcherActor.remote(
        1,
        1,
        "fcfs",
        False,
        lookahead_per_dp=1,
    )
    catalog = [
        {"request_id": f"r{index}", "group_id": "g", "fallback_cost": 128}
        for index in range(2)
    ]
    ray.get(actor.open_session.remote("s", catalog, ["p"]))

    first = ray.get(actor.acquire.remote("s", "p", "r0", "g", 128))
    second_ref = actor.acquire.remote("s", "p", "r1", "g", 128)
    ready, _ = ray.wait([second_ref], num_returns=1, timeout=0.05)
    assert ready == []

    ray.get(
        actor.fail_terminated.remote(
            first["request_id"],
            "vLLM add_request failed before frontend acknowledgement",
        )
    )
    second = ray.get(second_ref)
    assert second["request_id"] == "r1"
    confirm_frontend(actor, second)
    ray.get(actor.complete.remote("r1", 1))
    ray.get(actor.close_participant.remote("s", "p"))

    snapshot = ray.get(actor.snapshot.remote())
    assert [
        (event["request_id"], event["source"])
        for event in snapshot["engine_frontend_ack_history"]
    ] == [
        ("r0", "terminated_failure"),
        ("r1", "engine_frontend_submitted"),
    ]
    assert snapshot["launch_outstanding_by_dp"] == [None]


@pytest.mark.timeout(30)
def test_participant_close_skips_unclaimed_launch_queue_head(local_ray) -> None:
    actor = CrossDpDispatcherActor.remote(
        1,
        1,
        "fcfs",
        False,
        lookahead_per_dp=1,
    )
    catalog = [
        {
            "request_id": "p0-r0",
            "group_id": "a",
            "participant_id": "p0",
            "fallback_cost": 128,
        },
        {
            "request_id": "p1-r0",
            "group_id": "b",
            "participant_id": "p1",
            "fallback_cost": 128,
        },
    ]
    ray.get(actor.open_session.remote("s", catalog, ["p0", "p1"]))
    second_ref = actor.acquire.remote("s", "p1", "p1-r0", "b", 128)
    ready, _ = ray.wait([second_ref], num_returns=1, timeout=0.2)
    assert ready == []

    ray.get(actor.close_participant.remote("s", "p0"))
    second = ray.get(second_ref)
    assert second["request_id"] == "p1-r0"
    confirm_frontend(actor, second)
    ray.get(actor.complete.remote("p1-r0", 1))
    ray.get(actor.close_participant.remote("s", "p1"))


@pytest.mark.timeout(30)
def test_participant_close_fail_fasts_unknown_frontend_state(local_ray) -> None:
    actor = CrossDpDispatcherActor.remote(
        1,
        1,
        "fcfs",
        False,
        lookahead_per_dp=1,
    )
    catalog = [
        {
            "request_id": "p0-r0",
            "group_id": "a",
            "participant_id": "p0",
            "fallback_cost": 128,
        },
        {
            "request_id": "p1-r0",
            "group_id": "b",
            "participant_id": "p1",
            "fallback_cost": 128,
        },
    ]
    ray.get(actor.open_session.remote("s", catalog, ["p0", "p1"]))
    first = ray.get(
        actor.acquire.remote("s", "p0", "p0-r0", "a", 128)
    )
    assert first["request_id"] == "p0-r0"
    second_ref = actor.acquire.remote("s", "p1", "p1-r0", "b", 128)
    ready, _ = ray.wait([second_ref], num_returns=1, timeout=0.2)
    assert ready == []

    # No engine-frontend ACK was observed for p0-r0, but add_request() may have
    # already returned while that ACK RPC is in flight. The slot must not be
    # released or reused.
    ray.get(actor.close_participant.remote("s", "p0"))
    with pytest.raises(RuntimeError, match="remote state became unknown"):
        ray.get(second_ref)
    snapshot = ray.get(actor.snapshot.remote())
    assert snapshot["fatal_error"] is not None
    assert snapshot["launch_outstanding_by_dp"] == [None]
    assert snapshot["dp_inflight"] == [["p0-r0"]]
    assert snapshot["engine_frontend_ack_history"] == []
