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

import pytest

from nemo_rl.models.generation.vllm.lfs import (
    LFS_ADMISSION_FAIRNESS_SEMANTICS,
    CrossDpSchedulerState,
    build_async_cross_dp_group_ids,
)


class _Batch(dict):
    def __init__(self, *, size: int | None = None, **values) -> None:
        super().__init__(values)
        self._size = size

    @property
    def size(self) -> int:
        if self._size is not None:
            return self._size
        return len(next(iter(self.values())))


def test_async_cross_dp_group_ids_use_stable_dataset_idx() -> None:
    first_batch = _Batch(idx=[91, 17])
    second_batch = _Batch(idx=[17, 91])

    assert build_async_cross_dp_group_ids(
        first_batch, num_generations=3
    ) == ["91", "91", "91", "17", "17", "17"]
    assert build_async_cross_dp_group_ids(
        second_batch, num_generations=3
    ) == ["17", "17", "17", "91", "91", "91"]


def test_async_cross_dp_group_ids_fallback_and_validation() -> None:
    assert build_async_cross_dp_group_ids(
        _Batch(size=2), num_generations=2
    ) == ["0", "0", "1", "1"]
    with pytest.raises(ValueError, match="positive"):
        build_async_cross_dp_group_ids(_Batch(idx=[1]), num_generations=0)
    with pytest.raises(ValueError, match="idx count"):
        build_async_cross_dp_group_ids(
            _Batch(size=1, idx=[1, 2]), num_generations=1
        )


def catalog(groups: list[str], fallback_cost: int = 128) -> list[dict]:
    return [
        {
            "request_id": f"r{index}",
            "group_id": group,
            "fallback_cost": fallback_cost,
        }
        for index, group in enumerate(groups)
    ]


def designated_catalog(
    groups: list[str],
    designated_indices: set[int],
    fallback_cost: int = 128,
) -> list[dict]:
    return [
        {
            "request_id": f"r{index}",
            "group_id": group,
            "fallback_cost": fallback_cost,
            "is_designated_probe": index in designated_indices,
            "oracle_cost": (index + 1) * 10,
        }
        for index, group in enumerate(groups)
    ]


def acquire_assigned(state: CrossDpSchedulerState, request_id: str, group: str):
    lease = state.acquire("s", "s:participant:0", request_id, group, 128)
    assert lease is not None
    return lease


def test_predicted_lfs_orders_long_groups_from_session_open() -> None:
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=2,
        mode="predicted_lfs",
    )
    predicted_catalog = catalog(["short", "short", "long", "long"])
    for item in predicted_catalog:
        item["predicted_cost"] = (
            100 if item["group_id"] == "long" else 10
        )
    state.open_session("s", predicted_catalog)

    assert [
        (event["group_id"], event["tier"], event["estimate"])
        for event in state.assignment_history
    ] == [
        ("long", "predicted_lfs", 100),
        ("long", "predicted_lfs", 100),
    ]


def test_predicted_lfs_rejects_inconsistent_group_predictions() -> None:
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=1,
        mode="predicted_lfs",
    )
    predicted_catalog = catalog(["same", "same"])
    predicted_catalog[0]["predicted_cost"] = 10
    predicted_catalog[1]["predicted_cost"] = 11
    with pytest.raises(ValueError, match="identical within a prompt group"):
        state.open_session("s", predicted_catalog)


def test_prepare_acquire_preserves_assignment_until_ordered_claim() -> None:
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=2,
        mode="fcfs",
    )
    state.open_session("s", catalog(["a", "b"]))

    state.prepare_acquire(
        "s",
        "s:participant:0",
        "r1",
        "b",
        128,
    )
    assert state.requests["r1"].status == "assigned"
    assert not state.requests["r1"].lease_started

    lease = state.claim_if_assigned("r1")
    assert lease is not None
    event = state.assignment_history[1]
    assert lease["assignment_sequence"] == event["sequence"] == 1
    assert lease["dp_assignment_ordinal"] == event["dp_assignment_ordinal"] == 1
    assert (
        lease["session_dp_assignment_ordinal"]
        == event["session_dp_assignment_ordinal"]
        == 1
    )


def test_session_dp_assignment_ordinal_resets_but_global_ordinal_does_not() -> None:
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=1,
        mode="fcfs",
    )
    state.open_session("s", catalog(["a"]))
    acquire_assigned(state, "r0", "a")
    state.complete("r0", 1)
    state.close_participant("s", "s:participant:0")

    state.open_session(
        "next",
        [{"request_id": "next-r0", "group_id": "a", "fallback_cost": 128}],
    )
    event = state.assignment_history[-1]
    assert event["dp_assignment_ordinal"] == 1
    assert event["session_dp_assignment_ordinal"] == 0


def test_first_wave_probes_groups_then_round_robins_spare_capacity() -> None:
    state = CrossDpSchedulerState(dp_size=2, max_num_seqs_per_dp=4, mode="lfs")
    # Catalog arrival is deliberately group-contiguous, as repeat_interleave(G)
    # presents it in GRPO.
    groups = ["a"] * 3 + ["b"] * 3 + ["c"] * 3
    state.open_session("s", catalog(groups))

    first_wave = state.assignment_history[:8]
    assert [event["group_id"] for event in first_wave] == [
        "a",
        "b",
        "c",
        "a",
        "b",
        "c",
        "a",
        "b",
    ]
    assert [event["tier"] for event in first_wave[:3]] == [
        "probe",
        "probe",
        "probe",
    ]
    assert max(event["dp_inflight"][0] for event in first_wave) <= 4
    assert max(event["dp_inflight"][1] for event in first_wave) <= 4


def test_unresolved_upper_bound_ties_finite_estimate_numerically() -> None:
    state = CrossDpSchedulerState(dp_size=1, max_num_seqs_per_dp=2, mode="lfs")
    state.open_session("s", catalog(["a", "a", "b", "b"], fallback_cost=128))

    # Both groups receive their probe before any ordinary request.
    assert [
        (event["request_id"], event["tier"])
        for event in state.assignment_history
    ] == [("r0", "probe"), ("r2", "probe")]
    acquire_assigned(state, "r0", "a")
    acquire_assigned(state, "r2", "b")

    # Group a's finite estimate equals b's unresolved max_tokens upper bound.
    # They therefore tie in approximate-LFS and stable arrival order selects a;
    # unresolved b must not win merely because it used to occupy a lower tier.
    state.complete("r0", 128)

    event = state.assignment_history[-1]
    assert event["request_id"] == "r1"
    assert event["group_id"] == "a"
    assert event["tier"] == "lfs"
    assert event["estimate"] == 128
    assert event["pending_unknown_group_count_before_assignment"] == 1
    assert event["pending_finite_group_count_before_assignment"] == 1
    assert event["policy_active_priority_tier"] == 2
    assert {
        item["group_id"]
        for item in event["policy_eligible_pending_groups"]
    } == {"a", "b"}
    assert event["policy_priority_class_count_before_assignment"] == 1
    assert event["policy_expected_request_id"] == "r1"
    assert event["policy_expected_group_id"] == "a"
    assert event["policy_expected_priority_key"] == [
        0,
        2,
        -128.0,
        1,
        1,
    ]
    assert {
        item["group_id"]: item["policy_priority_key"]
        for item in event["policy_eligible_pending_groups"]
    } == {
        "a": [0, 2, -128.0, 1, 1],
        "b": [0, 2, -128.0, 1, 3],
    }


@pytest.mark.parametrize("mode", ["lfs", "oracle_probe_lfs"])
def test_explicit_designated_probes_only_reorder_lfs_group_queues(
    mode: str,
) -> None:
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=4,
        mode=mode,
    )
    state.open_session(
        "s",
        designated_catalog(
            ["a", "a", "a", "b", "b", "b"],
            designated_indices={2, 5},
        ),
    )

    events = state.assignment_history
    assert [event["request_id"] for event in events] == [
        "r2",
        "r5",
        "r0",
        "r3",
    ]
    assert [event["tier"] for event in events] == [
        "probe",
        "probe",
        "unknown_rr",
        "unknown_rr",
    ]
    assert [event["is_designated_probe"] for event in events] == [
        True,
        True,
        False,
        False,
    ]
    assert all(not event["probe_replacement"] for event in events)
    assert {
        event["probe_selection_semantics"] for event in events
    } == {"explicit_catalog_flag-v1"}

    # Explicit selection changes only the LFS group deque. Catalog arrival
    # order and the independent FCFS diagnostic remain untouched.
    assert [state.requests[f"r{index}"].arrival_seq for index in range(6)] == list(
        range(6)
    )
    assert events[0]["fcfs_front_request_id"] == "r0"
    assert events[0]["policy_expected_request_id"] == "r2"
    assert events[1]["policy_expected_request_id"] == "r5"
    assert events[0]["selected_differs_from_fcfs_front"]


def test_disabled_lfs_admission_fairness_has_no_candidates_or_actions() -> None:
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=2,
        mode="lfs",
        lfs_admission_fairness_interval=0,
    )
    state.open_session("s", catalog(["a", "a", "b", "b"]))
    acquire_assigned(state, "r0", "a")
    acquire_assigned(state, "r2", "b")
    state.complete("r0", 10)
    state.complete("r2", 100)

    assert all(
        event["admission_fairness_eligible_pending_groups"] == []
        and event["admission_fairness_candidate_count"] == 0
        and not event["admission_fairness_due"]
        and not event["admission_fairness_selected"]
        for event in state.assignment_history
    )
    assert state.snapshot()["lfs_admission_fairness"] == {
        "policy": "prose-inspired-idle-group-admission-age-v1",
        "semantics": LFS_ADMISSION_FAIRNESS_SEMANTICS,
        "interval": 0,
        "ordinary_admission_opportunity_count": 2,
        "due_count": 0,
        "selected_count": 0,
        "override_count": 0,
        "noop_count": 0,
        "no_candidate_count": 0,
    }


def test_lfs_admission_fairness_rescues_underestimated_idle_group() -> None:
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=2,
        mode="lfs",
        lfs_admission_fairness_interval=1,
    )
    state.open_session("s", catalog(["a", "a", "a", "b", "b"]))
    acquire_assigned(state, "r0", "a")
    acquire_assigned(state, "r3", "b")

    # Base LFS prefers unresolved group b at the numerical upper bound.
    # The due safeguard instead gives idle, underestimated group a another
    # sample without changing capacity or DP placement.
    state.complete("r0", 10)
    event = state.assignment_history[-1]
    assert event["request_id"] == "r1"
    assert event["base_policy_expected_request_id"] == "r4"
    assert event["admission_fairness_expected_request_id"] == "r1"
    assert event["admission_fairness_selected"]
    assert event["admission_fairness_changed_base_choice"]
    assert event["admission_selection_reason"] == "fairness_override"
    assert event["ordinary_admission_ordinal"] == 1
    assert event["admission_fairness_candidate_count"] == 1

    acquire_assigned(state, "r1", "a")
    state.complete("r1", 200)
    assert state.sessions["s"].estimates["a"] == 200
    rebase = state.estimate_rebase_history[-1]
    assert rebase["previous_estimate"] == 10
    assert rebase["estimate"] == 200
    assert rebase["completed_request_was_admission_fairness_selected"]
    assert rebase["completed_request_ordinary_admission_ordinal"] == 1


def test_lfs_admission_fairness_rotates_only_across_idle_groups() -> None:
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=3,
        mode="lfs",
        lfs_admission_fairness_interval=1,
    )
    state.open_session(
        "s",
        catalog(["a", "a", "b", "b", "c", "c"]),
    )
    for request_id, group in (("r0", "a"), ("r2", "b"), ("r4", "c")):
        acquire_assigned(state, request_id, group)

    state.complete("r0", 10)
    state.complete("r2", 20)
    state.complete("r4", 30)
    ordinary = [
        event
        for event in state.assignment_history
        if event["ordinary_admission_opportunity"]
    ]
    assert [event["group_id"] for event in ordinary] == ["a", "b", "c"]
    assert [
        event["ordinary_admission_ordinal"] for event in ordinary
    ] == [1, 2, 3]
    assert all(event["admission_fairness_selected"] for event in ordinary)


def test_lfs_admission_fairness_due_without_idle_group_falls_back() -> None:
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=2,
        mode="lfs",
        lfs_admission_fairness_interval=1,
    )
    state.open_session("s", catalog(["a", "a", "a"]))
    acquire_assigned(state, "r0", "a")
    acquire_assigned(state, "r1", "a")
    state.complete("r0", 10)

    event = state.assignment_history[-1]
    assert event["request_id"] == "r2"
    assert event["admission_fairness_due"]
    assert not event["admission_fairness_selected"]
    assert event["admission_fairness_due_but_no_candidate"]
    assert event["admission_selection_reason"] == (
        "fairness_no_candidate_fallback"
    )
    assert event["admission_fairness_candidate_count"] == 0


@pytest.mark.parametrize(
    ("mode", "interval"),
    [
        ("fcfs", 1),
        ("history_lfs", 1),
        ("exact_length_lpt", 1),
        ("lfs", -1),
        ("lfs", True),
    ],
)
def test_lfs_admission_fairness_rejects_invalid_configuration(
    mode: str,
    interval: int,
) -> None:
    with pytest.raises(ValueError, match="lfs_admission_fairness_interval"):
        CrossDpSchedulerState(
            dp_size=1,
            max_num_seqs_per_dp=1,
            mode=mode,  # type: ignore[arg-type]
            lfs_admission_fairness_interval=interval,
        )


def test_explicit_designated_probes_do_not_reorder_fcfs() -> None:
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=6,
        mode="fcfs",
    )
    state.open_session(
        "s",
        designated_catalog(
            ["a", "a", "a", "b", "b", "b"],
            designated_indices={2, 5},
        ),
    )

    events = state.assignment_history
    assert [event["request_id"] for event in events] == [
        f"r{index}" for index in range(6)
    ]
    assert [event["is_designated_probe"] for event in events] == [
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    assert all(not event["probe_replacement"] for event in events)
    assert {
        event["probe_selection_semantics"] for event in events
    } == {"explicit_catalog_flag-v1"}


def test_legacy_catalog_keeps_implicit_probe_selection_semantics() -> None:
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=1,
        mode="lfs",
    )
    state.open_session("s", catalog(["a", "a"]))

    event = state.assignment_history[0]
    assert event["request_id"] == "r0"
    assert not event["is_designated_probe"]
    assert not event["probe_replacement"]
    assert (
        event["probe_selection_semantics"]
        == "implicit_first_pending-v1"
    )


@pytest.mark.parametrize(
    ("request_catalog", "message"),
    [
        (
            [
                {
                    "request_id": "r0",
                    "group_id": "a",
                    "fallback_cost": 128,
                    "is_designated_probe": True,
                },
                {
                    "request_id": "r1",
                    "group_id": "a",
                    "fallback_cost": 128,
                },
            ],
            "must be present on every catalog request",
        ),
        (
            [
                {
                    "request_id": "r0",
                    "group_id": "a",
                    "fallback_cost": 128,
                    "is_designated_probe": 1,
                },
                {
                    "request_id": "r1",
                    "group_id": "a",
                    "fallback_cost": 128,
                    "is_designated_probe": False,
                },
            ],
            "must be a bool",
        ),
        (
            designated_catalog(["a", "a"], designated_indices=set()),
            "exactly one",
        ),
        (
            designated_catalog(["a", "a"], designated_indices={0, 1}),
            "exactly one",
        ),
    ],
)
def test_explicit_designated_probe_catalog_validation(
    request_catalog: list[dict],
    message: str,
) -> None:
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=1,
        mode="lfs",
    )

    with pytest.raises(ValueError, match=message):
        state.open_session("s", request_catalog)


def test_probe_catalog_spans_multiple_waves_when_capacity_is_small() -> None:
    state = CrossDpSchedulerState(dp_size=1, max_num_seqs_per_dp=2, mode="lfs")
    groups = ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2
    state.open_session("s", catalog(groups))

    assert [event["group_id"] for event in state.assignment_history] == ["a", "b"]
    acquire_assigned(state, "r0", "a")
    acquire_assigned(state, "r2", "b")
    state.complete("r0", 100)
    state.complete("r2", 10)

    # Newly learned estimates must not bypass groups that have not yet been
    # probed, even when the probe catalog spans more than one capacity wave.
    assert [event["group_id"] for event in state.assignment_history[:4]] == [
        "a",
        "b",
        "c",
        "d",
    ]
    assert [event["tier"] for event in state.assignment_history[:4]] == [
        "probe",
        "probe",
        "probe",
        "probe",
    ]


def test_explicit_empty_participant_list_is_rejected() -> None:
    state = CrossDpSchedulerState(dp_size=1, max_num_seqs_per_dp=1, mode="lfs")
    with pytest.raises(ValueError, match="participant_ids must be non-empty"):
        state.open_session("s", catalog(["a"]), [])


def test_probe_results_drive_global_lfs_across_dp_ranks() -> None:
    state = CrossDpSchedulerState(dp_size=3, max_num_seqs_per_dp=1, mode="lfs")
    groups = ["a"] * 3 + ["b"] * 3 + ["c"] * 3
    state.open_session("s", catalog(groups))

    # One probe from each group is assigned across all three DP ranks.
    assert [event["group_id"] for event in state.assignment_history[:3]] == [
        "a",
        "b",
        "c",
    ]
    assert [event["dp_idx"] for event in state.assignment_history[:3]] == [0, 1, 2]

    acquire_assigned(state, "r0", "a")
    acquire_assigned(state, "r3", "b")
    acquire_assigned(state, "r6", "c")
    state.complete("r0", 10)
    state.complete("r3", 100)
    state.complete("r6", 50)

    # While probes are unresolved, spare slots remain in the unknown RR tier.
    # Once all estimates exist, the newly freed slot goes to longest group b.
    assert state.assignment_history[5]["group_id"] == "b"
    assert state.assignment_history[5]["tier"] == "lfs"
    assert state.assignment_history[5]["estimate"] == 100
    assert state.assignment_history[5][
        "pending_finite_group_count_before_assignment"
    ] == 3
    assert state.assignment_history[5]["pending_finite_estimate_min"] == 10
    assert state.assignment_history[5]["pending_finite_estimate_max"] == 100
    assert state.assignment_history[5]["policy_eligible_session_ids"] == [
        "s"
    ]
    assert state.assignment_history[5]["policy_active_priority_tier"] == 2
    assert state.assignment_history[5][
        "policy_eligible_pending_finite_group_count"
    ] == 3
    assert state.assignment_history[5][
        "policy_eligible_pending_finite_estimate_min"
    ] == 10
    assert state.assignment_history[5][
        "policy_eligible_pending_finite_estimate_max"
    ] == 100
    assert state.assignment_history[5][
        "policy_priority_class_count_before_assignment"
    ] == 3
    assert state.assignment_history[5]["fcfs_front_request_id"] == "r1"
    assert state.assignment_history[5]["selected_differs_from_fcfs_front"]
    assert state.estimate_rebase_history[0]["pending_request_ids"] == [
        "r1",
        "r2",
    ]


def test_completed_probe_rebases_stale_inflight_group_upper_bounds() -> None:
    state = CrossDpSchedulerState(
        dp_size=2,
        max_num_seqs_per_dp=2,
        mode="lfs",
    )
    state.open_session("s", catalog(["a", "a", "b", "b"]))
    acquire_assigned(state, "r0", "a")

    assert state.requests["r1"].predicted_length == 128
    r1_dp = state.requests["r1"].dp_idx
    assert r1_dp is not None
    assert state.dp_inflight[r1_dp]["r1"] == 128

    state.complete("r0", 10)

    assert state.requests["r1"].predicted_length == 10
    assert state.dp_inflight[r1_dp]["r1"] == 10
    rebase = state.estimate_rebase_history[-1]
    assert rebase["group_id"] == "a"
    assert rebase["estimate"] == 10
    assert rebase["first_finite_estimate"]
    assert rebase["previous_estimate"] is None
    assert rebase["completed_request_id"] == "r0"
    assert rebase["pending_request_ids"] == []
    assert rebase["inflight_request_ids"] == ["r1"]
    assert rebase["touched_request_count"] == 1
    assert rebase["changed_request_count"] == 1
    assert rebase["no_op_request_count"] == 0
    assert rebase["changed_requests"] == [{
        "request_id": "r1",
        "dp_idx": r1_dp,
        "status_at_rebase": "assigned",
        "old_static_admission_cost": 128,
        "new_static_admission_cost": 10,
        "cost_changed": True,
    }]


@pytest.mark.parametrize("mode", ["lfs", "oracle_probe_lfs"])
def test_unknown_rr_completion_cannot_reveal_before_designated_probe(
    mode: str,
) -> None:
    request_catalog = [
        {
            "request_id": "ordinary",
            "group_id": "a",
            "fallback_cost": 128,
            "is_designated_probe": False,
            **({"oracle_cost": 10} if mode == "oracle_probe_lfs" else {}),
        },
        {
            "request_id": "probe",
            "group_id": "a",
            "fallback_cost": 128,
            "is_designated_probe": True,
            **({"oracle_cost": 100} if mode == "oracle_probe_lfs" else {}),
        },
        {
            "request_id": "other-probe",
            "group_id": "b",
            "fallback_cost": 128,
            "is_designated_probe": True,
            **({"oracle_cost": 50} if mode == "oracle_probe_lfs" else {}),
        },
    ]
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=3,
        mode=mode,
    )
    state.open_session("s", request_catalog)

    assert [event["request_id"] for event in state.assignment_history] == [
        "probe",
        "other-probe",
        "ordinary",
    ]
    assert state.requests["probe"].probe_admission is True
    assert state.requests["ordinary"].probe_admission is False
    acquire_assigned(state, "probe", "a")
    acquire_assigned(state, "other-probe", "b")
    acquire_assigned(state, "ordinary", "a")

    state.complete("ordinary", 10)
    assert "a" not in state.sessions["s"].estimates
    assert not any(
        item["completed_request_id"] == "ordinary"
        for item in state.estimate_rebase_history
    )

    state.complete("probe", 20)
    expected = 100 if mode == "oracle_probe_lfs" else 20
    assert state.sessions["s"].estimates["a"] == expected
    assert state.estimate_rebase_history[-1]["completed_request_id"] == "probe"
    assert state.estimate_rebase_history[-1]["first_finite_estimate"]
    assert state.estimate_rebase_history[-1][
        "completed_request_is_designated_probe"
    ]
    assert state.estimate_rebase_history[-1][
        "completed_request_is_probe_admission"
    ]
    assert (
        state.estimate_rebase_history[-1]["designated_probe_request_id"]
        == "probe"
    )


def test_rebase_distinguishes_unchanged_cost_refresh_from_change() -> None:
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=3,
        mode="lfs",
    )
    state.open_session("s", catalog(["a", "a", "a"]))
    for request_id in ("r0", "r1", "r2"):
        acquire_assigned(state, request_id, "a")

    state.complete("r0", 10)
    state.complete("r1", 5)

    rebase = state.estimate_rebase_history[-1]
    assert rebase["estimate_transition"] == "unchanged"
    assert rebase["touched_request_count"] == 1
    assert rebase["changed_request_count"] == 0
    assert rebase["no_op_request_count"] == 1
    assert not rebase["touched_requests"][0]["cost_changed"]


def test_completion_refills_only_the_freed_dp_without_migration() -> None:
    state = CrossDpSchedulerState(dp_size=2, max_num_seqs_per_dp=1, mode="fcfs")
    state.open_session("s", catalog(["a", "b", "c"]))
    first = acquire_assigned(state, "r0", "a")
    second = acquire_assigned(state, "r1", "b")
    assert (first["dp_idx"], second["dp_idx"]) == (0, 1)

    state.complete("r0", 20)
    third = acquire_assigned(state, "r2", "c")
    assert third["dp_idx"] == 0
    assert state.requests["r1"].dp_idx == 1
    refill = state.assignment_history[-1]
    assert refill["candidate_dp_count"] == 1
    assert refill["candidate_dp_indices"] == [0]
    assert refill["dp_static_admission_cost_before_assignment"] == [
        0,
        128,
    ]
    assert refill["candidate_dp_count"] == 1
    assert not refill["static_admission_cost_affected_dp_selection"]


def test_dispatcher_lookahead_prequeues_engine_waiting_requests() -> None:
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=2,
        mode="fcfs",
        lookahead_per_dp=1,
    )
    state.open_session("s", catalog(["a", "b", "c", "d"]))

    assert [event["request_id"] for event in state.assignment_history] == [
        "r0",
        "r1",
        "r2",
    ]
    snapshot = state.snapshot()
    assert snapshot["max_num_seqs_per_dp"] == 2
    assert snapshot["lookahead_per_dp"] == 1
    assert snapshot["admission_limit_per_dp"] == 3
    assert snapshot["max_total_inflight_observed"] == 3
    assert snapshot["max_inflight_observed_by_dp"] == [3]
    assert snapshot["pending"] == 1
    assert snapshot["pending_counter_matches_request_states"]
    assert snapshot["inflight_group_index_matches_dp_inflight"]


def test_global_window_preserves_cross_dp_choice_below_per_dp_caps() -> None:
    state = CrossDpSchedulerState(
        dp_size=2,
        max_num_seqs_per_dp=2,
        mode="fcfs",
        global_admission_limit=3,
        dp_selection_mode="static_cost",
    )
    request_catalog = [
        {"request_id": "r0", "group_id": "a", "fallback_cost": 100},
        {"request_id": "r1", "group_id": "b", "fallback_cost": 1},
        {"request_id": "r2", "group_id": "c", "fallback_cost": 1},
        {"request_id": "r3", "group_id": "d", "fallback_cost": 1},
    ]
    state.open_session("s", request_catalog)

    assert [event["dp_idx"] for event in state.assignment_history] == [0, 1, 1]
    snapshot = state.snapshot()
    assert snapshot["global_admission_limit"] == 3
    assert snapshot["max_total_inflight_observed"] == 3
    assert snapshot["max_inflight_observed_by_dp"] == [1, 2]
    assert snapshot["pending"] == 1

    assert state.claim_if_assigned("r1") is not None
    state.complete("r1", 1)

    refill = state.assignment_history[-1]
    assert refill["request_id"] == "r3"
    assert refill["candidate_dp_indices"] == [0, 1]
    assert refill["dp_static_admission_cost_before_assignment"] == [100, 1]
    assert refill["dp_inflight_count_before_assignment"] == [1, 1]
    assert refill["dp_selected_without_static_admission_cost"] == 0
    assert refill["dp_idx"] == 1
    assert refill["static_admission_cost_affected_dp_selection"]
    assert sum(refill["dp_inflight"]) == 3


def test_default_dp_selection_mode_is_length_independent() -> None:
    state = CrossDpSchedulerState(
        dp_size=2,
        max_num_seqs_per_dp=2,
        mode="fcfs",
        global_admission_limit=3,
    )
    state.open_session(
        "s",
        [
            {"request_id": "long", "group_id": "a", "fallback_cost": 100},
            {"request_id": "short-0", "group_id": "b", "fallback_cost": 1},
            {"request_id": "short-1", "group_id": "c", "fallback_cost": 1},
        ],
    )

    assert state.snapshot()["dp_selection_mode"] == "inflight_count"
    assert [event["dp_idx"] for event in state.assignment_history] == [0, 1, 0]
    assert all(
        event["dp_idx"] == event["dp_selected_by_inflight_count"]
        for event in state.assignment_history
    )
    assert state.assignment_history[-1][
        "static_admission_cost_affected_dp_selection"
    ]


def test_inflight_count_dp_selection_is_independent_of_length_costs() -> None:
    def placements(costs: list[int]) -> tuple[list[int], bool]:
        state = CrossDpSchedulerState(
            dp_size=2,
            max_num_seqs_per_dp=2,
            mode="fcfs",
            global_admission_limit=3,
            dp_selection_mode="inflight_count",
        )
        state.open_session(
            "s",
            [
                {
                    "request_id": f"r{index}",
                    "group_id": str(index),
                    "fallback_cost": cost,
                }
                for index, cost in enumerate(costs)
            ],
        )
        assert state.snapshot()["dp_selection_mode"] == "inflight_count"
        assert all(
            event["dp_selection_mode"] == "inflight_count"
            and event["dp_selection_matches_declared_mode"]
            and event["dp_idx"]
            == event["dp_selected_by_inflight_count"]
            and event["dp_selected_without_static_admission_cost"]
            == event["dp_selected_by_inflight_count"]
            for event in state.assignment_history
        )
        final_event = state.assignment_history[-1]
        return (
            [event["dp_idx"] for event in state.assignment_history],
            bool(
                final_event[
                    "static_admission_cost_affected_dp_selection"
                ]
            ),
        )

    first_placements, first_counterfactual_differs = placements(
        [1000, 1, 999]
    )
    second_placements, second_counterfactual_differs = placements(
        [1, 1000, 2]
    )
    assert first_placements == second_placements == [0, 1, 0]
    assert first_counterfactual_differs
    assert not second_counterfactual_differs


def test_empty_pumps_do_not_advance_dp_tiebreak_across_sessions() -> None:
    state = CrossDpSchedulerState(
        dp_size=3,
        max_num_seqs_per_dp=1,
        mode="fcfs",
        global_admission_limit=1,
    )
    state.open_session(
        "warmup",
        [
            {
                "request_id": "warmup-r0",
                "group_id": "warmup-group",
                "fallback_cost": 128,
            }
        ],
    )
    warmup_lease = state.claim_if_assigned("warmup-r0")
    assert warmup_lease is not None
    assert warmup_lease["dp_idx"] == 0

    # Both calls pump with no pending request. Neither is an assignment and
    # therefore neither may consume the next cyclic DP tie-break.
    state.complete("warmup-r0", 1)
    state.close_participant("warmup", "warmup:participant:0")

    state.open_session(
        "measured",
        [
            {
                "request_id": "measured-r0",
                "group_id": "measured-group",
                "fallback_cost": 128,
            }
        ],
    )
    measured_event = state.assignment_history[-1]
    assert measured_event["dp_tiebreak_before_assignment"] == 1
    assert measured_event["dp_idx"] == 1


def test_invalid_dp_selection_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="dp_selection_mode"):
        CrossDpSchedulerState(
            dp_size=2,
            max_num_seqs_per_dp=2,
            mode="fcfs",
            dp_selection_mode="length_magic",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("completed_request_id", ["r0", "r1"])
def test_half_aggregate_window_keeps_both_dp_ranks_eligible_after_completion(
    completed_request_id: str,
) -> None:
    state = CrossDpSchedulerState(
        dp_size=2,
        max_num_seqs_per_dp=2,
        mode="fcfs",
        global_admission_limit=2,
    )
    state.open_session("s", catalog(["a", "b", "c"]))

    assert [event["dp_idx"] for event in state.assignment_history] == [0, 1]
    assert state.claim_if_assigned(completed_request_id) is not None
    state.complete(completed_request_id, 1)

    assert state.assignment_history[-1]["candidate_dp_indices"] == [0, 1]


@pytest.mark.parametrize("global_limit", [0, 5])
def test_global_window_rejects_invalid_capacity(global_limit: int) -> None:
    with pytest.raises(ValueError, match="global_admission_limit"):
        CrossDpSchedulerState(
            dp_size=2,
            max_num_seqs_per_dp=2,
            mode="fcfs",
            global_admission_limit=global_limit,
        )


def test_unsubmitted_cancellation_releases_capacity() -> None:
    state = CrossDpSchedulerState(dp_size=1, max_num_seqs_per_dp=1, mode="lfs")
    state.open_session("s", catalog(["a", "b"]))
    first = acquire_assigned(state, "r0", "a")
    assert first["dp_idx"] == 0

    state.cancel_unsubmitted("r0")
    second = acquire_assigned(state, "r1", "b")
    assert second["dp_idx"] == 0


def test_cancelled_probe_is_retried_as_a_probe() -> None:
    state = CrossDpSchedulerState(dp_size=1, max_num_seqs_per_dp=1, mode="lfs")
    state.open_session("s", catalog(["a", "a", "b"]))
    acquire_assigned(state, "r0", "a")

    state.cancel_unsubmitted("r0")

    assert state.assignment_history[-1]["request_id"] == "r1"
    assert state.assignment_history[-1]["group_id"] == "a"
    assert state.assignment_history[-1]["tier"] == "probe"


def test_cancelled_designated_probe_records_replacement_probe() -> None:
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=1,
        mode="lfs",
    )
    state.open_session(
        "s",
        designated_catalog(["a", "a", "a"], designated_indices={1}),
    )
    acquire_assigned(state, "r1", "a")

    state.cancel_unsubmitted("r1")

    event = state.assignment_history[-1]
    assert event["request_id"] == "r0"
    assert event["tier"] == "probe"
    assert not event["is_designated_probe"]
    assert event["probe_replacement"]
    assert event["probe_selection_semantics"] == "explicit_catalog_flag-v1"


@pytest.mark.parametrize("mode", ["lfs", "oracle_probe_lfs"])
def test_terminally_failed_designated_probe_replacement_can_reveal(
    mode: str,
) -> None:
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=1,
        mode=mode,
    )
    state.open_session(
        "s",
        designated_catalog(["a", "a", "a"], designated_indices={1}),
    )
    acquire_assigned(state, "r1", "a")

    state.fail_terminated("r1", "worker raised")

    event = state.assignment_history[-1]
    assert event["request_id"] == "r0"
    assert event["tier"] == "probe"
    assert not event["is_designated_probe"]
    assert event["probe_replacement"]
    acquire_assigned(state, "r0", "a")
    state.complete("r0", 7)

    expected = 30 if mode == "oracle_probe_lfs" else 7
    assert state.sessions["s"].estimates["a"] == expected
    reveal = state.estimate_rebase_history[-1]
    assert reveal["completed_request_id"] == "r0"
    assert reveal["first_finite_estimate"]
    assert not reveal["completed_request_is_designated_probe"]
    assert reveal["completed_request_is_probe_admission"]
    assert reveal["designated_probe_request_id"] == "r1"
    assert reveal["designated_probe_request_status"] == "failed_terminal"


@pytest.mark.parametrize("mode", ["lfs", "oracle_probe_lfs"])
def test_inflight_unknown_replaces_terminally_failed_designated_probe(
    mode: str,
) -> None:
    state = CrossDpSchedulerState(
        dp_size=1,
        max_num_seqs_per_dp=2,
        mode=mode,
    )
    state.open_session(
        "s",
        designated_catalog(["a", "a", "a", "a"], designated_indices={1}),
    )
    assert [event["request_id"] for event in state.assignment_history] == [
        "r1",
        "r0",
    ]
    acquire_assigned(state, "r1", "a")
    acquire_assigned(state, "r0", "a")

    # An ordinary member that finishes before the designated probe fails may
    # not reveal an estimate. Its free slot admits another unknown member.
    state.complete("r0", 3)
    assert "a" not in state.sessions["s"].estimates
    assert state.assignment_history[-1]["request_id"] == "r2"
    assert state.assignment_history[-1]["tier"] == "unknown_rr"
    acquire_assigned(state, "r2", "a")

    # Once the designated probe is definitively gone, that already-in-flight
    # unknown is the next causal observation available as a replacement.
    state.fail_terminated("r1", "worker raised")
    state.complete("r2", 7)

    expected = 40 if mode == "oracle_probe_lfs" else 7
    assert state.sessions["s"].estimates["a"] == expected
    reveal = state.estimate_rebase_history[-1]
    assert reveal["completed_request_id"] == "r2"
    assert reveal["first_finite_estimate"]
    assert not reveal["completed_request_is_designated_probe"]
    assert not reveal["completed_request_is_probe_admission"]
    assert reveal["completed_request_is_unknown_admission"]
    assert reveal["designated_probe_request_id"] == "r1"
    assert reveal["designated_probe_request_status"] == "failed_terminal"


def test_cancelled_probe_uses_another_inflight_unknown_as_probe() -> None:
    state = CrossDpSchedulerState(dp_size=1, max_num_seqs_per_dp=2, mode="lfs")
    state.open_session("s", catalog(["a", "a", "a"]))
    acquire_assigned(state, "r0", "a")

    state.cancel_unsubmitted("r0")

    assert state.assignment_history[-1]["request_id"] == "r2"
    assert state.assignment_history[-1]["tier"] == "unknown_rr"


def test_last_cancelled_unknown_reopens_probe_tier() -> None:
    state = CrossDpSchedulerState(dp_size=1, max_num_seqs_per_dp=2, mode="lfs")
    state.open_session("s", catalog(["a", "a", "a"]))
    acquire_assigned(state, "r0", "a")
    acquire_assigned(state, "r1", "a")

    # The request originally labelled as the probe can leave before another
    # unknown admission. If every unresolved admission then leaves, the next
    # request must become the replacement probe.
    state.cancel_unsubmitted("r0")
    state.cancel_unsubmitted("r1")
    state.cancel_unsubmitted("r2")
    lease = state.acquire("s", "s:participant:0", "dynamic", "a", 128)

    assert lease is not None
    assert state.assignment_history[-1]["request_id"] == "dynamic"
    assert state.assignment_history[-1]["tier"] == "probe"


def test_definitively_terminated_rpc_releases_capacity() -> None:
    state = CrossDpSchedulerState(dp_size=1, max_num_seqs_per_dp=1, mode="lfs")
    state.open_session("s", catalog(["a", "b"]))
    acquire_assigned(state, "r0", "a")

    state.fail_terminated("r0", "worker raised")

    assert state.snapshot()["fatal_error"] is None
    assert acquire_assigned(state, "r1", "b")["dp_idx"] == 0


def test_submitted_failure_does_not_guess_that_remote_capacity_is_free() -> None:
    state = CrossDpSchedulerState(dp_size=1, max_num_seqs_per_dp=1, mode="lfs")
    state.open_session("s", catalog(["a", "b"]))
    acquire_assigned(state, "r0", "a")

    state.fail_unknown("r0", "timeout")

    assert state.snapshot()["dp_inflight"] == [["r0"]]
    with pytest.raises(RuntimeError, match="timeout"):
        state.acquire("s", "s:participant:0", "r1", "b", 128)


def test_older_session_is_not_starved_by_new_session_probes() -> None:
    state = CrossDpSchedulerState(dp_size=1, max_num_seqs_per_dp=1, mode="lfs")
    state.open_session("s", catalog(["a", "a"]))
    state.open_session(
        "new",
        [{"request_id": "new-r0", "group_id": "new", "fallback_cost": 128}],
    )
    acquire_assigned(state, "r0", "a")

    state.complete("r0", 10)

    assert state.assignment_history[-1]["request_id"] == "r1"
    assert state.assignment_history[-1]["session_id"] == "s"


def test_closing_one_participant_releases_only_its_reservations() -> None:
    state = CrossDpSchedulerState(dp_size=1, max_num_seqs_per_dp=1, mode="lfs")
    state.open_session(
        "s",
        [
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
        ],
        ["p0", "p1"],
    )

    state.close_participant("s", "p0")
    state.close_participant("s", "p0")  # idempotent per participant

    lease = state.acquire("s", "p1", "p1-r0", "b", 128)
    assert lease is not None and lease["dp_idx"] == 0
    assert state.sessions["s"].open_participants == {"p1"}


def test_closing_unknown_participant_is_rejected() -> None:
    state = CrossDpSchedulerState(dp_size=1, max_num_seqs_per_dp=1, mode="lfs")
    state.open_session("s", catalog(["a"]))

    with pytest.raises(KeyError, match="unknown participant"):
        state.close_participant("s", "typo")


def test_session_estimates_do_not_leak_into_the_next_rollout() -> None:
    state = CrossDpSchedulerState(dp_size=1, max_num_seqs_per_dp=1, mode="lfs")
    state.open_session("s", catalog(["a"]))
    acquire_assigned(state, "r0", "a")
    state.complete("r0", 100)
    state.close_participant("s", "s:participant:0")
    assert "s" not in state.sessions

    state.open_session(
        "next",
        [{"request_id": "next-r0", "group_id": "a", "fallback_cost": 128}],
    )
    assert state.assignment_history[-1]["tier"] == "probe"
    assert state.assignment_history[-1]["estimate"] is None


def test_history_lfs_uses_only_completed_previous_session_means() -> None:
    state = CrossDpSchedulerState(
        dp_size=1, max_num_seqs_per_dp=1, mode="history_lfs"
    )
    state.open_session("s", catalog(["a", "a", "b", "b"]))

    # Cold-start requests are FCFS and are never labelled/admitted as probes.
    for request_id, group, length in [
        ("r0", "a", 100),
        ("r1", "a", 60),
        ("r2", "b", 20),
        ("r3", "b", 40),
    ]:
        acquire_assigned(state, request_id, group)
        state.complete(request_id, length)
    assert {event["tier"] for event in state.assignment_history} == {
        "cold_start_fcfs"
    }
    assert state.group_history == {}

    state.close_participant("s", "s:participant:0")
    assert state.snapshot()["group_history"]["a"]["mean"] == 80
    assert state.snapshot()["group_history"]["b"]["mean"] == 30

    state.open_session(
        "next",
        [
            {"request_id": "next-b", "group_id": "b", "fallback_cost": 128},
            {"request_id": "next-a", "group_id": "a", "fallback_cost": 128},
        ],
    )
    # Catalog order is b,a, but historical longest-first schedules a first.
    event = state.assignment_history[-1]
    assert event["request_id"] == "next-a"
    assert event["tier"] == "history_lfs"
    assert event["estimate"] == 80


def test_history_lfs_does_not_use_partial_current_session_results() -> None:
    state = CrossDpSchedulerState(
        dp_size=1, max_num_seqs_per_dp=1, mode="history_lfs"
    )
    state.group_history["a"] = (100, 2)
    state.open_session("s", catalog(["a", "a"]))
    acquire_assigned(state, "r0", "a")
    state.complete("r0", 200)

    # The second request still uses the pre-session mean 50, not 200 or 100.
    event = state.assignment_history[-1]
    assert event["request_id"] == "r1"
    assert event["estimate"] == 50
    assert state.group_history["a"] == (100, 2)


def test_history_lfs_restores_previous_process_history() -> None:
    state = CrossDpSchedulerState(
        dp_size=1, max_num_seqs_per_dp=1, mode="history_lfs"
    )
    state.restore_group_history(
        {
            "a": {"sum": 240, "count": 2, "mean": 120},
            "b": {"sum": 40, "count": 2, "mean": 20},
        }
    )
    state.open_session(
        "resumed",
        [
            {"request_id": "resumed-b", "group_id": "b", "fallback_cost": 128},
            {"request_id": "resumed-a", "group_id": "a", "fallback_cost": 128},
        ],
    )

    event = state.assignment_history[-1]
    assert event["request_id"] == "resumed-a"
    assert event["tier"] == "history_lfs"
    assert event["estimate"] == 120


def test_history_restore_rejects_invalid_state() -> None:
    state = CrossDpSchedulerState(
        dp_size=1, max_num_seqs_per_dp=1, mode="history_lfs"
    )
    with pytest.raises(ValueError, match="count=0"):
        state.restore_group_history({"a": {"sum": 1, "count": 0}})


def test_exact_length_lpt_uses_per_request_output_lengths() -> None:
    state = CrossDpSchedulerState(
        dp_size=1, max_num_seqs_per_dp=1, mode="exact_length_lpt"
    )
    state.open_session(
        "s",
        [
            {"request_id": "short", "group_id": "a", "fallback_cost": 10},
            {"request_id": "long", "group_id": "a", "fallback_cost": 100},
            {"request_id": "medium", "group_id": "b", "fallback_cost": 50},
        ],
    )

    for request_id, group, expected_cost in [
        ("long", "a", 100),
        ("medium", "b", 50),
        ("short", "a", 10),
    ]:
        lease = state.acquire(
            "s", "s:participant:0", request_id, group, expected_cost
        )
        assert lease is not None
        assert lease["predicted_length"] == expected_cost
        event = state.assignment_history[-1]
        assert event["tier"] == "exact_length_lpt"
        assert event["preferred_dp_idx"] is None
        assert event["dp_placement_mode"] == "scheduler_selected"
        assert event["dp_selector_applied"]
        assert event["preferred_dp_pin_honored"] is None
        assert event["dp_selection_matches_declared_mode"]
        state.complete(request_id, expected_cost)


def test_exact_length_pinning_skips_full_dp_without_head_of_line_blocking() -> None:
    state = CrossDpSchedulerState(
        dp_size=2, max_num_seqs_per_dp=1, mode="exact_length_lpt"
    )
    state.open_session(
        "s",
        [
            {
                "request_id": "dp0-longest",
                "group_id": "a",
                "fallback_cost": 100,
                "preferred_dp_idx": 0,
            },
            {
                "request_id": "dp0-blocked",
                "group_id": "b",
                "fallback_cost": 90,
                "preferred_dp_idx": 0,
            },
            {
                "request_id": "dp1-runnable",
                "group_id": "c",
                "fallback_cost": 80,
                "preferred_dp_idx": 1,
            },
            {
                "request_id": "dp1-refill",
                "group_id": "d",
                "fallback_cost": 10,
                "preferred_dp_idx": 1,
            },
        ],
    )

    assert [
        (event["request_id"], event["dp_idx"])
        for event in state.assignment_history
    ] == [("dp0-longest", 0), ("dp1-runnable", 1)]
    assert state.requests["dp0-blocked"].status == "pending"
    assert all(
        event["preferred_dp_idx"] == event["dp_idx"]
        and event["dp_placement_mode"] == "preferred_dp_pinned"
        and not event["dp_selector_applied"]
        and event["preferred_dp_pin_honored"]
        and not event["dp_selection_matches_declared_mode"]
        for event in state.assignment_history
    )

    dp1_lease = state.acquire(
        "s", "s:participant:0", "dp1-runnable", "c", 80
    )
    assert dp1_lease is not None
    state.complete("dp1-runnable", 80)
    assert state.assignment_history[-1]["request_id"] == "dp1-refill"
    assert state.assignment_history[-1]["dp_idx"] == 1
    assert state.requests["dp0-blocked"].status == "pending"

    dp0_lease = state.acquire(
        "s", "s:participant:0", "dp0-longest", "a", 100
    )
    assert dp0_lease is not None
    state.complete("dp0-longest", 100)
    assert state.assignment_history[-1]["request_id"] == "dp0-blocked"
    assert state.assignment_history[-1]["dp_idx"] == 0

    snapshot = state.snapshot()
    assert snapshot["sessions"]["s"]["dp_placement_mode"] == (
        "preferred_dp_pinned"
    )
    assert snapshot["pending_preferred_dp"] == 0
    assert snapshot["pending_preferred_dp_counter_matches_request_states"]


def test_exact_length_pinned_session_rejects_later_unrouted_request() -> None:
    state = CrossDpSchedulerState(
        dp_size=2, max_num_seqs_per_dp=1, mode="exact_length_lpt"
    )
    state.open_session(
        "s",
        [
            {
                "request_id": "catalog-request",
                "group_id": "a",
                "fallback_cost": 100,
                "preferred_dp_idx": 0,
            }
        ],
    )

    with pytest.raises(ValueError, match="bounded first-turn request catalog"):
        state.prepare_acquire(
            "s",
            "s:participant:0",
            "later-request",
            "a",
            100,
        )

    snapshot = state.snapshot()
    assert "later-request" not in state.requests
    assert snapshot["pending"] == 0
    assert snapshot["pending_preferred_dp"] == 0
    assert snapshot["pending_counter_matches_request_states"]
    assert snapshot["pending_preferred_dp_counter_matches_request_states"]


def test_exact_length_pinning_cancellation_preserves_pending_counter() -> None:
    state = CrossDpSchedulerState(
        dp_size=2,
        max_num_seqs_per_dp=1,
        global_admission_limit=1,
        mode="exact_length_lpt",
    )
    state.open_session(
        "s",
        [
            {
                "request_id": "assigned",
                "group_id": "a",
                "fallback_cost": 100,
                "preferred_dp_idx": 0,
            },
            {
                "request_id": "cancelled",
                "group_id": "b",
                "fallback_cost": 90,
                "preferred_dp_idx": 1,
            },
        ],
    )

    assert state.snapshot()["pending_preferred_dp"] == 1
    state.cancel_unsubmitted("cancelled")
    snapshot = state.snapshot()
    assert snapshot["pending"] == 0
    assert snapshot["pending_preferred_dp"] == 0
    assert snapshot["pending_counter_matches_request_states"]
    assert snapshot["pending_preferred_dp_counter_matches_request_states"]


def test_exact_length_pinning_catalog_validation_is_fail_closed() -> None:
    state = CrossDpSchedulerState(
        dp_size=2, max_num_seqs_per_dp=1, mode="exact_length_lpt"
    )
    with pytest.raises(ValueError, match="present on every catalog request"):
        state.open_session(
            "partial",
            [
                {
                    "request_id": "a",
                    "group_id": "a",
                    "fallback_cost": 10,
                    "preferred_dp_idx": 0,
                },
                {
                    "request_id": "b",
                    "group_id": "b",
                    "fallback_cost": 10,
                },
            ],
        )

    for session_id, invalid_value in (
        ("bool", True),
        ("float", 1.0),
        ("negative", -1),
        ("too-large", 2),
    ):
        with pytest.raises(ValueError, match=r"int \(not bool\) in \[0, 2\)"):
            state.open_session(
                session_id,
                [
                    {
                        "request_id": f"{session_id}-request",
                        "group_id": "a",
                        "fallback_cost": 10,
                        "preferred_dp_idx": invalid_value,
                    }
                ],
            )

    fcfs_state = CrossDpSchedulerState(
        dp_size=2, max_num_seqs_per_dp=1, mode="fcfs"
    )
    with pytest.raises(ValueError, match="only supported for exact_length_lpt"):
        fcfs_state.open_session(
            "wrong-mode",
            [
                {
                    "request_id": "request",
                    "group_id": "a",
                    "fallback_cost": 10,
                    "preferred_dp_idx": 0,
                }
            ],
        )


def test_oracle_probe_lfs_reveals_true_group_max_after_probe() -> None:
    state = CrossDpSchedulerState(
        dp_size=1, max_num_seqs_per_dp=1, mode="oracle_probe_lfs"
    )
    state.open_session(
        "s",
        [
            {
                "request_id": "a-short",
                "group_id": "a",
                "fallback_cost": 200,
                "oracle_cost": 10,
            },
            {
                "request_id": "b",
                "group_id": "b",
                "fallback_cost": 200,
                "oracle_cost": 50,
            },
            {
                "request_id": "a-long",
                "group_id": "a",
                "fallback_cost": 200,
                "oracle_cost": 100,
            },
        ],
    )

    assert state.snapshot()["sessions"]["s"]["estimates"] == {}
    assert state.sessions["s"].oracle_estimates == {"a": 100, "b": 50}
    expected = [
        ("a-short", "a", "probe", 200, 100),
        ("b", "b", "probe", 200, 50),
        ("a-long", "a", "oracle_probe_lfs", 200, 100),
    ]
    for request_id, group, tier, fallback, group_max in expected:
        lease = state.acquire(
            "s", "s:participant:0", request_id, group, group_max
        )
        assert lease is not None
        event = state.assignment_history[-1]
        assert event["tier"] == tier
        if tier == "probe":
            assert event["estimate"] is None
            assert lease["predicted_length"] == fallback
        else:
            assert event["estimate"] == group_max
            assert lease["predicted_length"] == group_max
        state.complete(request_id, 1)

    assert state.sessions["s"].probed_groups == {"a", "b"}
    assert state.snapshot()["sessions"]["s"]["estimates"] == {
        "a": 100,
        "b": 50,
    }


def test_close_is_idempotent_after_the_session_is_dropped() -> None:
    state = CrossDpSchedulerState(dp_size=1, max_num_seqs_per_dp=1, mode="lfs")
    state.open_session("s", catalog(["a"]))
    acquire_assigned(state, "r0", "a")
    state.complete("r0", 10)
    state.close_participant("s", "s:participant:0")

    assert "s" not in state.sessions
    state.close_participant("s", "s:participant:0")
    with pytest.raises(KeyError, match="unknown cross-DP session"):
        state.close_participant("typo", "s:participant:0")


def test_assignment_history_is_bounded() -> None:
    state = CrossDpSchedulerState(dp_size=1, max_num_seqs_per_dp=1, mode="fcfs")
    state.open_session("s", catalog(["a"]))
    acquire_assigned(state, "r0", "a")
    state.complete("r0", 1)

    for index in range(1, 12501):
        request_id = f"dynamic-{index}"
        lease = state.acquire(
            "s", "s:participant:0", request_id, "a", 128
        )
        assert lease is not None
        state.complete(request_id, 1)

    assert len(state.assignment_history) <= 12000
    assert state.assignment_history[0]["sequence"] > 0
