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

"""Unit tests for GenerationRouter (shard table, health state-machine, lifecycle)
and FaultInjector / maybe_launch_fault_injector.

No Ray, no GPU, no network required — all heavy external dependencies are mocked.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.models.generation.fault_inject import (
    FaultInjector,
    maybe_launch_fault_injector,
)
from nemo_rl.models.generation.generation_router import GenerationRouter, ShardEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _router(**kw) -> GenerationRouter:
    """Create a GenerationRouter with sensible, fast unit-test defaults."""
    defaults = dict(
        health_poll_interval_s=0.1,
        health_timeout_s=1.0,
        failure_threshold=3,
        join_success_threshold=2,
        reset_collective_timeout_s=5.0,
        joinable_min_age_s=0.0,  # disable age gate unless a test overrides it
        auto_recover=False,  # don't auto-spawn in unit tests
    )
    defaults.update(kw)
    return GenerationRouter(**defaults)


def _entry(shard_id: str, url: str = "http://host:8000/v1", **kw) -> ShardEntry:
    return ShardEntry(shard_id=shard_id, url=url, **kw)


# Minimal fake `ray` module used by remove_shard / add_shard internals.
_FAKE_RAY = MagicMock()
_FAKE_RAY.kill = MagicMock()


# ---------------------------------------------------------------------------
# ShardEntry
# ---------------------------------------------------------------------------


class TestShardEntry:
    def test_health_url_strips_v1_suffix(self):
        e = _entry("dp-0", "http://host:8000/v1")
        assert e._health_url == "http://host:8000/openapi.json"

    def test_health_url_without_v1(self):
        e = _entry("dp-0", "http://host:8000")
        assert e._health_url == "http://host:8000/openapi.json"

    def test_joined_at_defaults_to_now(self):
        before = time.monotonic()
        e = _entry("dp-0")
        after = time.monotonic()
        assert before <= e.joined_at <= after

    def test_explicit_joined_at_respected(self):
        e = _entry("dp-0", joined_at=1234.0)
        assert e.joined_at == 1234.0

    def test_default_status_is_joining(self):
        e = _entry("dp-0")
        assert e.status == "joining"

    def test_in_comm_defaults_false(self):
        assert _entry("dp-0").in_comm is False

    def test_proven_defaults_false(self):
        assert _entry("dp-0").proven is False


# ---------------------------------------------------------------------------
# register_shards
# ---------------------------------------------------------------------------


class TestRegisterShards:
    def test_seeds_shard_table(self):
        r = _router()
        r.register_shards([("dp-0", "http://h:8000/v1"), ("dp-1", "http://h:8001/v1")])
        assert set(r._shards.keys()) == {"dp-0", "dp-1"}

    def test_initial_status_is_ready(self):
        r = _router()
        r.register_shards([("dp-0", "http://h:8000/v1")])
        assert r._shards["dp-0"].status == "ready"

    def test_target_shard_count_set(self):
        r = _router()
        r.register_shards([("dp-0", "http://h:8000/v1"), ("dp-1", "http://h:8001/v1")])
        assert r._target_shard_count == 2

    def test_total_shards_at_bootstrap(self):
        r = _router()
        r.register_shards([("dp-0", "http://h:8000/v1")])
        assert r._total_shards_at_bootstrap == 1

    def test_per_shard_world_size_stored(self):
        r = _router()
        r.register_shards([("dp-0", "http://h:8000/v1")], per_shard_world_size=4)
        assert r._per_shard_world_size == 4

    def test_actor_handles_and_node_id_wired(self):
        r = _router()
        h = MagicMock()
        r.register_shards(
            [("dp-0", "http://h:8000/v1")],
            actor_handles_by_shard={"dp-0": [h]},
            node_id_by_shard={"dp-0": "node-1"},
        )
        assert r._shards["dp-0"].actor_handles == [h]
        assert r._shards["dp-0"].node_id == "node-1"

    def test_worker_indices_wired(self):
        r = _router()
        r.register_shards(
            [("dp-0", "http://h:8000/v1")],
            worker_indices_by_shard={"dp-0": [0, 1, 2, 3]},
        )
        assert r._shards["dp-0"].worker_indices == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# cordon / uncordon
# ---------------------------------------------------------------------------


class TestCordonUncordon:
    def test_cordon_ready_shard(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready")
        asyncio.run(r.cordon("dp-0", "test"))
        assert r._shards["dp-0"].status == "cordoned"
        assert r._shards["dp-0"].consecutive_successes == 0

    def test_cordon_missing_shard_is_noop(self):
        r = _router()
        asyncio.run(r.cordon("dp-99", "test"))  # must not raise

    def test_cordon_already_cordoned_is_noop(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="cordoned", consecutive_successes=5)
        asyncio.run(r.cordon("dp-0", "again"))
        assert r._shards["dp-0"].consecutive_successes == 5  # unchanged

    def test_uncordon_moves_to_joining(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="cordoned")
        asyncio.run(r.uncordon("dp-0"))
        assert r._shards["dp-0"].status == "joining"

    def test_uncordon_resets_failure_counters(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="cordoned", consecutive_failures=7)
        asyncio.run(r.uncordon("dp-0"))
        assert r._shards["dp-0"].consecutive_failures == 0

    def test_uncordon_already_ready_is_noop(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready")
        asyncio.run(r.uncordon("dp-0"))
        assert r._shards["dp-0"].status == "ready"


# ---------------------------------------------------------------------------
# _next_shard_id
# ---------------------------------------------------------------------------


class TestNextShardId:
    def test_first_id_is_dp0(self):
        assert _router()._next_shard_id() == "dp-0"

    def test_increments_past_existing(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0")
        r._shards["dp-1"] = _entry("dp-1")
        assert r._next_shard_id() == "dp-2"

    def test_picks_max_plus_one_not_min_gap(self):
        # Gap at dp-1 but max is dp-2, so next should be dp-3
        r = _router()
        r._shards["dp-0"] = _entry("dp-0")
        r._shards["dp-2"] = _entry("dp-2")
        assert r._next_shard_id() == "dp-3"

    def test_non_dp_shard_ids_ignored(self):
        r = _router()
        r._shards["custom-shard"] = _entry("custom-shard")
        assert r._next_shard_id() == "dp-0"


# ---------------------------------------------------------------------------
# Counting / world-size accessors
# ---------------------------------------------------------------------------


class TestCounters:
    def _pop(self) -> GenerationRouter:
        r = _router()
        r._per_shard_world_size = 4
        r._shards = {
            "dp-0": _entry("dp-0", status="ready"),
            "dp-1": _entry("dp-1", status="joining"),
            "dp-2": _entry("dp-2", status="cordoned"),
        }
        return r

    def test_shard_count_ready(self):
        assert self._pop().shard_count_ready() == 1

    def test_shard_count_alive_for_collective(self):
        # ready + joining = 2; cordoned excluded
        assert self._pop().shard_count_alive_for_collective() == 2

    def test_current_gen_world_size(self):
        # 2 alive × 4 per-shard
        assert self._pop().current_gen_world_size() == 8

    def test_world_size_zero_when_all_cordoned(self):
        r = _router()
        r._per_shard_world_size = 4
        r._shards["dp-0"] = _entry("dp-0", status="cordoned")
        assert r.current_gen_world_size() == 0


# ---------------------------------------------------------------------------
# refit_ready_state
# ---------------------------------------------------------------------------


class TestRefitReadyState:
    def test_ok_when_ready_shard_exists(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready")
        ok, reason = r.refit_ready_state()
        assert ok is True
        assert reason == "ok"

    def test_blocked_when_nccl_reinit_in_progress(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready")
        r._nccl_reinit_in_progress = True
        ok, reason = r.refit_ready_state()
        assert ok is False
        assert "nccl_reinit" in reason

    def test_blocked_when_no_shards_alive(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="cordoned")
        ok, reason = r.refit_ready_state()
        assert ok is False
        assert "no_shards_alive" in reason

    def test_joining_shard_counts_as_alive(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="joining")
        ok, _ = r.refit_ready_state()
        assert ok is True

    def test_empty_table_is_blocked(self):
        ok, reason = _router().refit_ready_state()
        assert ok is False
        assert "no_shards_alive" in reason


# ---------------------------------------------------------------------------
# promote_all_joining + _eligible_promote_sids
# ---------------------------------------------------------------------------


class TestPromoteAllJoining:
    def test_promotes_all_joining(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="joining")
        r._shards["dp-1"] = _entry("dp-1", status="joining")
        r._shards["dp-2"] = _entry("dp-2", status="ready")
        promoted = r.promote_all_joining()
        assert set(promoted) == {"dp-0", "dp-1"}
        assert r._shards["dp-0"].status == "ready"
        assert r._shards["dp-1"].status == "ready"

    def test_eligible_sids_filter(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="joining")
        r._shards["dp-1"] = _entry("dp-1", status="joining")
        promoted = r.promote_all_joining(eligible_sids={"dp-0"})
        assert promoted == ["dp-0"]
        assert r._shards["dp-1"].status == "joining"  # not promoted

    def test_eligible_sids_none_promotes_all(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="joining")
        r._shards["dp-1"] = _entry("dp-1", status="joining")
        promoted = r.promote_all_joining(eligible_sids=None)
        assert len(promoted) == 2

    def test_cordoned_not_promoted(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="cordoned")
        assert r.promote_all_joining() == []
        assert r._shards["dp-0"].status == "cordoned"


class TestEligiblePromoteSids:
    def test_only_joining_and_in_comm(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="joining", in_comm=True)
        r._shards["dp-1"] = _entry("dp-1", status="joining", in_comm=False)
        r._shards["dp-2"] = _entry("dp-2", status="ready", in_comm=True)
        assert r._eligible_promote_sids() == {"dp-0"}

    def test_empty_when_no_joining_in_comm(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready", in_comm=True)
        assert r._eligible_promote_sids() == set()


# ---------------------------------------------------------------------------
# _is_joinable / joinable_shard_count / _joinable_cohort
# ---------------------------------------------------------------------------


class TestJoinable:
    def test_ready_always_joinable(self):
        r = _router(joinable_min_age_s=9999.0)
        s = _entry("dp-0", status="ready")
        assert r._is_joinable(s, time.monotonic()) is True

    def test_joining_requires_enough_successes(self):
        r = _router(join_success_threshold=2, joinable_min_age_s=0.0)
        s = _entry("dp-0", status="joining", consecutive_successes=1)
        assert r._is_joinable(s, time.monotonic()) is False

    def test_joining_proven_bypasses_age_gate(self):
        r = _router(join_success_threshold=2, joinable_min_age_s=9999.0)
        s = _entry("dp-0", status="joining", consecutive_successes=2, proven=True, joined_at=1.0)
        assert r._is_joinable(s, time.monotonic()) is True

    def test_joining_unproven_blocked_by_age_gate(self):
        r = _router(join_success_threshold=2, joinable_min_age_s=9999.0)
        # joined_at=now → too young
        s = _entry("dp-0", status="joining", consecutive_successes=2, proven=False)
        assert r._is_joinable(s, time.monotonic()) is False

    def test_joining_unproven_aged_past_gate(self):
        r = _router(join_success_threshold=2, joinable_min_age_s=1.0)
        # joined_at=1.0 (ancient monotonic value) → well past 1s age gate
        s = _entry("dp-0", status="joining", consecutive_successes=2, proven=False, joined_at=1.0)
        assert r._is_joinable(s, time.monotonic()) is True

    def test_cordoned_not_joinable(self):
        r = _router()
        s = _entry("dp-0", status="cordoned", consecutive_successes=99, proven=True)
        assert r._is_joinable(s, time.monotonic()) is False

    def test_joinable_shard_count(self):
        r = _router(joinable_min_age_s=0.0, join_success_threshold=1)
        r._shards["dp-0"] = _entry("dp-0", status="ready")
        r._shards["dp-1"] = _entry("dp-1", status="joining", consecutive_successes=1, proven=True)
        r._shards["dp-2"] = _entry("dp-2", status="cordoned")
        assert r.joinable_shard_count() == 2

    def test_joinable_cohort_sorted_by_leader_worker_index(self):
        r = _router(joinable_min_age_s=0.0)
        r._shards["dp-0"] = _entry("dp-0", status="ready", worker_indices=[4, 5])
        r._shards["dp-1"] = _entry("dp-1", status="ready", worker_indices=[0, 1])
        sids, indices = r._joinable_cohort()
        # dp-1 has leader idx 0, dp-0 has leader idx 4 → dp-1 first
        assert sids == ["dp-1", "dp-0"]
        assert indices == [0, 1, 4, 5]

    def test_joinable_cohort_excludes_non_joinable(self):
        r = _router(joinable_min_age_s=9999.0)
        r._shards["dp-0"] = _entry("dp-0", status="ready", worker_indices=[0])
        # Joining but 0 successes and unproven → not joinable regardless of age
        r._shards["dp-1"] = _entry(
            "dp-1", status="joining", consecutive_successes=0, proven=False, worker_indices=[1]
        )
        sids, indices = r._joinable_cohort()
        assert sids == ["dp-0"]
        assert indices == [0]

    def test_joinable_world_size(self):
        r = _router(joinable_min_age_s=0.0)
        r._per_shard_world_size = 4
        r._shards["dp-0"] = _entry("dp-0", status="ready")
        r._shards["dp-1"] = _entry("dp-1", status="ready")
        assert r.joinable_world_size() == 8


# ---------------------------------------------------------------------------
# _apply_shard_health_result (synchronous state machine)
# ---------------------------------------------------------------------------


class TestApplyShardHealthResult:
    """Run inside asyncio.run() so asyncio.create_task() in the cordon path works."""

    def _apply(
        self,
        router: GenerationRouter,
        entry: ShardEntry,
        ok: bool,
        n_alive: int = 1,
        nccl_paused: bool = False,
        definitive_death: bool = False,
    ) -> int:
        async def _inner():
            result = router._apply_shard_health_result(
                entry, entry.shard_id, ok, n_alive, nccl_paused,
                definitive_death=definitive_death,
            )
            await asyncio.sleep(0)  # drain any created tasks
            return result

        return asyncio.run(_inner())

    def test_ok_resets_failures_increments_successes(self):
        r = _router()
        e = _entry("dp-0", status="ready", consecutive_failures=5)
        self._apply(r, e, ok=True)
        assert e.consecutive_failures == 0
        assert e.consecutive_successes == 1

    def test_ok_updates_last_health_ok_at(self):
        r = _router()
        e = _entry("dp-0", status="ready")
        before = time.monotonic()
        self._apply(r, e, ok=True)
        assert e.last_health_ok_at >= before

    def test_failure_increments_counter_below_threshold(self):
        r = _router(failure_threshold=3)
        e = _entry("dp-0", status="ready", consecutive_failures=1)
        self._apply(r, e, ok=False, n_alive=1)
        assert e.consecutive_failures == 2
        assert e.status == "ready"

    def test_failure_at_threshold_cordons(self):
        r = _router(failure_threshold=3)
        e = _entry("dp-0", status="ready", consecutive_failures=2)
        self._apply(r, e, ok=False, n_alive=2)
        assert e.consecutive_failures == 3
        assert e.status == "cordoned"

    def test_nccl_paused_resets_failures_no_cordon(self):
        r = _router(failure_threshold=3)
        e = _entry("dp-0", status="ready", consecutive_failures=2)
        self._apply(r, e, ok=False, nccl_paused=True)
        assert e.consecutive_failures == 0
        assert e.status == "ready"

    def test_definitive_death_bypasses_nccl_paused_and_threshold(self):
        r = _router(failure_threshold=3)
        e = _entry("dp-0", status="ready", consecutive_failures=0)
        # Even with nccl_paused=True and failures=0, definitive_death cordons immediately
        self._apply(r, e, ok=False, n_alive=2, nccl_paused=True, definitive_death=True)
        assert e.status == "cordoned"

    def test_auto_uncordon_when_all_dead_after_enough_successes(self):
        # consecutive_successes starts at 1; after one ok tick it reaches threshold=2
        r = _router(join_success_threshold=2)
        e = _entry("dp-0", status="cordoned", consecutive_successes=1)
        n = self._apply(r, e, ok=True, n_alive=0)
        assert e.status == "joining"
        assert n == 1

    def test_auto_uncordon_not_triggered_before_threshold(self):
        r = _router(join_success_threshold=3)
        e = _entry("dp-0", status="cordoned", consecutive_successes=0)
        self._apply(r, e, ok=True, n_alive=0)
        # Only 1 success so far, threshold is 3 → no uncordon
        assert e.status == "cordoned"

    def test_failure_on_joining_shard_cordons_at_threshold(self):
        r = _router(failure_threshold=2)
        e = _entry("dp-0", status="joining", consecutive_failures=1)
        self._apply(r, e, ok=False, n_alive=2)
        assert e.status == "cordoned"


# ---------------------------------------------------------------------------
# _is_rendezvous_master_failure
# ---------------------------------------------------------------------------


class TestIsRendezvousMasterFailure:
    def test_empty_failed_idxs_returns_false(self):
        assert GenerationRouter._is_rendezvous_master_failure([], [None, None]) is False

    def test_all_dist_store_errors_returns_true(self):
        exc_types = ["DistStoreError", "DistStoreError"]
        assert GenerationRouter._is_rendezvous_master_failure([0, 1], exc_types) is True

    def test_all_dist_network_errors_returns_true(self):
        exc_types = ["DistNetworkError", "DistNetworkError"]
        assert GenerationRouter._is_rendezvous_master_failure([0, 1], exc_types) is True

    def test_partial_failure_returns_false(self):
        # Only 1 of 2 workers failed → not a master-side single point
        exc_types = ["DistStoreError", None]
        assert GenerationRouter._is_rendezvous_master_failure([0], exc_types) is False

    def test_mixed_exception_types_returns_false(self):
        exc_types = ["DistStoreError", "RuntimeError"]
        assert GenerationRouter._is_rendezvous_master_failure([0, 1], exc_types) is False

    def test_pending_mixed_with_rendezvous_returns_false(self):
        exc_types = ["DistStoreError", "PENDING"]
        assert GenerationRouter._is_rendezvous_master_failure([0, 1], exc_types) is False

    def test_ray_actor_error_returns_false(self):
        exc_types = ["RayActorError"]
        assert GenerationRouter._is_rendezvous_master_failure([0], exc_types) is False


# ---------------------------------------------------------------------------
# remove_shard (async, Ray mocked)
# ---------------------------------------------------------------------------


class TestRemoveShard:
    def _run(self, coro):
        with patch.dict(sys.modules, {"ray": _FAKE_RAY}):
            return asyncio.run(coro)

    def test_removes_shard_from_table(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready")
        res = self._run(r.remove_shard("dp-0", reason="test"))
        assert "dp-0" not in r._shards
        assert res["removed"] is True

    def test_not_found_returns_removed_false(self):
        r = _router()
        res = self._run(r.remove_shard("dp-99", reason="missing"))
        assert res["removed"] is False

    def test_increments_cumulative_removed(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready")
        self._run(r.remove_shard("dp-0", "test"))
        assert r._cumulative_shards_removed == 1

    def test_freed_shard_slots_appended_with_node_id(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready", node_id="node-A")
        self._run(r.remove_shard("dp-0", "test"))
        assert len(r._freed_shard_slots) == 1
        node_id, _ = r._freed_shard_slots[0]
        assert node_id == "node-A"

    def test_comm_reset_epoch_bumped_for_in_comm_shard(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready", in_comm=True)
        self._run(r.remove_shard("dp-0", "test"))
        assert r._comm_reset_epoch == 1

    def test_comm_reset_epoch_not_bumped_for_non_in_comm_shard(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready", in_comm=False)
        self._run(r.remove_shard("dp-0", "test"))
        assert r._comm_reset_epoch == 0

    def test_nccl_reinit_cleared_after_remove(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready")
        self._run(r.remove_shard("dp-0", "test"))
        assert r._nccl_reinit_in_progress is False

    def test_actor_handles_killed(self):
        _FAKE_RAY.kill.reset_mock()
        r = _router()
        mock_actor = MagicMock()
        r._shards["dp-0"] = _entry("dp-0", status="ready", actor_handles=[mock_actor])
        self._run(r.remove_shard("dp-0", "test"))
        _FAKE_RAY.kill.assert_called_once_with(mock_actor, no_restart=True)

    def test_freed_slots_fifo_order_across_multiple_removes(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready", node_id="node-A")
        r._shards["dp-1"] = _entry("dp-1", status="ready", node_id="node-B")

        async def _go():
            await r.remove_shard("dp-0", "test")
            await r.remove_shard("dp-1", "test")

        self._run(_go())
        assert r._freed_shard_slots[0][0] == "node-A"
        assert r._freed_shard_slots[1][0] == "node-B"

    def test_world_size_in_result(self):
        r = _router()
        r._per_shard_world_size = 2
        r._shards["dp-0"] = _entry("dp-0", status="ready")
        r._shards["dp-1"] = _entry("dp-1", status="ready")
        res = self._run(r.remove_shard("dp-0", "test"))
        # After removing dp-0, 1 ready shard × 2 per-shard = 2
        assert res["world_size"] == 2


# ---------------------------------------------------------------------------
# add_shard (async, generation mocked)
# ---------------------------------------------------------------------------


def _mock_gen(leader_indices: list[int] | None = None) -> MagicMock:
    gen = MagicMock()
    gen.add_dp_worker.return_value = ([], ([], []), [0], "http://host:9000/v1")
    wg = MagicMock()
    wg.dp_leader_worker_indices = leader_indices if leader_indices is not None else [0]
    wg.dp_size = 1
    gen.worker_group = wg
    return gen


class TestAddShard:
    def _run(self, coro):
        with patch.dict(sys.modules, {"ray": _FAKE_RAY}):
            return asyncio.run(coro)

    def test_add_shard_no_generation_returns_error(self):
        r = _router()
        res = asyncio.run(r.add_shard("test"))
        assert res["added"] is False
        assert "no generation" in res["reason"]

    def test_add_shard_registers_joining_entry(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready")
        r._generation = _mock_gen(leader_indices=[0, 1])
        res = self._run(r.add_shard("test"))
        assert res["added"] is True
        joining = [s for s in r._shards.values() if s.status == "joining"]
        assert len(joining) == 1

    def test_add_shard_increments_cumulative_added(self):
        r = _router()
        r._generation = _mock_gen()
        self._run(r.add_shard("test"))
        assert r._cumulative_shards_added == 1

    def test_add_shard_pops_freed_slot_and_passes_to_add_dp_worker(self):
        r = _router()
        r._generation = _mock_gen()
        r._freed_shard_slots.append(("node-X", ("idx-tuple", [3])))
        self._run(r.add_shard("test"))
        # Slot consumed
        assert len(r._freed_shard_slots) == 0
        call_kw = r._generation.add_dp_worker.call_args.kwargs
        assert call_kw["node_id"] == "node-X"

    def test_add_shard_clears_nccl_reinit_on_dp_worker_failure(self):
        r = _router()
        gen = MagicMock()
        gen.add_dp_worker.side_effect = RuntimeError("spawn failed")
        r._generation = gen
        res = asyncio.run(r.add_shard("test"))
        assert res["added"] is False
        assert r._nccl_reinit_in_progress is False

    def test_add_shard_uses_next_dp_id(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready")
        r._generation = _mock_gen(leader_indices=[0, 1])
        self._run(r.add_shard("test"))
        assert "dp-1" in r._shards

    def test_add_shard_result_status_is_joining(self):
        r = _router()
        r._generation = _mock_gen()
        res = self._run(r.add_shard("test"))
        assert res["status"] == "joining"


# ---------------------------------------------------------------------------
# metrics_snapshot
# ---------------------------------------------------------------------------


class TestMetricsSnapshot:
    def test_counts_by_status(self):
        r = _router()
        r._per_shard_world_size = 2
        r._shards["dp-0"] = _entry("dp-0", status="ready")
        r._shards["dp-1"] = _entry("dp-1", status="joining")
        r._shards["dp-2"] = _entry("dp-2", status="cordoned")
        snap = r.metrics_snapshot()
        assert snap["num_ready_shards"] == 1
        assert snap["num_joining_shards"] == 1
        assert snap["num_cordoned_shards"] == 1
        assert snap["num_total_shards"] == 3
        assert snap["current_gen_world_size"] == 4  # 2 alive × 2

    def test_counters_included(self):
        r = _router()
        r._total_shards_at_bootstrap = 4
        r._cumulative_shards_removed = 2
        r._cumulative_shards_added = 1
        r._comm_reset_epoch = 3
        snap = r.metrics_snapshot()
        assert snap["total_shards_at_bootstrap"] == 4
        assert snap["cumulative_shards_removed"] == 2
        assert snap["cumulative_shards_added"] == 1
        assert snap["comm_epoch"] == 3

    def test_per_shard_list_has_correct_fields(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready", consecutive_failures=2)
        snap = r.metrics_snapshot()
        assert len(snap["per_shard"]) == 1
        ps = snap["per_shard"][0]
        assert ps["shard_id"] == "dp-0"
        assert ps["status"] == "ready"
        assert ps["consecutive_failures"] == 2

    def test_nccl_reinit_flag_reflected(self):
        r = _router()
        r._nccl_reinit_in_progress = True
        snap = r.metrics_snapshot()
        assert snap["nccl_reinit_in_progress"] is True


# ---------------------------------------------------------------------------
# get_shards_list
# ---------------------------------------------------------------------------


class TestGetShardsList:
    def test_returns_all_shards(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready")
        r._shards["dp-1"] = _entry("dp-1", status="joining")
        lst = r.get_shards_list()
        assert {s["shard_id"] for s in lst} == {"dp-0", "dp-1"}

    def test_required_fields_present(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready", consecutive_failures=3)
        s = r.get_shards_list()[0]
        for field in ("shard_id", "status", "url", "consecutive_failures", "consecutive_successes"):
            assert field in s

    def test_empty_table_returns_empty_list(self):
        assert _router().get_shards_list() == []


# ---------------------------------------------------------------------------
# _fire_cordon_hook: last-alive guard
# ---------------------------------------------------------------------------


class TestFireCordonHook:
    def test_skips_remove_when_no_alive_shards(self):
        """If the shard is already cordoned and it's the only one, auto-remove
        is skipped to avoid dropping the fleet to 0."""
        r = _router()
        r._generation = MagicMock()
        r._shards["dp-0"] = _entry("dp-0", status="cordoned")

        removed = []
        with patch.object(r, "remove_shard", side_effect=lambda *a, **kw: removed.append(True)):
            asyncio.run(r._fire_cordon_hook("dp-0", "test"))

        assert removed == []  # remove_shard must NOT be called

    def test_no_generation_returns_without_removing(self):
        r = _router()
        r._generation = None
        r._shards["dp-0"] = _entry("dp-0", status="ready")
        r._shards["dp-1"] = _entry("dp-1", status="cordoned")

        removed = []
        with patch.object(r, "remove_shard", side_effect=lambda *a, **kw: removed.append(True)):
            asyncio.run(r._fire_cordon_hook("dp-1", "test"))

        assert removed == []


# ---------------------------------------------------------------------------
# Joinable stability tracking
# ---------------------------------------------------------------------------


class TestJoinableStability:
    def test_timer_unchanged_when_count_stable(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready")
        r._last_joinable_count = 1  # matches current count
        changed_at = r._joinable_changed_at
        r._refresh_joinable_stability()
        assert r._joinable_changed_at == changed_at

    def test_timer_resets_when_count_changes(self):
        r = _router()
        r._shards["dp-0"] = _entry("dp-0", status="ready")
        r._last_joinable_count = 99  # mismatch
        old = r._joinable_changed_at
        time.sleep(0.01)
        r._refresh_joinable_stability()
        assert r._joinable_changed_at > old

    def test_joinable_stable_for_s_grows_over_time(self):
        r = _router()
        r._joinable_changed_at = time.monotonic() - 5.0
        assert r.joinable_stable_for_s() >= 4.9


# ---------------------------------------------------------------------------
# FaultInjector._wait_for_trigger
# ---------------------------------------------------------------------------


class TestFaultInjectorWaitForTrigger:
    def _fi(self, **kw) -> FaultInjector:
        gen = MagicMock()
        gen._router = MagicMock()
        return FaultInjector(vllm_gen=gen, mode="ray-kill", target_shard="dp-0", **kw)

    def test_sleeps_for_exact_duration(self):
        fi = self._fi()
        slept = []
        with patch("nemo_rl.models.generation.fault_inject.time.sleep", side_effect=slept.append):
            fi._wait_for_trigger(0.123)
        assert slept == [0.123]

    def test_zero_delay_still_calls_sleep(self):
        fi = self._fi()
        slept = []
        with patch("nemo_rl.models.generation.fault_inject.time.sleep", side_effect=slept.append):
            fi._wait_for_trigger(0.0)
        assert slept == [0.0]


# ---------------------------------------------------------------------------
# FaultInjector._wait_for_training_started
# ---------------------------------------------------------------------------


class TestWaitForTrainingStarted:
    def test_returns_when_refit_count_reaches_one(self):
        gen = MagicMock()
        router = MagicMock()
        router._refit_attempts = 0
        gen._router = router
        fi = FaultInjector(vllm_gen=gen, mode="ray-kill", target_shard="dp-0")

        call_n = [0]

        def _side(s):
            call_n[0] += 1
            if call_n[0] == 2:
                router._refit_attempts = 1

        with patch("nemo_rl.models.generation.fault_inject.time.sleep", side_effect=_side):
            fi._wait_for_training_started(timeout_s=10.0, poll_every_s=0.001)

        assert router._refit_attempts == 1

    def test_proceeds_after_timeout_without_raising(self):
        gen = MagicMock()
        router = MagicMock()
        router._refit_attempts = 0
        gen._router = router
        fi = FaultInjector(vllm_gen=gen, mode="ray-kill", target_shard="dp-0")
        # timeout_s=0 → deadline already past on first check
        fi._wait_for_training_started(timeout_s=0.0, poll_every_s=10.0)


# ---------------------------------------------------------------------------
# FaultInjector._wait_for_steady_state
# ---------------------------------------------------------------------------


class TestWaitForSteadyState:
    def _fi_with_router(self, shards_list, refit_ready):
        gen = MagicMock()
        router = MagicMock()
        router.get_shards_list.return_value = shards_list
        router.refit_ready_state.return_value = refit_ready
        router._target_shard_count = sum(
            1 for s in shards_list if s["status"] == "ready"
        )
        gen._router = router
        return FaultInjector(vllm_gen=gen, mode="ray-kill", target_shard="dp-0")

    def test_returns_true_when_fleet_ready(self):
        fi = self._fi_with_router(
            [{"shard_id": "dp-0", "status": "ready"}], (True, "ok")
        )
        with patch("nemo_rl.models.generation.fault_inject.time.sleep"):
            assert fi._wait_for_steady_state(timeout_s=1.0) is True

    def test_returns_false_on_timeout(self):
        fi = self._fi_with_router(
            [{"shard_id": "dp-0", "status": "joining"}],
            (False, "nccl_reinit_in_progress"),
        )
        with patch("nemo_rl.models.generation.fault_inject.time.sleep"):
            assert fi._wait_for_steady_state(timeout_s=0.0, poll_every_s=0.0) is False

    def test_blocks_while_shards_joining(self):
        gen = MagicMock()
        router = MagicMock()
        call_n = [0]

        def _shards():
            call_n[0] += 1
            if call_n[0] < 3:
                return [{"shard_id": "dp-0", "status": "joining"}]
            return [{"shard_id": "dp-0", "status": "ready"}]

        router.get_shards_list.side_effect = _shards
        router.refit_ready_state.side_effect = lambda: (True, "ok") if call_n[0] >= 3 else (False, "x")
        router._target_shard_count = 1
        gen._router = router
        fi = FaultInjector(vllm_gen=gen, mode="ray-kill", target_shard="dp-0")

        with patch("nemo_rl.models.generation.fault_inject.time.sleep"):
            result = fi._wait_for_steady_state(timeout_s=10.0)

        assert result is True
        assert call_n[0] >= 3


# ---------------------------------------------------------------------------
# FaultInjector._pick_first_ready_shard
# ---------------------------------------------------------------------------


class TestPickFirstReadyShard:
    def _fi(self, shards_list, grace_s=0.0):
        gen = MagicMock()
        router = MagicMock()
        router.get_shards_list.return_value = shards_list
        gen._router = router
        return FaultInjector(
            vllm_gen=gen,
            mode="ray-kill",
            target_shard="dp-0",
            new_shard_grace_period_s=grace_s,
        )

    def test_picks_lowest_numbered_ready_shard(self):
        fi = self._fi([
            {"shard_id": "dp-0", "status": "ready"},
            {"shard_id": "dp-1", "status": "ready"},
        ])
        assert fi._pick_first_ready_shard() == "dp-0"

    def test_excludes_specified_shard_ids(self):
        fi = self._fi([
            {"shard_id": "dp-0", "status": "ready"},
            {"shard_id": "dp-1", "status": "ready"},
        ])
        assert fi._pick_first_ready_shard(exclude={"dp-0"}) == "dp-1"

    def test_returns_none_when_no_ready_shards(self):
        fi = self._fi([{"shard_id": "dp-0", "status": "joining"}])
        assert fi._pick_first_ready_shard() is None

    def test_returns_none_when_all_excluded(self):
        fi = self._fi([{"shard_id": "dp-0", "status": "ready"}])
        assert fi._pick_first_ready_shard(exclude={"dp-0"}) is None

    def test_grace_period_falls_back_to_oldest_ready(self):
        fi = self._fi([{"shard_id": "dp-0", "status": "ready"}], grace_s=9999.0)
        # Seed first_seen so shard is "new" (within grace period)
        fi._first_seen_ready["dp-0"] = time.monotonic()
        # Fallback: should still return dp-0 (only candidate)
        assert fi._pick_first_ready_shard() == "dp-0"

    def test_aged_shard_preferred_over_new(self):
        fi = self._fi([
            {"shard_id": "dp-0", "status": "ready"},
            {"shard_id": "dp-1", "status": "ready"},
        ], grace_s=9999.0)
        now = time.monotonic()
        # dp-1 is old (past grace period), dp-0 is brand new
        fi._first_seen_ready["dp-0"] = now
        fi._first_seen_ready["dp-1"] = now - 10000.0
        assert fi._pick_first_ready_shard() == "dp-1"

    def test_non_ready_shards_ignored(self):
        fi = self._fi([
            {"shard_id": "dp-0", "status": "joining"},
            {"shard_id": "dp-1", "status": "ready"},
        ])
        assert fi._pick_first_ready_shard() == "dp-1"


# ---------------------------------------------------------------------------
# maybe_launch_fault_injector
# ---------------------------------------------------------------------------


class TestMaybeLaunchFaultInjector:
    def test_disabled_returns_empty_list(self):
        cfg = {"fault_inject": {"enabled": False}}
        assert maybe_launch_fault_injector(cfg, MagicMock()) == []

    def test_no_fault_inject_key_returns_empty_list(self):
        assert maybe_launch_fault_injector({}, MagicMock()) == []

    def test_none_config_returns_empty_list(self):
        assert maybe_launch_fault_injector(None, MagicMock()) == []

    def test_no_router_returns_empty_list(self):
        gen = MagicMock(spec=[])  # spec=[] means no _router attribute
        cfg = {"fault_inject": {"enabled": True, "mode": "ray-kill"}}
        assert maybe_launch_fault_injector(cfg, gen) == []

    def test_none_vllm_gen_returns_empty_list(self):
        cfg = {"fault_inject": {"enabled": True, "mode": "ray-kill"}}
        assert maybe_launch_fault_injector(cfg, None) == []

    def test_single_fault_launches_one_daemon_thread(self):
        gen = MagicMock()
        gen._router = MagicMock()
        cfg = {
            "fault_inject": {
                "enabled": True,
                "mode": "ray-kill",
                "target_shard": "dp-0",
                "trigger_after_s": 999_999,  # will never fire in the test
            }
        }
        threads = maybe_launch_fault_injector(cfg, gen)
        assert len(threads) == 1
        assert threads[0].daemon is True

    def test_schedule_launches_one_thread_per_entry(self):
        gen = MagicMock()
        gen._router = MagicMock()
        cfg = {
            "fault_inject": {
                "enabled": True,
                "mode": "ray-kill",
                "schedule": [
                    {"target_shard": "dp-0", "trigger_after_s": 999_999},
                    {"target_shard": "dp-1", "trigger_after_s": 999_999},
                    {"target_shard": "dp-2", "trigger_after_s": 999_999},
                ],
            }
        }
        threads = maybe_launch_fault_injector(cfg, gen)
        assert len(threads) == 3

    def test_mode_overridden_per_schedule_entry(self):
        gen = MagicMock()
        gen._router = MagicMock()
        cfg = {
            "fault_inject": {
                "enabled": True,
                "mode": "actor-kill",  # default
                "schedule": [
                    {"target_shard": "dp-0", "trigger_after_s": 999_999, "mode": "ray-kill"},
                    {"target_shard": "dp-1", "trigger_after_s": 999_999},
                ],
            }
        }
        injectors: list[FaultInjector] = []

        def _capture_start(self_fi):
            injectors.append(self_fi)
            t = threading.Thread(target=lambda: None, daemon=True)
            t.start()
            return t

        with patch.object(FaultInjector, "start", _capture_start):
            maybe_launch_fault_injector(cfg, gen)

        assert injectors[0].mode == "ray-kill"
        assert injectors[1].mode == "actor-kill"

    def test_repeat_every_s_wired_to_injector(self):
        gen = MagicMock()
        gen._router = MagicMock()
        cfg = {
            "fault_inject": {
                "enabled": True,
                "mode": "ray-kill",
                "target_shard": "dp-0",
                "trigger_after_s": 999_999,
                "repeat_every_s": 300,
            }
        }
        injectors: list[FaultInjector] = []

        def _capture(self_fi):
            injectors.append(self_fi)
            t = threading.Thread(target=lambda: None, daemon=True)
            t.start()
            return t

        with patch.object(FaultInjector, "start", _capture):
            maybe_launch_fault_injector(cfg, gen)

        assert injectors[0].repeat_every_s == 300


# ---------------------------------------------------------------------------
# _pick_ready_leader_idx: joining-shard skip during generate_async
# ---------------------------------------------------------------------------


def _make_gen_stub(shards: list[dict]) -> MagicMock:
    """Build a minimal VllmGeneration-like stub for _pick_ready_leader_idx.

    ``shards`` is a list of dicts with keys:
      - ``status``: "ready" | "joining" | "cordoned"
      - ``worker_indices``: list[int]  (leader is [0])

    The stub exposes ``_router._shards``, ``worker_group``, and
    ``current_generate_dp_shard_idx`` to match the real implementation.
    """
    from types import SimpleNamespace

    # Build the router shard table.
    router_shards: dict[str, Any] = {}
    for i, s in enumerate(shards):
        shard_id = f"dp-{i}"
        e = _entry(
            shard_id,
            status=s["status"],
            worker_indices=s["worker_indices"],
        )
        router_shards[shard_id] = e

    router = MagicMock()
    router._shards = router_shards

    # Build the worker_group: dp_leader_worker_indices are the first index
    # of each shard's worker_indices list.
    leaders = [s["worker_indices"][0] for s in shards]
    wg = MagicMock()
    wg.dp_size = len(leaders)
    wg.get_dp_leader_worker_idx.side_effect = lambda idx: leaders[idx % len(leaders)]

    stub = SimpleNamespace(
        _router=router,
        worker_group=wg,
        current_generate_dp_shard_idx=0,
    )
    # Bind the method to the stub (it reads self._router, self.worker_group, self.current_generate_dp_shard_idx)
    from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
    stub._pick_ready_leader_idx = VllmGeneration._pick_ready_leader_idx.__get__(stub)
    return stub


class TestPickReadyLeaderIdx:
    """Unit tests for VllmGeneration._pick_ready_leader_idx.

    The method must skip joining shards (stale dummy weights) and return the
    next ready shard's leader worker index, preserving round-robin correctness.
    """

    def test_ready_shard_returned_directly(self):
        gen = _make_gen_stub([
            {"status": "ready", "worker_indices": [0]},
        ])
        leader = gen._pick_ready_leader_idx()
        assert leader == 0
        assert gen.current_generate_dp_shard_idx == 0  # not advanced

    def test_joining_shard_skipped_next_ready_selected(self):
        gen = _make_gen_stub([
            {"status": "joining", "worker_indices": [0]},  # dp-0: joining → skip
            {"status": "ready",   "worker_indices": [1]},  # dp-1: ready → pick
        ])
        leader = gen._pick_ready_leader_idx()
        assert leader == 1
        # Index should have advanced past the joining shard
        assert gen.current_generate_dp_shard_idx == 1

    def test_multiple_joining_shards_skipped(self):
        gen = _make_gen_stub([
            {"status": "joining", "worker_indices": [0]},
            {"status": "joining", "worker_indices": [1]},
            {"status": "ready",   "worker_indices": [2]},
        ])
        leader = gen._pick_ready_leader_idx()
        assert leader == 2
        assert gen.current_generate_dp_shard_idx == 2

    def test_all_joining_falls_back_to_current(self):
        """Full-fleet recovery edge case: every shard is joining.
        _pick_ready_leader_idx must still return a leader (no infinite loop).
        """
        gen = _make_gen_stub([
            {"status": "joining", "worker_indices": [0]},
            {"status": "joining", "worker_indices": [1]},
        ])
        leader = gen._pick_ready_leader_idx()
        # Falls back — must return a valid leader (not raise)
        assert leader in (0, 1)

    def test_no_router_returns_current_leader(self):
        """Without a router (colocated mode) every shard is dispatched directly."""
        gen = _make_gen_stub([
            {"status": "joining", "worker_indices": [0]},
            {"status": "joining", "worker_indices": [1]},
        ])
        gen._router = None  # colocated / no FT
        leader = gen._pick_ready_leader_idx()
        assert leader == 0  # returns current index without skipping

    def test_cordoned_shard_treated_as_non_joining(self):
        """Cordoned shards are already being evicted; health poller handles them.
        They should NOT be additionally skipped by the joining check."""
        gen = _make_gen_stub([
            {"status": "joining",  "worker_indices": [0]},
            {"status": "cordoned", "worker_indices": [1]},
        ])
        leader = gen._pick_ready_leader_idx()
        # Should skip joining dp-0 and dispatch to cordoned dp-1 (status != joining)
        assert leader == 1

    def test_round_robin_advances_correctly_after_skip(self):
        """After picking a ready shard that is NOT at start_idx, the index
        must be set to the picked shard so the subsequent round-robin +1 in
        _async_generate_base produces the correct next shard."""
        gen = _make_gen_stub([
            {"status": "joining", "worker_indices": [0]},
            {"status": "ready",   "worker_indices": [1]},
            {"status": "ready",   "worker_indices": [2]},
        ])
        leader1 = gen._pick_ready_leader_idx()
        # _async_generate_base increments by 1 after each dispatch
        gen.current_generate_dp_shard_idx = (gen.current_generate_dp_shard_idx + 1) % 3
        leader2 = gen._pick_ready_leader_idx()
        # First dispatch → dp-1 (leader=1), second → dp-2 (leader=2)
        assert leader1 == 1
        assert leader2 == 2

    def test_worker_indices_lookup_handles_tp_shards(self):
        """With TP>1 each shard has multiple worker indices.  The lookup
        must match the LEADER (first) index, not TP followers."""
        gen = _make_gen_stub([
            {"status": "joining", "worker_indices": [0, 1, 2, 3]},   # dp-0 TP=4, joining
            {"status": "ready",   "worker_indices": [4, 5, 6, 7]},   # dp-1 TP=4, ready
        ])
        leader = gen._pick_ready_leader_idx()
        # dp-0 leader=0, dp-1 leader=4
        assert leader == 4


# ---------------------------------------------------------------------------
# update_weights_from_collective: poisoned-comm reset on failure
# ---------------------------------------------------------------------------


class TestUpdateWeightsFromCollectivePoisonedCommReset:
    """When update_weights_from_collective returns success=False the method must
    reset the gen-side NCCL comms and invalidate _last_synced_world_size so the
    next ensure_collective_synced forces a fresh init_collective instead of
    reusing the poisoned comm.
    """

    def _make_vllm_gen_stub(self, router_result: dict) -> MagicMock:
        """Minimal VllmGeneration stub with a router that returns router_result."""
        from types import SimpleNamespace
        from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration

        router = MagicMock()
        router.call_async.return_value = router_result
        router.run_update_weights_from_collective.return_value = MagicMock()  # coroutine placeholder

        stub = SimpleNamespace(
            _router=router,
            _last_synced_world_size=48,
            _refitting=MagicMock(),  # threading.Event mock
        )

        reset_calls: list[str] = []

        def _reset_collective():
            reset_calls.append("reset")

        stub._reset_calls = reset_calls
        stub.reset_collective = _reset_collective

        # Bind the real method
        stub.update_weights_from_collective = (
            VllmGeneration.update_weights_from_collective.__get__(stub)
        )
        return stub

    def test_success_does_not_reset_or_invalidate(self):
        stub = self._make_vllm_gen_stub({"success": True, "promoted_shards": []})
        stub.update_weights_from_collective()
        assert stub._reset_calls == []
        assert stub._last_synced_world_size == 48  # unchanged

    def test_failure_calls_reset_collective(self):
        stub = self._make_vllm_gen_stub(
            {"success": False, "error": "3 of 4 workers failed weight broadcast; evicted 0"}
        )
        with pytest.raises(RuntimeError, match="update_weights_from_collective failed"):
            stub.update_weights_from_collective()
        assert stub._reset_calls == ["reset"]

    def test_failure_invalidates_last_synced_world_size(self):
        stub = self._make_vllm_gen_stub(
            {"success": False, "error": "3 of 4 workers failed"}
        )
        with pytest.raises(RuntimeError):
            stub.update_weights_from_collective()
        # Must be None so ensure_collective_synced forces a fresh init_collective.
        assert stub._last_synced_world_size is None

    def test_failure_raises_with_error_message(self):
        import re

        msg = "8 of 8 workers failed weight broadcast; evicted 1 (ids=['dp-1'])"
        stub = self._make_vllm_gen_stub({"success": False, "error": msg})
        with pytest.raises(RuntimeError, match=re.escape(msg)):
            stub.update_weights_from_collective()

    def test_reset_collective_exception_is_non_fatal(self):
        """Even if reset_collective itself raises, we still invalidate and re-raise."""
        stub = self._make_vllm_gen_stub({"success": False, "error": "broadcast failed"})
        stub.reset_collective = MagicMock(side_effect=RuntimeError("reset failed"))
        with pytest.raises(RuntimeError, match="update_weights_from_collective failed"):
            stub.update_weights_from_collective()
        assert stub._last_synced_world_size is None
