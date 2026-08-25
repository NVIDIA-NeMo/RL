# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""A shard the fleet has written off must actually be killed, not just relabelled.

Job 6521181: both trainers' py-spy dumps sat in ``init_nccl_communicator`` with the
abandoned ``ncclCommAbort`` threads still stuck in native code 25 minutes later. A frozen
rank holds its sockets open, so its peers' aborts never retire, so NCCL cannot bootstrap the
replacement communicator. Marking it DEAD in the ledger changes none of that.
"""

from types import SimpleNamespace

import pytest

from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.models.generation.fleet_health import (
    FleetHealthPolicy,
    GenerationFleetHealth,
)


class _Worker:
    def __init__(self, idx):
        self.idx = idx


def _controller(*, shard_count=2, workers_per_shard=1, killed=None):
    ctrl = object.__new__(SingleControllerActor.__ray_metadata__.modified_class)
    ctrl._gen_fleet = GenerationFleetHealth(
        shard_count=shard_count,
        policy=FleetHealthPolicy(),
        base_urls=[f"http://shard{i}:8000/v1" for i in range(shard_count)],
    )
    ctrl._evicted_shards = set()
    total = shard_count * workers_per_shard
    ctrl._gen = SimpleNamespace(
        worker_group=SimpleNamespace(
            workers=[_Worker(i) for i in range(total)],
            dp_leader_worker_indices=[
                i * workers_per_shard for i in range(shard_count)
            ],
        )
    )
    return ctrl


@pytest.fixture
def killed(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "nemo_rl.algorithms.single_controller.ray.kill",
        lambda worker, no_restart=False: seen.append((worker.idx, no_restart)),
    )
    return seen


def test_an_absent_shard_is_killed_so_its_peers_aborts_can_retire(killed):
    ctrl = _controller()
    ctrl._evict_absent_but_alive({0})
    assert killed == [(0, True)], (
        "the absent shard's worker must be killed with no_restart; leaving it frozen is "
        "what blocked the rebuild's NCCL bootstrap in job 6521181"
    )


def test_every_worker_of_the_shard_is_killed_not_just_its_leader(killed):
    """tp/pp > 1 puts several workers behind one dp shard, and one survivor is enough.

    The frozen rank blocks its peers by holding sockets, so killing 3 of 4 leaves the
    fourth doing exactly what the whole eviction exists to stop.
    """
    ctrl = _controller(shard_count=2, workers_per_shard=3)
    ctrl._evict_absent_but_alive({1})
    assert [w for w, _ in killed] == [3, 4, 5]


def test_a_shard_that_stays_absent_is_evicted_once(killed):
    ctrl = _controller()
    for _ in range(4):
        ctrl._evict_absent_but_alive({0})
    assert killed == [(0, True)]


def test_a_healthy_fleet_kills_nothing(killed):
    ctrl = _controller()
    ctrl._evict_absent_but_alive(set())
    assert killed == []


def test_a_failed_kill_does_not_break_the_recovery(monkeypatch):
    """The eviction must never be the reason a recovery fails.

    A worker that is already gone raises from ray.kill, and that is the outcome this
    wanted -- so it has to be swallowed, and the shard still marked evicted.
    """

    def _boom(worker, no_restart=False):
        raise ValueError("actor already dead")

    monkeypatch.setattr("nemo_rl.algorithms.single_controller.ray.kill", _boom)
    ctrl = _controller()
    ctrl._evict_absent_but_alive({0})
    assert ctrl._evicted_shards == {0}


def test_the_eviction_runs_before_the_rebuild_not_after():
    """Ordering is the whole point: the rebuild is what the survivors are blocked in.

    Evicting afterwards would leave the sockets open for exactly the call that cannot
    proceed while they are, which is the failure this fixes.
    """
    import ast
    import inspect
    import textwrap

    cls = SingleControllerActor.__ray_metadata__.modified_class
    src = textwrap.dedent(inspect.getsource(cls._reconcile_refit_membership))
    # By position, not call order: reconcile_communicator is handed to asyncio.to_thread
    # rather than called, so it is an Attribute in the args and never an ast.Call func.
    where = {
        node.attr: node.lineno
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Attribute)
        and node.attr in ("_evict_absent_but_alive", "reconcile_communicator")
    }
    assert "_evict_absent_but_alive" in where, (
        "_reconcile_refit_membership must evict absent-but-alive shards"
    )
    assert "reconcile_communicator" in where, (
        "_reconcile_refit_membership must still drive the communicator rebuild"
    )
    assert where["_evict_absent_but_alive"] < where["reconcile_communicator"], (
        "the eviction must run before the communicator rebuild, not after it"
    )
