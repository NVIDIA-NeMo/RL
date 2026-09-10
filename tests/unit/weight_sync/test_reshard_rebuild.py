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

"""Recovering the nccl_reshard transport, and the refit dispatch both transports share.

Two failures are pinned here, both silent rather than loud:

* The refit plan is a function of the inference world size. Reusing one built for the
  old fleet does not error -- a stale mesh is still a valid mesh -- it just has survivors
  writing the slices the dead shard owned and leaving their own unwritten.
* Rebuilding the communicator is only half a recovery. The refit *dispatch* walks the
  worker group, so without excluding the dead shard the next sync still calls its actor.
"""

from types import SimpleNamespace

import pytest

from nemo_rl.weight_sync.membership import plan_refit_membership
from nemo_rl.weight_sync.nccl_reshard_weight_synchronizer import (
    NcclReshardWeightSynchronizer,
)


class _Worker:
    def __init__(self, idx, *, dead=False):
        self.idx = idx
        self.dead = dead
        self.calls = []

    class _M:
        def __init__(self, worker, name):
            self._w, self._n = worker, name

        def remote(self, **kwargs):
            if self._w.dead:
                raise AssertionError(
                    f"worker {self._w.idx} is gone; calling it is what the rebuild "
                    "exists to stop"
                )
            self._w.calls.append((self._n, kwargs))
            return f"f-{self._w.idx}-{self._n}"

    def __getattr__(self, name):
        if name.startswith(("init_", "prepare_", "nccl_reshard", "update_weights")):
            return _Worker._M(self, name)
        raise AttributeError(name)


def _reshard(dp_size=4, workers_per_shard=1, dead_shards=(), train_world_size=8):
    """Fleet is built ALIVE. `kill()` on the returned handle stages the death, matching
    the real order: init_communicator runs over a whole fleet, a shard dies later."""
    workers = [
        _Worker(shard * workers_per_shard + w)
        for shard in range(dp_size)
        for w in range(workers_per_shard)
    ]
    pending_dead = set(dead_shards)
    from nemo_rl.models.generation.vllm import vllm_generation

    gen = SimpleNamespace(
        cfg={"vllm_cfg": {"async_engine": False}},
        worker_group=SimpleNamespace(workers=workers, dp_size=dp_size),
        dp_size=dp_size,
        # Declared, because VllmGeneration declares it. It used to be read with a getattr
        # default, which meant this fake could omit it and still pass -- the getattr was
        # covering for the fake rather than for a genuinely optional field.
        _refit_membership=None,
        # Same principle: the rebuild asks which bootstrap protocol the receiver speaks,
        # so a stand-in has to answer. Omitting it is how the reshard rebuild went on
        # passing its tests while silently defaulting to "nemo".
        get_collective_sender_spec=lambda: SimpleNamespace(nccl_peer="nemo"),
    )
    for name in (
        "rebuild_collective",
        "rebuild_nccl_reshard_comm_group",
        "prepare_nccl_reshard_refit_info",
        "set_refit_membership",
        "nccl_reshard_refit",
        "_refit_leader_workers",
    ):
        setattr(
            gen,
            name,
            (
                lambda n: lambda *a, **k: getattr(vllm_generation.VllmGeneration, n)(
                    gen, *a, **k
                )
            )(name),
        )

    plan_calls = []
    policy = SimpleNamespace(
        cfg={
            "megatron_cfg": {
                "tensor_model_parallel_size": 1,
                "expert_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
            },
            "generation": {"vllm_cfg": {"tensor_parallel_size": 1}},
        },
        init_collective=lambda *a, **k: ["train-f"],
        init_nccl_reshard_comm_group=lambda **k: ["train-bulk"],
        prepare_nccl_reshard_refit_info=lambda tp, gp, tws, iws: (
            plan_calls.append({"train_world_size": tws, "gen_world_size": iws})
            or {"gen_world_size": iws}
        ),
    )
    ports = iter(range(9001, 9200))
    train_cluster = SimpleNamespace(
        world_size=lambda: train_world_size,
        num_gpus_per_node=8,
        get_master_address_and_port=lambda: ("10.0.0.1", next(ports)),
        get_available_address_and_port=lambda pg_idx, bundle_idx: (
            "10.0.0.1",
            next(ports),
        ),
    )
    sync = NcclReshardWeightSynchronizer(
        policy=policy,
        generation=gen,
        train_cluster=train_cluster,
        inference_cluster=SimpleNamespace(
            world_size=lambda: dp_size * workers_per_shard
        ),
    )

    def kill():
        for shard in pending_dead:
            for w in range(workers_per_shard):
                workers[shard * workers_per_shard + w].dead = True

    return sync, gen, workers, plan_calls, kill


@pytest.fixture(autouse=True)
def _no_ray(monkeypatch):
    monkeypatch.setattr("ray.get", lambda futures: futures)


class TestPlanRegeneration:
    def test_the_plan_is_rebuilt_for_the_smaller_fleet(self):
        """The whole reason reshard could not simply resize."""
        sync, _, _, plan_calls, kill = _reshard(dp_size=4, dead_shards=(2,))
        sync.init_communicator()
        kill()
        assert plan_calls[-1]["gen_world_size"] == 4

        sync.reconcile_communicator([2])

        assert plan_calls[-1]["gen_world_size"] == 3, (
            "a plan sized for the old fleet would have survivors writing the dead "
            "shard's slices and leaving their own unwritten, with no error"
        )

    def test_the_plan_accounts_for_shard_width(self):
        sync, _, _, plan_calls, kill = _reshard(
            dp_size=4, workers_per_shard=2, dead_shards=(0,)
        )
        sync.init_communicator()
        kill()

        sync.reconcile_communicator([0])

        assert plan_calls[-1]["gen_world_size"] == 6

    def test_trainers_are_untouched_by_a_rebuild(self):
        sync, _, _, plan_calls, kill = _reshard(dead_shards=(1,), train_world_size=16)
        sync.init_communicator()
        kill()

        sync.reconcile_communicator([1])

        assert plan_calls[-1]["train_world_size"] == 16


class TestBothCommunicatorFamilies:
    def test_the_dead_shard_receives_neither_family(self):
        sync, _, workers, _, kill = _reshard(dp_size=4, dead_shards=(3,))
        sync.init_communicator()  # whole fleet, including shard 3
        kill()
        for w in workers:
            w.calls.clear()  # otherwise shard 3's pre-death calls mask the result

        # _Worker raises if touched, so surviving this is the assertion.
        assert sync.reconcile_communicator([3]) is True
        assert [w.idx for w in workers if w.calls] == [0, 1, 2]

    def test_survivors_get_both_the_shared_group_and_the_bulk_group(self):
        sync, _, workers, _, kill = _reshard(dp_size=3, dead_shards=(1,))
        sync.init_communicator()
        kill()
        for w in workers:
            w.calls.clear()

        sync.reconcile_communicator([1])

        for w in workers:
            if w.idx == 1:
                continue
            called = {name for name, _ in w.calls}
            assert "init_collective" in called
            assert "init_nccl_reshard_comm_group" in called

    def test_the_bulk_group_world_shrinks_with_the_fleet(self):
        """sub_world_size = train_ranks_per_stage + inference_world_size."""
        sync, _, workers, _, kill = _reshard(
            dp_size=4, dead_shards=(0,), train_world_size=8
        )
        sync.init_communicator()
        kill()
        for w in workers:
            w.calls.clear()

        sync.reconcile_communicator([0])

        bulk = [
            kwargs
            for w in workers
            for name, kwargs in w.calls
            if name == "init_nccl_reshard_comm_group"
        ]
        assert bulk and all(k["sub_world_size"] == 8 + 3 for k in bulk)


class TestRefitDispatchExcludesTheDeadShard:
    """The half of recovery that rebuilding the communicator does not cover.

    Without this the communicator is correct and the very next refit still calls the
    dead actor, so the run dies with RayActorError instead of continuing.
    """

    def test_reshard_refit_skips_the_dead_shard(self):
        sync, gen, workers, _, kill = _reshard(dp_size=4, dead_shards=(2,))
        sync.init_communicator()
        kill()
        sync.reconcile_communicator([2])
        for w in workers:
            w.calls.clear()

        gen.nccl_reshard_refit()

        assert [w.idx for w in workers if w.calls] == [0, 1, 3]

    def test_collective_refit_skips_the_dead_shard(self):
        from nemo_rl.models.generation.vllm import vllm_generation

        sync, gen, workers, _, kill = _reshard(dp_size=4, dead_shards=(1,))
        gen.update_weights_from_collective = (
            lambda: vllm_generation.VllmGeneration.update_weights_from_collective(gen)
        )
        gen.set_refit_membership(
            plan_refit_membership(
                surviving_shards=[0, 2, 3],
                dp_size=4,
                total_gen_workers=4,
                train_world_size=8,
            )
        )

        gen.update_weights_from_collective()

        assert [w.idx for w in workers if w.calls] == [0, 2, 3]

    def test_every_shard_is_addressed_before_any_loss(self):
        """The default path, which is the whole life of a run that never loses a shard."""
        sync, gen, workers, _, kill = _reshard(dp_size=3)

        gen.nccl_reshard_refit()

        assert [w.idx for w in workers if w.calls] == [0, 1, 2]


class TestTheRefitDeadlineReachesThisTransport:
    """The abort machinery was wired to the collective path only.

    factory.py passed refit_timeout_s to CollectiveWeightSynchronizer and constructed
    NcclReshardWeightSynchronizer without it, so this transport ran with no deadline at
    all: a shard dying inside a reshard refit wedged exactly as before the watchdog
    existed. Nothing indicated it -- the recovery-reshard test kills at a step boundary,
    so it passed via the between-refits path and never touched the missing machinery.
    """

    def test_both_sides_are_given_the_deadline(self):
        sync, gen, workers, _, _ = _reshard(dp_size=2)
        sync._refit_timeout_s = 60.0
        sync.init_communicator()
        for w in workers:
            w.calls.clear()
        train_kwargs = {}
        sync._policy.nccl_reshard_refit = lambda **k: (
            train_kwargs.update(k) or ["train-f"]
        )

        sync.sync_weights()

        assert train_kwargs.get("refit_timeout_s") == 60.0, (
            "the producer side must be able to abort its own transfer"
        )
        refits = [c for w in workers for c in w.calls if c[0] == "nccl_reshard_refit"]
        assert refits, "no generation worker was asked to refit"
        for _name, kwargs in refits:
            assert kwargs.get("refit_timeout_s") == 60.0

    def test_an_unset_deadline_still_reaches_both_sides_as_none(self):
        """Default-off must mean None arrives, not that the argument is absent.

        Ray validates at dispatch, so an entrypoint that never receives the keyword is
        indistinguishable here from one that cannot accept it -- until a real run.
        """
        sync, gen, workers, _, _ = _reshard(dp_size=2)
        sync.init_communicator()
        for w in workers:
            w.calls.clear()
        sync._policy.nccl_reshard_refit = lambda **k: ["train-f"]

        sync.sync_weights()

        refits = [c for w in workers for c in w.calls if c[0] == "nccl_reshard_refit"]
        assert refits, "no generation worker was asked to refit"
        for _name, kwargs in refits:
            assert kwargs.get("refit_timeout_s", "MISSING") is None
