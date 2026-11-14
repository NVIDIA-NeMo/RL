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
"""RefitActor / RemoteWeightSynchronizer on a local CPU Ray cluster (no GPUs)."""

import gc
import os
from types import SimpleNamespace

import pytest
import ray
from ray import cloudpickle

from nemo_rl.algorithms.single_controller_utils.config import (
    AsyncRLConfig,
    RefitActorConfig,
)
from nemo_rl.weight_sync.refit_actor import RemoteWeightSynchronizer, _disarm_copies


def _fakes():
    """Fake generation + synchronizer, defined here so Ray ships them by value."""

    class FakeGeneration:
        def __init__(self, workers):
            self.worker_group = SimpleNamespace(workers=list(workers))
            self.membership = None
            self.weight_synchronizer = None

        def set_refit_membership(self, membership):
            self.membership = membership

    class FakeSynchronizer:
        def __init__(self, generation):
            self._generation = generation
            self._built_membership = ("built", tuple(generation.worker_group.workers))
            self._stale = True
            self.syncs = 0

        @property
        def is_stale(self):
            return self._stale

        def init_communicator(self):
            self._built_membership = (
                "init",
                tuple(self._generation.worker_group.workers),
            )

        def sync_weights(self, *, timer=None, kv_scales=None):
            self.syncs += 1
            self._stale = False
            # What the actor-side copy of the generation looks like right now.
            return {
                "workers": list(self._generation.worker_group.workers),
                "syncs": self.syncs,
                "kv_scales": kv_scales,
            }

        def reconcile_communicator(self, absent_shards, force=False):
            self._built_membership = (
                "rebuilt",
                tuple(absent_shards),
                tuple(self._generation.worker_group.workers),
            )
            return True

        def shutdown(self):
            self._generation = None

    return FakeGeneration, FakeSynchronizer


@pytest.fixture
def remote():
    FakeGeneration, FakeSynchronizer = _fakes()
    generation = FakeGeneration(["w0", "w1"])
    synchronizer = FakeSynchronizer(generation)
    proxy = RemoteWeightSynchronizer(synchronizer, generation)
    generation.weight_synchronizer = proxy
    yield proxy, generation
    proxy.shutdown()


def test_config_default_is_off():
    assert RefitActorConfig().enabled is False
    assert AsyncRLConfig().refit_actor.enabled is False
    assert AsyncRLConfig(refit_actor={"enabled": True}).refit_actor.enabled is True


def test_sync_weights_runs_in_actor(remote):
    proxy, _ = remote
    assert proxy.is_stale is True
    result = proxy.sync_weights(kv_scales={"layer0": 1.0})
    assert result["syncs"] == 1
    assert result["kv_scales"] == {"layer0": 1.0}
    assert proxy.is_stale is False


def test_submit_sync_weights_returns_awaitable_ref(remote):
    proxy, _ = remote
    ref = proxy.submit_sync_weights(None)
    assert isinstance(ref, ray.ObjectRef)
    assert ray.get(ref)["syncs"] == 1


def test_reconcile_mirrors_membership_to_controller(remote):
    proxy, generation = remote
    assert proxy.reconcile_communicator([1], force=True) is True
    assert generation.membership == ("rebuilt", (1,), ("w0", "w1"))


def test_restarted_shard_handle_reaches_actor(remote):
    proxy, generation = remote
    # restart_shard swaps the handle object in place.
    generation.worker_group.workers[1] = "w1-restarted"
    result = proxy.sync_weights()
    assert result["workers"] == ["w0", "w1-restarted"]


def test_dead_actor_is_replaced_on_reconcile(remote):
    proxy, generation = remote
    first = proxy._actor
    ray.kill(first)
    # ray.kill is asynchronous; wait until the actor is really gone.
    with pytest.raises(ray.exceptions.RayActorError):
        for _ in range(100):
            ray.get(first.ping.remote(), timeout=30)
    assert proxy.reconcile_communicator([], force=True) is True
    assert proxy._actor is not first
    assert generation.membership == ("rebuilt", (), ("w0", "w1"))
    assert proxy.sync_weights()["syncs"] == 1


def test_setup_rejects_non_nccl_transport():
    # Deferred: setup pulls in the whole controller stack, which only this test needs.
    from nemo_rl.algorithms.single_controller_utils.setup import (
        _move_synchronizer_to_refit_actor,
    )

    FakeGeneration, FakeSynchronizer = _fakes()
    generation = FakeGeneration(["w0"])
    with pytest.raises(ValueError, match="NCCL refit transports"):
        _move_synchronizer_to_refit_actor(FakeSynchronizer(generation), generation)


def _roundtrip(obj):
    """Pickle as Ray does when the proxy travels inside the controller's actor args."""

    return cloudpickle.loads(cloudpickle.dumps(obj))


def test_unpickled_proxy_needs_attach(remote):
    proxy, _ = remote
    copy = _roundtrip(proxy)
    # Pickling keeps only the actor handle; using the copy before attach() must say
    # so instead of failing on a missing generation.
    with pytest.raises(RuntimeError, match="attach"):
        copy.sync_weights()


def test_attached_copy_syncs_and_ships_its_generation(remote):
    FakeGeneration, _ = _fakes()
    proxy, _ = remote
    copy = _roundtrip(proxy)
    controller_generation = FakeGeneration(["w0", "w1-controller"])
    copy.attach(controller_generation)
    result = copy.sync_weights()
    # The actor now works with the generation of the process that attached.
    assert result["workers"] == ["w0", "w1-controller"]
    assert copy.reconcile_communicator([0], force=True) is True
    assert controller_generation.membership == (
        "rebuilt",
        (0,),
        ("w0", "w1-controller"),
    )


def test_attached_copy_replaces_dead_actor_from_factory(remote):
    FakeGeneration, FakeSynchronizer = _fakes()
    proxy, _ = remote
    copy = _roundtrip(proxy)
    controller_generation = FakeGeneration(["w0", "w1"])
    built = []

    def factory():
        built.append(1)
        return FakeSynchronizer(controller_generation)

    copy.attach(controller_generation, synchronizer_factory=factory)
    first = copy._actor
    ray.kill(first)
    with pytest.raises(ray.exceptions.RayActorError):
        for _ in range(100):
            ray.get(first.ping.remote(), timeout=30)
    assert copy.reconcile_communicator([], force=True) is True
    assert built == [1]
    assert copy._actor is not first
    assert copy.sync_weights()["syncs"] == 1
    copy.shutdown()


def test_unpickled_copy_without_factory_cannot_replace_actor(remote):
    FakeGeneration, _ = _fakes()
    proxy, _ = remote
    copy = _roundtrip(proxy)
    copy.attach(FakeGeneration(["w0"]))
    first = copy._actor
    ray.kill(first)
    with pytest.raises(ray.exceptions.RayActorError):
        for _ in range(100):
            ray.get(first.ping.remote(), timeout=30)
    with pytest.raises(RuntimeError, match="synchronizer_factory"):
        copy.reconcile_communicator([], force=True)


def _self_shutting_fakes(marker_dir):
    """Fakes that, like VllmGeneration/RayVirtualCluster, shut down from __del__.

    shutdown() leaves a file in ``marker_dir`` so a call made inside the RefitActor's
    process is visible to the test.
    """

    class FakeCluster:
        def __init__(self):
            self.marker_dir = marker_dir

        def shutdown(self):
            open(
                os.path.join(self.marker_dir, f"cluster-{os.getpid()}-{id(self)}"), "w"
            ).close()
            return True

        def __del__(self):
            self.shutdown()

    class SelfShuttingGeneration:
        def __init__(self, workers):
            self.worker_group = SimpleNamespace(workers=list(workers))
            self.cluster = FakeCluster()
            self.membership = None
            self.weight_synchronizer = None
            self.marker_dir = marker_dir

        def set_refit_membership(self, membership):
            self.membership = membership

        def shutdown(self):
            open(
                os.path.join(self.marker_dir, f"gen-{os.getpid()}-{id(self)}"), "w"
            ).close()
            if self.weight_synchronizer is not None:
                self.weight_synchronizer.shutdown()
            return True

        def __del__(self):
            self.shutdown()

    return SelfShuttingGeneration


def test_resending_generation_never_shuts_anything_down(tmp_path):
    # Re-shipping the generation drops the actor's previous copy. Its __del__ ->
    # shutdown() must not stop the real workers or ray.kill the RefitActor itself.
    _, FakeSynchronizer = _fakes()
    SelfShuttingGeneration = _self_shutting_fakes(str(tmp_path))
    generation = SelfShuttingGeneration(["w0", "w1"])
    proxy = RemoteWeightSynchronizer(FakeSynchronizer(generation), generation)
    generation.weight_synchronizer = proxy
    copy = _roundtrip(proxy)
    copy.attach(generation)
    for restarted in ("w1-a", "w1-b", "w1-c"):
        generation.worker_group.workers[1] = restarted
        assert copy.sync_weights()["workers"] == ["w0", restarted]
        gc.collect()
    assert copy.is_stale is False  # the actor is still alive
    actor_side = [m for m in os.listdir(tmp_path) if f"-{os.getpid()}-" not in m]
    assert actor_side == [], f"copies shut down inside the RefitActor: {actor_side}"
    generation.weight_synchronizer = None
    proxy.shutdown()


def test_disarm_copies_reaches_nested_objects_and_skips_handles(tmp_path):
    SelfShuttingGeneration = _self_shutting_fakes(str(tmp_path))

    class HandleLike:  # like ray.actor.ActorHandle: __del__, methods via __getattr__
        __module__ = "ray.actor"

        def __getattr__(self, name):
            return name

        def __del__(self):
            pass

    generation = SelfShuttingGeneration(["w0"])
    generation.handles = [HandleLike()]
    generation.weight_synchronizer = object()
    holder = SimpleNamespace(_generation=generation)
    assert _disarm_copies(holder) == 2  # the generation and its nested cluster
    assert generation.weight_synchronizer is None
    assert "shutdown" not in vars(generation.handles[0])
    generation.shutdown()
    generation.cluster.shutdown()
    assert list(tmp_path.iterdir()) == []


def test_second_shutdown_of_the_same_actor_is_quiet(remote, capsys):
    # The controller's proxy and the copy its generation object holds both shut the
    # same RefitActor down at the end of a run.
    proxy, _ = remote
    copy = _roundtrip(proxy)
    copy.shutdown()
    with pytest.raises(ray.exceptions.RayActorError):
        for _ in range(100):
            ray.get(proxy._actor.ping.remote(), timeout=30)
    proxy.shutdown()
    assert "refit actor" not in capsys.readouterr().out
