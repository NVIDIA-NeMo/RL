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

from nemo_rl.weight_sync.factory import validate_refit_actor_transport
from nemo_rl.weight_sync.refit_actor import (
    RemoteWeightSynchronizer,
    _disarm_synchronizer,
)


def _fakes():
    """Fake generation + synchronizer, defined here so Ray ships them by value."""

    class FakeGeneration:
        def __init__(self, workers):
            self.worker_group = SimpleNamespace(
                workers=list(workers), cluster=SimpleNamespace()
            )
            self.membership = None
            self.weight_synchronizer = None

        def set_refit_membership(self, membership):
            self.membership = membership

    class FakeSynchronizer:
        def __init__(self, generation):
            self._generation = generation
            # The attributes the NCCL transports carry and the RefitActor disarms.
            self._policy = SimpleNamespace(
                worker_group=SimpleNamespace(cluster=SimpleNamespace())
            )
            self._train_cluster = self._policy.worker_group.cluster
            self._inference_cluster = generation.worker_group.cluster
            self._built_membership = ("built", tuple(generation.worker_group.workers))
            self._stale = True
            self.syncs = 0

        @property
        def is_stale(self):
            return self._stale

        @property
        def built_membership(self):
            return self._built_membership

        def rebind_generation(self, generation):
            self._generation = generation
            if self._built_membership is not None:
                generation.set_refit_membership(self._built_membership)

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


@pytest.mark.parametrize(
    ("backend", "refit_transport"),
    [("vllm", None), ("vllm", "nccl_reshard"), ("trtllm", None)],
)
def test_nccl_transports_are_accepted(backend, refit_transport):
    validate_refit_actor_transport(
        generation_backend=backend, colocated=False, refit_transport=refit_transport
    )


@pytest.mark.parametrize(
    ("backend", "refit_transport"),
    [
        ("vllm", "nixl"),
        ("vllm", "vllm_zmq_sparse"),
        ("megatron", None),
        ("megatron", "mcore"),
        ("sglang", None),
    ],
)
def test_non_nccl_transports_are_rejected(backend, refit_transport):
    with pytest.raises(ValueError, match="NCCL refit transports"):
        validate_refit_actor_transport(
            generation_backend=backend,
            colocated=False,
            refit_transport=refit_transport,
        )


def test_colocated_generation_is_rejected():
    with pytest.raises(ValueError, match="non-colocated"):
        validate_refit_actor_transport(
            generation_backend="vllm", colocated=True, refit_transport=None
        )


def _kill_and_wait(actor):
    ray.kill(actor)
    # ray.kill is asynchronous; wait until the actor is really gone.
    with pytest.raises(ray.exceptions.RayActorError):
        for _ in range(100):
            ray.get(actor.ping.remote(), timeout=30)


def test_a_dead_refit_actor_is_a_clear_error(remote):
    proxy, _ = remote
    _kill_and_wait(proxy._actor)
    with pytest.raises(RuntimeError, match="RefitActor died"):
        proxy.reconcile_communicator([], force=True)
    with pytest.raises(RuntimeError, match="RefitActor died"):
        proxy.sync_weights()
    with pytest.raises(RuntimeError, match="RefitActor died"):
        _ = proxy.is_stale


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

    class SelfShuttingPolicy:
        def __init__(self):
            self.worker_group = SimpleNamespace(cluster=FakeCluster())
            self.marker_dir = marker_dir

        def shutdown(self):
            open(
                os.path.join(self.marker_dir, f"policy-{os.getpid()}-{id(self)}"), "w"
            ).close()
            return True

        def __del__(self):
            self.shutdown()

    class SelfShuttingGeneration:
        def __init__(self, workers):
            self.worker_group = SimpleNamespace(
                workers=list(workers), cluster=FakeCluster()
            )
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

    def make_synchronizer(generation):
        """A synchronizer whose policy, generation and clusters all shut down on GC."""
        _, FakeSynchronizer = _fakes()
        synchronizer = FakeSynchronizer(generation)
        synchronizer._policy = SelfShuttingPolicy()
        synchronizer._train_cluster = synchronizer._policy.worker_group.cluster
        synchronizer._inference_cluster = generation.worker_group.cluster
        return synchronizer

    return SelfShuttingGeneration, make_synchronizer


def test_resending_generation_never_shuts_anything_down(tmp_path):
    # Re-shipping the generation drops the actor's previous copy. Its __del__ ->
    # shutdown() must not stop the real workers or ray.kill the RefitActor itself.
    SelfShuttingGeneration, make_synchronizer = _self_shutting_fakes(str(tmp_path))
    generation = SelfShuttingGeneration(["w0", "w1"])
    proxy = RemoteWeightSynchronizer(make_synchronizer(generation), generation)
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


def test_disarm_synchronizer_covers_every_self_shutting_copy(tmp_path):
    SelfShuttingGeneration, make_synchronizer = _self_shutting_fakes(str(tmp_path))
    generation = SelfShuttingGeneration(["w0"])
    synchronizer = make_synchronizer(generation)
    generation.weight_synchronizer = object()
    _disarm_synchronizer(synchronizer)
    assert generation.weight_synchronizer is None
    for copy in (
        generation,
        generation.worker_group.cluster,
        synchronizer._policy,
        synchronizer._policy.worker_group.cluster,
        synchronizer._train_cluster,
        synchronizer._inference_cluster,
    ):
        copy.shutdown()
    assert list(tmp_path.iterdir()) == []


def test_disarm_synchronizer_fails_on_a_missing_attribute():
    # A renamed attribute must not leave a copy armed without a signal.
    FakeGeneration, FakeSynchronizer = _fakes()
    synchronizer = FakeSynchronizer(FakeGeneration(["w0"]))
    del synchronizer._train_cluster
    with pytest.raises(AttributeError, match="_train_cluster"):
        _disarm_synchronizer(synchronizer)


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
