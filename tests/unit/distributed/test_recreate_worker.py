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

"""Rebuilding one worker in place.

This is how a dead generation shard comes back, and it is replay rather than
reconstruction: the arguments the original creation call was made with are recorded and
replayed, so a replacement cannot drift from the original by re-deriving the placement
group, bundle or venv differently.

What is pinned here is that replay: same bundle, same placement group, old actor killed
first, name made unique. Whether a real vLLM engine then comes up on that bundle needs
GPUs and is covered by tests/functional/grpo_sc_generation_shard_recovery.sh.
"""

from types import SimpleNamespace

import pytest

from nemo_rl.distributed.worker_groups import RayWorkerGroup


class _Initializer:
    def __init__(self):
        self.calls = []

    class _CreateWorker:
        def __init__(self, outer):
            self._outer = outer

        def remote(self, pg, bundle_idx, num_gpus, bundle_indices, **kwargs):
            self._outer.calls.append(
                {
                    "pg": pg,
                    "bundle_idx": bundle_idx,
                    "num_gpus": num_gpus,
                    "bundle_indices": bundle_indices,
                    **kwargs,
                }
            )
            return f"new-worker-{len(self._outer.calls)}"

    @property
    def create_worker(self):
        return _Initializer._CreateWorker(self)


def _group(num_workers=2, killable=True):
    group = object.__new__(RayWorkerGroup)
    group._workers = [f"old-worker-{i}" for i in range(num_workers)]
    group._worker_metadata = []
    group._worker_incarnations = {}
    group.cluster = SimpleNamespace(num_gpus_per_node=8)
    initializer = _Initializer()
    group._initializer_pool = {0: initializer}
    group._worker_specs = {
        i: {
            "pg_idx": 0,
            "pg": f"placement-group-{i}",
            "bundle_idx": i,
            "num_gpus": 1.0,
            "worker_bundle_indices": (0, [i]),
            "extra_options": {"name": f"vllm_policy-0-{i}", "runtime_env": {"x": i}},
        }
        for i in range(num_workers)
    }
    return group, initializer


@pytest.fixture(autouse=True)
def _no_ray_kill(monkeypatch):
    killed = []
    monkeypatch.setattr(
        "ray.kill", lambda actor, no_restart=False: killed.append(actor)
    )
    monkeypatch.setattr("ray.get", lambda ref: ref)
    return killed


class TestReplay:
    def test_the_replacement_reuses_the_original_bundle(self):
        """The dead actor's placement-group bundle stays reserved, so the replacement
        lands on the same GPU. Choosing a different bundle would either fail on
        resources or silently move the shard to another device."""
        group, initializer = _group()

        group.recreate_worker(1)

        call = initializer.calls[0]
        assert call["bundle_idx"] == 1
        assert call["pg"] == "placement-group-1"
        assert call["bundle_indices"] == (0, [1])

    def test_the_replacement_keeps_the_original_resources_and_runtime_env(self):
        group, initializer = _group()

        group.recreate_worker(0)

        call = initializer.calls[0]
        assert call["num_gpus"] == 1.0
        assert call["runtime_env"] == {"x": 0}
        assert call["num_gpus_per_node"] == 8

    def test_the_new_handle_replaces_the_old_one_in_place(self):
        group, _ = _group()

        returned = group.recreate_worker(1)

        assert group.workers[1] == returned
        assert group.workers[1] != "old-worker-1"
        assert group.workers[0] == "old-worker-0", "other workers must be untouched"


class TestOldActorIsRemoved:
    def test_the_old_actor_is_killed_before_the_replacement_is_made(self, _no_ray_kill):
        """An unresponsive-but-alive worker still holds the GPU; creating its
        replacement first would fail on memory rather than on anything informative."""
        group, _ = _group()

        group.recreate_worker(1)

        assert _no_ray_kill == ["old-worker-1"]

    def test_an_unkillable_actor_does_not_block_the_replacement(self, monkeypatch):
        """Usually it is already dead, and ray.kill raises variously for that."""
        group, initializer = _group()
        monkeypatch.setattr(
            "ray.kill",
            lambda actor, no_restart=False: (_ for _ in ()).throw(
                ValueError("actor already dead")
            ),
        )

        group.recreate_worker(1)

        assert len(initializer.calls) == 1


class TestActorNaming:
    def test_each_incarnation_gets_a_distinct_name(self):
        """Ray rejects a duplicate named actor, and a dead one's registration can
        outlive its process -- so reusing the name makes the second restart fail."""
        group, initializer = _group()

        group.recreate_worker(1)
        group.recreate_worker(1)
        group.recreate_worker(1)

        names = [call["name"] for call in initializer.calls]
        assert len(set(names)) == 3, names
        assert all(name.startswith("vllm_policy-0-1") for name in names)

    def test_the_original_spec_is_not_mutated_by_renaming(self):
        """Otherwise names compound across restarts: ...-r1-r2-r3."""
        group, initializer = _group()

        group.recreate_worker(1)
        group.recreate_worker(1)

        assert group._worker_specs[1]["extra_options"]["name"] == "vllm_policy-0-1"
        assert initializer.calls[1]["name"].count("-r") == 1


class TestRejected:
    def test_an_unknown_worker_is_refused(self):
        group, _ = _group(num_workers=2)

        with pytest.raises(KeyError, match="no creation spec"):
            group.recreate_worker(9)
