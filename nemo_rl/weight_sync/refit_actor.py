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
"""Run the weight synchronizer in its own Ray actor (async_rl.refit_actor).

The SingleController runs every refit fan-out from its own process: the calls to each
train worker and each generation leader, and the wait on all of their replies. That
process also dispatches rollouts and computes advantages, so a busy controller delays
the refit and a slow refit occupies the controller. ``RefitActor`` moves the
synchronizer into a separate CPU actor; ``RemoteWeightSynchronizer`` is the
``WeightSynchronizer`` the controller holds instead, forwarding each call.

What stays with the controller: the rollout gate, generation pause/resume, fleet health,
refit membership decisions, the refit deadline and recovery. Only the transport moves.

Two copies of the generation object exist once the synchronizer is in another process,
and two pieces of state must not drift between them:

- Worker handles. ``restart_shard`` replaces a shard's Ray actor in the controller's
  worker group; the actor's copy would keep the dead handle. Before every forwarded call
  the proxy compares the controller's worker handles with the ones it last sent and ships
  the generation object again when they differ.
- Refit membership. The synchronizer records which shards take part in refits with
  ``generation.set_refit_membership``, and the controller's own calls -- pause, resume,
  KV invalidation -- dispatch through that membership. Every call that can change it
  returns the membership the actor built, and the proxy applies it to the controller's
  copy.

The actor holds no NCCL communicator itself (those live in the train and generation
workers), only the refit plan and membership, so a RefitActor that dies is replaced from
the controller-side synchronizer and rebuilt by the next ``reconcile_communicator(force=
True)``, which the refit recovery path already issues.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import ray
from ray.exceptions import GetTimeoutError, RayActorError
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer

# Control calls (status, membership refresh, shutdown) are short; a RefitActor that does
# not answer one within this is treated as gone.
_CONTROL_CALL_TIMEOUT_S = 120.0


def _noop_shutdown() -> bool:
    return True


# VllmGeneration, Policy and RayVirtualCluster each define ``__del__`` -> ``shutdown()``
# as a safety net. Inside the RefitActor they are copies that hold handles to the real
# vLLM workers, train workers and placement groups, so a copy that is garbage collected
# (e.g. the old generation when ``update_generation`` replaces it) would shut those down
# -- and, through ``generation.weight_synchronizer`` (the proxy), ``ray.kill`` the
# RefitActor itself. The functions below give each such copy a no-op ``shutdown``
# instance attribute, which ``__del__`` then resolves to. Shutting the real workers
# down stays with the controller's own objects.


def _disarm_owner(owner: Any) -> None:
    """Disarm a generation or policy copy and the cluster its worker group holds."""
    owner.shutdown = _noop_shutdown
    owner.worker_group.cluster.shutdown = _noop_shutdown


def _disarm_generation(generation: Any) -> None:
    """Disarm the RefitActor's copy of the generation object."""
    _disarm_owner(generation)
    generation.weight_synchronizer = None


def _disarm_synchronizer(synchronizer: Any) -> None:
    """Disarm the copies an NCCL synchronizer brings into the RefitActor."""
    _disarm_generation(synchronizer._generation)
    _disarm_owner(synchronizer._policy)
    synchronizer._train_cluster.shutdown = _noop_shutdown
    synchronizer._inference_cluster.shutdown = _noop_shutdown


@dataclass
class RefitActorStatus:
    """What the controller reads back from the RefitActor between refits."""

    is_stale: bool
    # The RefitMembership the communicators were last built over, or None before the
    # first build.
    built_membership: Any


@ray.remote(num_cpus=1, max_concurrency=4)  # pragma: no cover
class RefitActor:
    """Owns the weight synchronizer and runs its calls in a separate process.

    ``max_concurrency`` > 1 so that ``status`` and ``reconcile_communicator`` still
    answer while a ``sync_weights`` that outlived the controller's deadline is stuck in
    one thread -- the same situation the in-process path leaves behind in an orphaned
    daemon thread.
    """

    def __init__(self, synchronizer: WeightSynchronizer):
        # Copies of the policy, generation and clusters arrive with the synchronizer;
        # none of them may shut the real ones down when collected.
        _disarm_synchronizer(synchronizer)
        self._sync = synchronizer

    def ping(self) -> bool:
        return True

    def update_generation(self, generation: Any) -> None:
        """Replace the synchronizer's generation object with the controller's current one."""
        # Disarm the new copy before the old one is dropped (both are copies here).
        _disarm_generation(generation)
        self._sync._generation = generation
        built = self._sync._built_membership
        if built is not None:
            generation.set_refit_membership(built)

    def init_communicator(self) -> Any:
        self._sync.init_communicator()
        return self._sync._built_membership

    def sync_weights(
        self, kv_scales: Optional[dict[str, float]]
    ) -> Optional[dict[str, float]]:
        return self._sync.sync_weights(kv_scales=kv_scales)

    def reconcile_communicator(
        self, absent_shards: Sequence[int], force: bool
    ) -> tuple[Optional[bool], Any]:
        rebuilt = self._sync.reconcile_communicator(absent_shards, force)
        return rebuilt, self._sync._built_membership

    def status(self) -> RefitActorStatus:
        return RefitActorStatus(
            is_stale=self._sync.is_stale,
            built_membership=self._sync._built_membership,
        )

    def shutdown(self) -> None:
        self._sync.shutdown()


class RemoteWeightSynchronizer(WeightSynchronizer):
    """Controller-side ``WeightSynchronizer`` that forwards to a ``RefitActor``.

    Args:
        synchronizer: an initialized in-process synchronizer
            (``init_communicator`` already called). It is shipped to the actor as-is,
            and kept here to start a replacement actor if the first one dies. It must
            carry ``_generation`` and ``_built_membership``, as the NCCL transports do.
        generation: the controller's generation object, the one ``synchronizer`` was
            built over. Its worker handles and refit membership are kept in step with
            the actor's copy.
    """

    def __init__(self, synchronizer: WeightSynchronizer, generation: Any):
        self._template: Optional[WeightSynchronizer] = synchronizer
        self._synchronizer_factory: Optional[Callable[[], WeightSynchronizer]] = None
        self._generation = generation
        self._actor = self._start_actor()
        self._sent_workers = self._worker_identity()

    def attach(
        self,
        generation: Any,
        synchronizer_factory: Optional[Callable[[], WeightSynchronizer]] = None,
    ) -> None:
        """Bind a proxy that was pickled into another process to that process's objects.

        Pickling keeps only the actor handle (see ``__getstate__``), so a proxy that
        crossed a process boundary -- the SingleControllerActor receives it inside its
        constructor arguments -- has no generation to keep in step and no template to
        start a replacement actor from. The new owner calls this once with its own
        generation object, and with a factory that builds a fresh, uninitialized
        synchronizer over it for replacing a dead RefitActor (the refit recovery that
        follows forces the rebuild).

        The generation is shipped to the actor again on the next forwarded call: its
        worker handles are the ones the actor already holds, but nothing here can prove
        that across processes, and one extra shipment is cheap.
        """
        self._generation = generation
        self._synchronizer_factory = synchronizer_factory
        self._sent_workers = ()

    # -- actor lifecycle ------------------------------------------------------------

    def _start_actor(self) -> Any:
        template = self._template
        if template is None:
            if self._synchronizer_factory is None:
                raise RuntimeError(
                    "Cannot start a RefitActor: this RemoteWeightSynchronizer was "
                    "unpickled and attach() was not given a synchronizer_factory."
                )
            template = self._synchronizer_factory()
        actor = RefitActor.options(
            # Beside the controller (unlike AdvantageActor's SPREAD): it only waits on
            # remote workers, so keep its round trips local. Soft: a full node can't block.
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(), soft=True
            ),
        ).remote(template)
        ray.get(actor.ping.remote(), timeout=_CONTROL_CALL_TIMEOUT_S)
        return actor

    def _actor_alive(self) -> bool:
        try:
            return ray.get(self._actor.ping.remote(), timeout=_CONTROL_CALL_TIMEOUT_S)
        except (RayActorError, GetTimeoutError):
            return False

    def _replace_dead_actor(self) -> None:
        """Start a new RefitActor if the current one is gone. No-op while it answers.

        The replacement starts from the controller-side synchronizer, whose recorded
        membership may predate the failure; the refit recovery path forces a rebuild
        right after, which brings the communicators and plan back in line.
        """
        if self._actor_alive():
            return
        print("  refit actor: RefitActor is gone; starting a replacement", flush=True)
        self._actor = self._start_actor()
        self._sent_workers = self._worker_identity()

    # -- controller/actor consistency ----------------------------------------------

    def _require_generation(self) -> Any:
        if self._generation is None:
            raise RuntimeError(
                "RemoteWeightSynchronizer has no generation object: it was unpickled in "
                "another process and attach() was not called there."
            )
        return self._generation

    def _worker_identity(self) -> tuple[int, ...]:
        # restart_shard swaps the handle object in place, so object identity is enough.
        generation = self._require_generation()
        return tuple(id(worker) for worker in generation.worker_group.workers)

    def _send_generation_if_changed(self) -> None:
        current = self._worker_identity()
        if current == self._sent_workers:
            return
        ray.get(
            self._actor.update_generation.remote(self._require_generation()),
            timeout=_CONTROL_CALL_TIMEOUT_S,
        )
        self._sent_workers = current

    def _mirror_membership(self, built: Any) -> None:
        if built is not None:
            self._require_generation().set_refit_membership(built)

    # -- WeightSynchronizer ---------------------------------------------------------

    def submit_sync_weights(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> ray.ObjectRef:
        """Start a refit in the actor and return its ObjectRef, for an async caller."""
        self._send_generation_if_changed()
        return self._actor.sync_weights.remote(kv_scales)

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> Optional[dict[str, float]]:
        # A Timer cannot be shared with another process; phases are timed by the caller.
        del timer
        return ray.get(self.submit_sync_weights(kv_scales))

    @property
    def is_stale(self) -> bool:
        self._replace_dead_actor()
        status = ray.get(self._actor.status.remote(), timeout=_CONTROL_CALL_TIMEOUT_S)
        return status.is_stale

    def init_communicator(self) -> None:
        self._send_generation_if_changed()
        built = ray.get(self._actor.init_communicator.remote())
        self._mirror_membership(built)

    def reconcile_communicator(
        self, absent_shards: Sequence[int], force: bool = False
    ) -> Optional[bool]:
        # Recovery calls this with force=True after a failed refit, which may have been
        # the RefitActor itself dying; replace it first so the rebuild has somewhere to run.
        self._replace_dead_actor()
        self._send_generation_if_changed()
        rebuilt, built = ray.get(
            self._actor.reconcile_communicator.remote(list(absent_shards), force)
        )
        self._mirror_membership(built)
        return rebuilt

    def shutdown(self) -> None:
        # More than one proxy can reach shutdown for the same actor (the controller's,
        # and the copy its generation object holds); after the first one, the actor is
        # already gone and there is nothing left to do.
        try:
            ray.get(self._actor.shutdown.remote(), timeout=_CONTROL_CALL_TIMEOUT_S)
        except RayActorError:
            return
        except GetTimeoutError:
            print(
                "  refit actor: shutdown did not finish within "
                f"{_CONTROL_CALL_TIMEOUT_S}s; killing the RefitActor",
                flush=True,
            )
        ray.kill(self._actor)

    # The generation object points back at this proxy (generation.weight_synchronizer),
    # so shipping the generation to the actor would otherwise pickle the controller-side
    # synchronizer, and through it the policy, with it. Only the actor handle travels.
    def __getstate__(self) -> dict[str, Any]:
        return {"_actor": self._actor}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._actor = state["_actor"]
        self._template = None
        self._synchronizer_factory = None
        self._generation = None
        self._sent_workers = ()
