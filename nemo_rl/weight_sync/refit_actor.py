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
"""Run the weight synchronizer in its own Ray actor (``async_rl.refit_actor``).

``RefitActor`` holds the synchronizer in a separate CPU process.
``RemoteWeightSynchronizer`` is the controller's proxy: it forwards each call and keeps
worker handles and refit membership in step with the actor's copy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import ray
from ray.exceptions import GetTimeoutError, RayActorError
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer

if TYPE_CHECKING:
    from nemo_rl.weight_sync.collective_weight_synchronizer import (
        CollectiveWeightSynchronizer,
    )
    from nemo_rl.weight_sync.membership import RefitMembership
    from nemo_rl.weight_sync.nccl_reshard_weight_synchronizer import (
        NcclReshardWeightSynchronizer,
    )

    NcclWeightSynchronizer = (
        CollectiveWeightSynchronizer | NcclReshardWeightSynchronizer
    )

# A RefitActor that does not answer a control call within this is treated as gone.
_CONTROL_CALL_TIMEOUT_S = 120.0


def _noop_shutdown() -> bool:
    return True


# VllmGeneration, Policy and RayVirtualCluster shut their real workers and placement
# groups down from ``__del__``. Inside the RefitActor they are copies, so each one gets
# a no-op ``shutdown`` that ``__del__`` resolves to instead.


def _disarm_owner(owner: Any) -> None:
    """Disarm a generation or policy copy and the cluster its worker group holds."""
    owner.shutdown = _noop_shutdown
    owner.worker_group.cluster.shutdown = _noop_shutdown


def _disarm_generation(generation: Any) -> None:
    """Disarm the RefitActor's copy of the generation object."""
    _disarm_owner(generation)
    generation.weight_synchronizer = None


def _disarm_synchronizer(synchronizer: NcclWeightSynchronizer) -> None:
    """Disarm the copies an NCCL synchronizer brings into the RefitActor."""
    _disarm_generation(synchronizer._generation)
    _disarm_owner(synchronizer._policy)
    synchronizer._train_cluster.shutdown = _noop_shutdown
    synchronizer._inference_cluster.shutdown = _noop_shutdown


@dataclass
class RefitActorStatus:
    """What the controller reads back from the RefitActor between refits."""

    is_stale: bool
    built_membership: Optional[RefitMembership]


@ray.remote(num_cpus=1, max_concurrency=4)  # pragma: no cover
class RefitActor:
    """Owns the weight synchronizer; ``status`` answers while a refit is stuck."""

    def __init__(self, synchronizer: NcclWeightSynchronizer):
        _disarm_synchronizer(synchronizer)
        self._sync = synchronizer

    def ping(self) -> bool:
        return True

    def update_generation(self, generation: Any) -> None:
        # Disarm the new copy before the synchronizer drops the old one.
        _disarm_generation(generation)
        self._sync.rebind_generation(generation)

    def init_communicator(self) -> Optional[RefitMembership]:
        self._sync.init_communicator()
        return self._sync.built_membership

    def sync_weights(
        self, kv_scales: Optional[dict[str, float]]
    ) -> Optional[dict[str, float]]:
        return self._sync.sync_weights(kv_scales=kv_scales)

    def reconcile_communicator(
        self, absent_shards: Sequence[int], force: bool
    ) -> tuple[Optional[bool], Optional[RefitMembership]]:
        rebuilt = self._sync.reconcile_communicator(absent_shards, force)
        return rebuilt, self._sync.built_membership

    def status(self) -> RefitActorStatus:
        return RefitActorStatus(
            is_stale=self._sync.is_stale,
            built_membership=self._sync.built_membership,
        )

    def shutdown(self) -> None:
        self._sync.shutdown()


class RemoteWeightSynchronizer(WeightSynchronizer):
    """Controller-side ``WeightSynchronizer`` that forwards each call to a ``RefitActor``.

    Args:
        synchronizer: an initialized NCCL synchronizer (the config check guarantees
            one), shipped to the actor.
        generation: the controller's generation object that ``synchronizer`` was built over.
    """

    def __init__(self, synchronizer: WeightSynchronizer, generation: Any):
        self._generation = generation
        self._actor = RefitActor.options(
            # Beside the controller (unlike AdvantageActor's SPREAD): it only waits on
            # remote workers, so keep its round trips local. Soft: a full node can't block.
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(), soft=True
            ),
        ).remote(synchronizer)
        self._get(self._actor.ping.remote())
        self._sent_workers = self._worker_identity()

    def attach(self, generation: Any) -> None:
        """Bind a proxy unpickled in another process to that process's generation."""
        self._generation = generation
        # Ship it on the next call: the handles can't be compared across processes.
        self._sent_workers = ()

    # -- actor calls ---------------------------------------------------------------

    def raise_if_actor_died(self, error: BaseException) -> None:
        """Raise RuntimeError if the RefitActor is gone; a worker's death passes through."""
        try:
            ray.get(self._actor.ping.remote(), timeout=_CONTROL_CALL_TIMEOUT_S)
        except (RayActorError, GetTimeoutError):
            raise RuntimeError(
                "the RefitActor died, so this refit cannot be recovered; check the "
                "controller node for an OOM or node loss."
            ) from error

    def _get(
        self, ref: ray.ObjectRef, timeout: Optional[float] = _CONTROL_CALL_TIMEOUT_S
    ) -> Any:
        try:
            return ray.get(ref, timeout=timeout)
        except RayActorError as error:
            self.raise_if_actor_died(error)
            raise

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
        self._get(self._actor.update_generation.remote(self._require_generation()))
        self._sent_workers = current

    def _mirror_membership(self, built: Optional[RefitMembership]) -> None:
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
        return self._get(self.submit_sync_weights(kv_scales), timeout=None)

    @property
    def is_stale(self) -> bool:
        return self._get(self._actor.status.remote()).is_stale

    def init_communicator(self) -> None:
        self._send_generation_if_changed()
        built = self._get(self._actor.init_communicator.remote(), timeout=None)
        self._mirror_membership(built)

    def reconcile_communicator(
        self, absent_shards: Sequence[int], force: bool = False
    ) -> Optional[bool]:
        self._send_generation_if_changed()
        rebuilt, built = self._get(
            self._actor.reconcile_communicator.remote(list(absent_shards), force),
            timeout=None,
        )
        self._mirror_membership(built)
        return rebuilt

    def shutdown(self) -> None:
        # The controller's proxy and the copy its generation holds both reach this;
        # after the first, the actor is already gone.
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

    # The generation points back at this proxy, so only the actor handle is pickled;
    # otherwise shipping the generation would drag the policy along.
    def __getstate__(self) -> dict[str, Any]:
        return {"_actor": self._actor}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._actor = state["_actor"]
        self._generation = None
        self._sent_workers = ()
