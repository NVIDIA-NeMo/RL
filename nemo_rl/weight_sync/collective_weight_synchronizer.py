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

"""NCCL collective weight synchronizer for non-colocated deployments.

Handles weight transfer between policy and generation workers running on
separate GPU clusters using NCCL collective communication. The policy
broadcasts its weights, and generation workers receive them via the
established NCCL process group.

Lifecycle per sync:
  1. policy.broadcast_weights_for_collective()    -- send via NCCL
     generation.update_weights_from_collective()  -- receive via NCCL
  2. Verify transfer success

No offload/restore steps are needed since policy and generation run on
separate GPUs with dedicated memory.
"""

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, Optional

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer
from nemo_rl.weight_sync.membership import RefitMembership, desired_membership


def _settle_before_propagating(futures, budget_s, what: str) -> None:
    """Let every rank finish unwinding before a refit failure reaches the caller.

    ``ray.get`` raises on the FIRST future that fails and leaves the rest running. That is
    fine when the caller is going to stop, and wrong when it is going to rebuild: a
    communicator rebuild is itself a collective, so dispatching ``init_collective`` while
    some ranks are still inside the old refit means they join late or not at all, and the
    rendezvous times out instead of coming up.

    Job 6512153 measured exactly that on the reshard kill variant. Rank 0 gave up on its
    own deadline, the controller went straight into the recovery, and the rebuild began --
    line 963 of the log -- two lines BEFORE rank 1's watchdog fired at all. The surviving
    generation worker then spent 300s twice failing to reach a store that never came up,
    and the run died at 690s having done everything else right.

    Bounded, and swallowing whatever the stragglers raise: they are unwinding from the same
    failure the caller is already holding, and replacing it with a straggler's version
    would lose the diagnosis. If the budget runs out, propagate anyway -- a caller stuck
    here would be a worse wedge than the one being recovered from.
    """
    if not futures:
        return
    try:
        ready, pending = ray.wait(
            list(futures), num_returns=len(futures), timeout=budget_s
        )
        if pending:
            print(
                f"  refit: {len(pending)} of {len(futures)} {what} rank(s) had not "
                f"unwound after {budget_s}s; rebuilding anyway",
                flush=True,
            )
    except Exception:  # noqa: BLE001 - the caller's failure is the one that matters
        pass


class CollectiveWeightSynchronizer(WeightSynchronizer):
    """Weight synchronizer using NCCL collectives for non-colocated deployments.

    Policy and generation workers run on separate GPU clusters. Weights are
    synchronized via NCCL broadcast over a pre-established process group.

    Args:
        policy: Policy object implementing ColocatablePolicyInterface.
        generation: Generation object implementing GenerationInterface.
        train_cluster: RayVirtualCluster for the training workers, used to
            obtain the master address/port and world size for collective init.
        inference_cluster: RayVirtualCluster for the inference workers.
    """

    def __init__(
        self,
        policy: Any,
        generation: Any,
        train_cluster: Any,
        inference_cluster: Any,
        refit_timeout_s: Optional[float] = None,
    ):
        # None disarms the abort watchdog in every worker, which is the default and
        # reproduces the pre-existing behaviour exactly.
        self._refit_timeout_s = refit_timeout_s
        self._policy = policy
        self._generation = generation
        self._train_cluster = train_cluster
        self._inference_cluster = inference_cluster
        self._stale = True
        # What the communicator was last built over. None until init_communicator.
        self._built_membership: Optional[RefitMembership] = None

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        timer_context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )
        with timer_context:
            sender_spec = self._generation.get_collective_sender_spec()
            futures_train = self._policy.broadcast_weights_for_collective(
                kv_scales=kv_scales,
                refit_timeout_s=self._refit_timeout_s,
                buffer_size_bytes=sender_spec.buffer_size_bytes,
                num_buffers=sender_spec.num_buffers,
            )
            futures_inference = self._generation.update_weights_from_collective(
                refit_timeout_s=self._refit_timeout_s
            )

            try:
                ray.get(futures_train)
            except BaseException:
                # Every train rank must be out of the old refit before the caller can
                # rebuild over the survivors; see _settle_before_propagating.
                _settle_before_propagating(
                    futures_train, self._settle_budget_s(), "train"
                )
                raise
            results = ray.get(futures_inference)
            update_success = all(result for result in results if result is not None)

            if not update_success:
                raise RuntimeError(
                    "Weight transfer failed during NCCL collective sync. "
                    "This often indicates an issue with the NCCL process group "
                    "or the generation backend worker."
                )

        self._stale = False

    @property
    def is_stale(self) -> bool:
        return self._stale

    def init_communicator(self) -> None:
        # prepare_refit_info is called before init_collective. This matches
        # distillation.py ordering. Neither call depends on the other today,
        # but we document this as the canonical ordering for future reference.
        state_dict_info = self._policy.prepare_refit_info()
        self._generation.prepare_refit_info(state_dict_info)

        ip, port = self._train_cluster.get_master_address_and_port()
        train_world_size = self._train_cluster.world_size()
        inference_world_size = self._generation.get_inference_world_size()
        if inference_world_size is None:
            inference_world_size = self._inference_cluster.world_size()
        world_size = train_world_size + inference_world_size

        sender_spec = self._generation.get_collective_sender_spec()
        futures_train = self._policy.init_collective(
            ip,
            port,
            world_size,
            train_world_size=train_world_size,
            nccl_peer=sender_spec.nccl_peer,
        )
        futures_inference = self._generation.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        ray.get(futures_train + futures_inference)
        # Recorded so the first reconcile can tell "unchanged" from "never built".
        self._built_membership = desired_membership(
            absent_shards=[],
            dp_size=self._generation.worker_group.dp_size,
            total_gen_workers=len(self._generation.worker_group.workers),
            train_world_size=train_world_size,
        )

    def _settle_budget_s(self) -> float:
        """How long to let stragglers unwind: their own deadline, plus a little.

        A rank that has not given up yet will do so when its watchdog fires, which is the
        same ``refit_timeout_s`` every rank was armed with. Without a configured deadline
        there is nothing bounding them, so fall back to a fixed wait rather than blocking
        the recovery indefinitely.
        """
        return (self._refit_timeout_s or 60.0) + 30.0

    def reconcile_communicator(
        self, absent_shards: Sequence[int], force: bool = False
    ) -> bool:
        """Rebuild the refit communicator over the surviving generation shards.

        ``model_update_group`` spans every train and inference rank and was built once,
        at setup, over the full fleet. The refit is a broadcast on that group, so a
        missing rank blocks it forever -- inside NCCL, where it produces no error and no
        progress while Ray still reports every actor healthy. Rebuilding without the dead
        ranks is what lets the run continue.

        Safe for the broadcast because rank 0 is a trainer and trainers are never
        excluded, so the root is stable across a rebuild and each receiver still slices
        the same byte stream locally.

        Rebuild rather than ``shrink``/``grow``. The pinned NCCL exports both -- checked
        against ``nccl.core.communicator.Communicator``, which also exports ``revoke``,
        ``suspend``, ``resume`` and ``split`` (2.28.9 exported only ``shrink``; uv.lock
        pins 2.30.7 and 2.30.4 in the dev image already has them). So this is a choice
        rather than a limitation: the
        nccl_reshard transport has to regenerate its refit plan on any membership change
        whatever NCCL supports, restore is dominated by the minutes an engine takes to
        reload, and one path shared with ``init_communicator`` is exercised by every
        normal run instead of only after a failure.
        """
        membership = desired_membership(
            absent_shards=absent_shards,
            dp_size=self._generation.worker_group.dp_size,
            total_gen_workers=len(self._generation.worker_group.workers),
            train_world_size=self._train_cluster.world_size(),
        )
        # Compared against what was built, not against "is anything absent". Keyed off
        # the absent set alone this would return False the moment a restarted shard came
        # back, leaving it permanently excluded from a communicator it should rejoin.
        #
        # An unrecorded membership means the full fleet: init_communicator builds over
        # everything, so "not recorded" is not "unknown", and treating it as a difference
        # would rebuild pointlessly on the very first refit of every run.
        if self._built_membership is None:
            self._built_membership = desired_membership(
                absent_shards=[],
                dp_size=self._generation.worker_group.dp_size,
                total_gen_workers=len(self._generation.worker_group.workers),
                train_world_size=membership.train_world_size,
            )
        # `force` is how the recovery path says the communicator is GONE rather than
        # merely unchanged: after an abort the membership is identical and the
        # communicator is dead, so skipping would retry over nothing.
        #
        # It only overrides the skip when something IS absent. Forcing a rebuild with
        # an empty absent set would produce a communicator that still contains the
        # rank that just went silent, and the retry would hang on it exactly as the
        # first attempt did. That case -- a frozen-but-alive rank, which never becomes
        # absent -- must fall through to False so the caller reports "no generation
        # shard could be identified as absent" and stops.
        unchanged = membership.shard_prefixes == self._built_membership.shard_prefixes
        if unchanged and not (force and absent_shards):
            return False

        # A fresh port every time: the rendezvous store for the previous world may still
        # be bound, and the cluster hands out a unique port per call for exactly this.
        ip, port = self._train_cluster.get_master_address_and_port()
        print(
            f"  refit: rebuilding communicator over shards "
            f"{membership.surviving_shards}; world_size {membership.world_size}, "
            f"port {port}",
            flush=True,
        )

        # Re-run for the whole fleet, not just for new shards. A restarted engine has no
        # state_dict_info at all -- update_weights_from_collective asserts on it -- and
        # this is metadata rather than weights, so redistributing it to shards that
        # already have it is cheap and removes the need to track who is new.
        state_dict_info = self._policy.prepare_refit_info()
        self._generation.prepare_refit_info(state_dict_info)

        # nccl_peer, exactly as init_communicator passes it. The receiver's bootstrap is
        # not negotiable: "nemo" publishes a raw unique ID and warms up with a rank-0
        # broadcast, "vllm" adds a pickled ID key and warms up with an all-reduce. Omitting
        # it here silently rebuilt with the "nemo" default, and mismatched warmups on one
        # communicator HANG rather than error -- the exact failure this path exists to
        # remove, reappearing inside the recovery.
        sender_spec = self._generation.get_collective_sender_spec()
        futures_train = self._policy.init_collective(
            ip,
            port,
            membership.world_size,
            train_world_size=membership.train_world_size,
            nccl_peer=sender_spec.nccl_peer,
        )
        # Recorded before dispatching, so nothing downstream can fall back to the old
        # membership. Rebuilding the communicator is only half of it: the refit dispatch
        # walks the worker group, so without this it keeps calling the dead shard's actor
        # and the next sync_weights fails with RayActorError -- the run still dies, just
        # later and with a less obvious cause.
        self._generation.set_refit_membership(membership)
        futures_inference = self._generation.rebuild_collective(membership, ip, port)
        ray.get(futures_train + futures_inference)
        self._built_membership = membership
        return True

    def shutdown(self) -> None:
        # The NCCL process group lifecycle is managed by Ray actor teardown.
        # Explicit destroy_process_group() is not needed here because the
        # workers that own the group are destroyed when the cluster shuts down.
        pass
