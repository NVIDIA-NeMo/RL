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
from nemo_rl.weight_sync.membership import plan_refit_membership


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
        refit_timeout_s: Deadline for one refit collective. Each participating worker
            arms a watchdog and aborts its own communicator when it expires, which is
            what lets the controller rebuild over the survivors instead of blocking in
            NCCL forever. ``None`` disarms it entirely, so the hang protection is lost.
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
        # The absent set this synchronizer's current communicator was built with, so a
        # membership that has not changed can skip the rebuild. None means "never rebuilt",
        # i.e. still the full-fleet group from setup.
        #
        # Without this, reconcile_communicator rebuilt on EVERY call once a shard was gone,
        # because absent_shards() never empties again -- nothing in production calls
        # mark_restarting or mark_loaded. _sync_weights reconciles twice per step, so a run
        # that lost a shard at step 10 and trains to 10,000 paid ~20,000 full rebuilds: a
        # fresh port, a fresh TCPStore and a fresh NCCL bootstrap across every train and
        # inference rank each time, plus a plan regeneration on nccl_reshard. The steady
        # state this feature exists to produce was the expensive one.
        self._built_with_absent: Optional[frozenset[int]] = None

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
                # Every rank must be out of the old refit before the caller can rebuild
                # over the survivors; see _settle_before_propagating. BOTH sides: the
                # rebuild dispatches init_collective to the generation ranks too, and
                # ray.get(futures_train) raising leaves futures_inference running. Settling
                # only the train half is the same half-applied fix as the widened
                # signatures in design_vllm_fault_tolerance.md section 8.5.5.
                _settle_before_propagating(
                    futures_train, self._settle_budget_s(), "train"
                )
                _settle_before_propagating(
                    futures_inference, self._settle_budget_s(), "generation"
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
        if not absent_shards:
            return False

        # Unchanged membership over a live communicator: nothing to do. `force` is how the
        # recovery path says the communicator is gone rather than merely unchanged -- after
        # an abort it must be rebuilt even though the absent set is identical, and skipping
        # it there would fail the recovery with "no shard could be identified as absent".
        if not force and self._built_with_absent == frozenset(absent_shards):
            return False

        dp_size = self._generation.worker_group.dp_size
        surviving = [idx for idx in range(dp_size) if idx not in set(absent_shards)]
        membership = plan_refit_membership(
            surviving_shards=surviving,
            dp_size=dp_size,
            total_gen_workers=len(self._generation.worker_group.workers),
            train_world_size=self._train_cluster.world_size(),
        )

        # A fresh port every time: the rendezvous store for the previous world may still
        # be bound, and the cluster hands out a unique port per call for exactly this.
        ip, port = self._train_cluster.get_master_address_and_port()
        print(
            f"  refit: rebuilding communicator without shards {sorted(absent_shards)}; "
            f"world_size {membership.world_size}, port {port}",
            flush=True,
        )

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
        # Recorded only after the rebuild has actually happened, so a rebuild that
        # raises leaves the cache describing the communicator we still have.
        self._built_with_absent = frozenset(absent_shards)
        return True

    def shutdown(self) -> None:
        # The NCCL process group lifecycle is managed by Ray actor teardown.
        # Explicit destroy_process_group() is not needed here because the
        # workers that own the group are destroyed when the cluster shuts down.
        pass
