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

import time
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Optional

import ray

from nemo_rl.models.generation.interfaces import reject_unenforceable_refit_deadline
from nemo_rl.models.megatron import sm_partition
from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.collective_weight_synchronizer import (
    CollectiveWeightSynchronizer,
)
from nemo_rl.weight_sync.interfaces import WeightSynchronizer
from nemo_rl.weight_sync.nccl_reshard_weight_synchronizer import (
    NcclReshardWeightSynchronizer,
)


class MegatronWeightSynchronizer(WeightSynchronizer):
    """Weight synchronization for the Megatron generation backend, both colocation modes.

    Colocated is the degenerate path: generation either aliases the training
    weights outright (reshardless) or re-partitions them into the worker's
    dedicated inference model inside ``prepare_for_generation`` (when the
    configured inference layout/impl differs) — a genuine parallelism-changing
    transfer, but one the worker performs internally on wake. Sync therefore
    reduces to dropping training-only buffers and re-entering inference mode.

    Non-colocated generation keeps the Megatron engine lifecycle here and
    delegates the transfer to native MCore refit, packed collective, or M2N.
    """

    def __init__(
        self,
        policy: Any,
        generation: Any,
        *,
        colocated: bool,
        train_cluster: Optional[Any] = None,
        inference_cluster: Optional[Any] = None,
        refit_timeout_s: Optional[float] = None,
        offload_policy_before_refit: bool = False,
    ) -> None:
        if not colocated and (train_cluster is None or inference_cluster is None):
            raise ValueError(
                "train_cluster and inference_cluster are required for "
                "non-colocated Megatron weight synchronization."
            )
        if not colocated and generation.uses_native_refit:
            # Native MCore's copy service does not expose an abortable collective.
            # Reject during setup, before either side can enter a refit.
            reject_unenforceable_refit_deadline("native MCore", refit_timeout_s)
        self._policy = policy
        self._generation = generation
        self._colocated = colocated
        self._train_cluster = train_cluster
        self._inference_cluster = inference_cluster
        self._refit_timeout_s = refit_timeout_s
        self._offload_policy_before_refit = offload_policy_before_refit
        self._refit_backend: Optional[str] = None
        self._transport: Optional[WeightSynchronizer] = None
        if colocated:
            # Colocated refit always uses the in-place wake-reshard, so
            # any other transport is inert. Reject rather than silently ignoring it:
            # a user asking for packed collective refit would otherwise get the
            # native one with no indication their setting did nothing.
            if not generation.uses_native_refit:
                raise ValueError(
                    "policy.generation.refit_transport must be 'mcore' with "
                    "colocated Megatron generation, which always uses the in-place "
                    "wake-reshard. Set colocated.enabled=false to use the packed "
                    "collective or nccl_reshard transport."
                )
        if not colocated and not generation.uses_native_refit:
            if generation.cfg.get("refit_transport") == "ipc":
                self._transport = CoresidentRefit(
                    policy=policy, generation=generation, refit_timeout_s=refit_timeout_s
                )
            elif generation.cfg.get("refit_transport") == "nccl_reshard":
                self._transport = NcclReshardWeightSynchronizer(
                    policy=policy,
                    generation=generation,
                    train_cluster=train_cluster,
                    inference_cluster=inference_cluster,
                    refit_timeout_s=refit_timeout_s,
                )
            else:
                self._transport = CollectiveWeightSynchronizer(
                    policy=policy,
                    generation=generation,
                    train_cluster=train_cluster,
                    inference_cluster=inference_cluster,
                    refit_timeout_s=refit_timeout_s,
                )
        self._stale = True

    def init_communicator(self) -> None:
        """Wire the cross-group refit collective (non-colocated only).

        Colocated generation shares the training worker group, so there is
        nothing to wire.
        """
        if self._colocated:
            return
        if self._transport is not None:
            self._transport.init_communicator()
            return
        ip, port = self._train_cluster.get_master_address_and_port()
        print(f"Using ip: {ip}, port: {port} for collective communication", flush=True)
        train_world_size = self._train_cluster.world_size()
        world_size = train_world_size + self._inference_cluster.world_size()
        self._refit_backend = self._generation.cfg["mcore_generation_config"][
            "refit_backend"
        ]
        refit_execution_batch_bytes = self._generation.cfg["mcore_generation_config"][
            "refit_execution_batch_bytes"
        ]
        futures_train = self._policy.init_collective_mcore_generation(
            ip,
            port,
            world_size,
            rank_offset=0,
            refit_execution_batch_bytes=refit_execution_batch_bytes,
            refit_backend=self._refit_backend,
        )
        futures_inference = self._generation.init_collective(
            ip,
            port,
            world_size,
            train_world_size=train_world_size,
        )
        ray.get(futures_train + futures_inference)

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> Optional[dict[str, float]]:
        def timed_phase(name: str) -> AbstractContextManager[None]:
            return timer.time(name) if timer is not None else nullcontext()

        if self._colocated:
            # The wake below carries any configured reshard; the loop already slept the engine
            # before training, so no suspend is needed.
            # Tagging the call bypasses the worker's engine-awake early-return, so the reshard
            # copy riding this wake cannot be skipped. Any tag except "weights" works: the worker
            # treats "weights" as the wake-suppressing mid-refit call.
            with timed_phase("prepare_for_generation/offload_policy"):
                self._policy.offload_before_refit()
            with timed_phase("prepare_for_generation/prepare_weights"):
                self._generation.prepare_for_generation(tags=["colocated_refit"])
            self._stale = False
            return {}

        if isinstance(self._transport, CoresidentRefit):
            # Co-resident engine: pause at a step boundary, same-GPU copy, epoch bump, unpause (no suspend, no
            # conversion, no collectives).
            with timed_phase("prepare_for_generation/transfer_and_update_weights"):
                stats = self._transport.sync_weights(kv_scales=kv_scales)
            self._stale = False
            return stats

        # The engine serves continuously in non-colocated mode; pause it exactly
        # around the swap.
        with timed_phase("prepare_for_generation/suspend_for_refit"):
            self._generation.suspend_for_refit()
        if self._offload_policy_before_refit:
            with timed_phase("prepare_for_generation/offload_policy"):
                self._policy.offload_before_refit()
        with timed_phase("prepare_for_generation/prepare_weights"):
            self._generation.prepare_for_generation(tags=["weights"])

        if self._refit_backend == "nvshmem":
            with timed_phase("prepare_for_generation/preinit_nvshmem"):
                futures_train = self._policy.preinit_nvshmem()
                futures_inference = self._generation.preinit_nvshmem_collective()
                ray.get(futures_train + futures_inference)

        timer_context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )
        with timer_context:
            if self._transport is not None:
                self._transport.sync_weights(kv_scales=kv_scales)
            else:
                # Dispatch the destination first: unsupported deadlines are rejected
                # before any source rank can enter the blocking native transfer.
                futures_inference = self._generation.update_weights_from_collective(
                    refit_timeout_s=self._refit_timeout_s
                )
                futures_train = self._policy.swap_weights_via_reshard(is_source=True)
                ray.get(futures_train)
                results = ray.get(futures_inference)
                if not all(result for result in results if result is not None):
                    raise RuntimeError(
                        "❌ Error: Updating weights for the generation policy failed "
                        "during refit.\nThis often indicates an issue with the "
                        "refit copy service or a problem within the generation "
                        "backend.\n"
                    )

        with timed_phase("prepare_for_generation/prepare_kv_cache"):
            self._generation.prepare_for_generation(tags=["kv_cache"])
        with timed_phase("prepare_for_generation/resume_after_refit"):
            self._generation.resume_after_refit()
        self._stale = False
        return {}

    def before_learner_write(self) -> None:
        if isinstance(self._transport, CoresidentRefit):
            self._transport.before_learner_write()

    @property
    def is_stale(self) -> bool:
        return self._stale

    def shutdown(self) -> None:
        """Release any resources owned by the delegated transport."""
        if self._transport is not None:
            self._transport.shutdown()


def _get_fail_fast(refs: list, *, timeout_s: Optional[float] = None) -> list:
    """Return worker results or raise the first failure without waiting for peers.

    ray.get that raises the first failure as soon as it lands (a peer of a failed call may otherwise block until its
    own timeout). O(1) driver calls while nothing fails.
    """
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("co-resident refit timeout must be positive")
    deadline = None if timeout_s is None else time.perf_counter() + timeout_s
    pending = list(refs)
    while pending:
        wait_s = 1.0
        if deadline is not None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"co-resident refit: {len(pending)} worker call(s) did not return within {timeout_s}s")
            wait_s = min(wait_s, remaining)
        done, pending = ray.wait(pending, num_returns=len(pending), timeout=wait_s)
        ray.get(done)
    return ray.get(list(refs))


class CoresidentRefit:
    """Refit of a co-resident Megatron engine (nemo_rl/models/megatron/sm_partition.py).

    Setup: every learner rank exports CUDA IPC handles of its local refit tensors once; the engine rank on the same GPU
    maps them (and, with ``engine_weights=shared``, binds the matching ones as its own, before it captures its graphs).

    Per step, ``sync_weights`` (the loop's refit, after its parameter all-gather): pause every engine at a step boundary
    unless ``before_learner_write`` already did; check that the learners' writes are complete and their storage
    unmoved; copy the non-shared tensors, refresh derived weights and bump the weight epoch on every engine rank; then
    unpause. With shared weights the engine reads the learner's memory, so it is paused from before the optimizer step
    writes it until this refit publishes the new version.

    Actor calls use the configured refit timeout, or a 360-second fallback, so an unresponsive worker fails the run.
    """

    def __init__(self, policy: Any, generation: Any, *, refit_timeout_s: Optional[float] = None) -> None:
        self._policy = policy
        self._generation = generation
        self._fingerprints: list[str] = []
        self._shared = sm_partition.engine_weights_mode(policy.cfg) == "shared"
        self._engine_running = False
        self._paused_at: Optional[float] = None
        self._pause_s = 0.0
        self._control_timeout_s = 360.0 if refit_timeout_s is None else refit_timeout_s

    def init_communicator(self) -> None:
        exported = _get_fail_fast(
            self._policy.worker_group.run_all_workers_single_data("export_coresident_refit_handles"),
            timeout_s=self._control_timeout_s,
        )
        by_device = {dev: (handles, coords) for dev, handles, _, coords in exported}
        if len(by_device) != len(exported):
            raise RuntimeError("co-resident refit: two learner ranks report the same GPU")
        self._fingerprints = [fp for _, _, fp, _ in exported]
        bound = _get_fail_fast(
            self._generation.worker_group.run_all_workers_single_data(
                "open_coresident_refit_handles", handles_by_device=by_device, share=self._shared
            ),
            timeout_s=self._control_timeout_s,
        )
        print(f"[sm_partition] co-resident engine weights ({'shared' if self._shared else 'copy'}) bound: {bound[0]}",
              flush=True)

    def _pause(self) -> float:
        t0 = time.perf_counter()
        _get_fail_fast(
            self._generation.worker_group.run_all_workers_single_data("coresident_pause"),
            timeout_s=self._control_timeout_s,
        )
        self._paused_at = t0
        return time.perf_counter() - t0

    def before_learner_write(self) -> None:
        if self._shared and self._engine_running:
            self._pause_s = self._pause()

    def sync_weights(self, *, kv_scales: Optional[dict[str, float]] = None) -> dict[str, float]:
        if kv_scales:
            raise NotImplementedError("co-resident refit with FP8 KV scales is not built yet")
        if self._paused_at is None:
            self._pause_s = self._pause()
        gen = self._generation.worker_group
        t0 = time.perf_counter()
        ready = _get_fail_fast(
            self._policy.worker_group.run_all_workers_single_data("coresident_refit_source_ready"),
            timeout_s=self._control_timeout_s,
        )
        if ready != self._fingerprints:
            raise RuntimeError(
                "co-resident refit: a learner rank's refit tensors moved since their IPC handles were taken "
                "(offload / reload / buffer rebuild); the engine would read freed memory"
            )
        t1 = time.perf_counter()
        copies = _get_fail_fast(
            gen.run_all_workers_single_data("coresident_refit_copy"), timeout_s=self._control_timeout_s
        )
        t2 = time.perf_counter()
        _get_fail_fast(gen.run_all_workers_single_data("coresident_resume"), timeout_s=self._control_timeout_s)
        t3 = time.perf_counter()
        stats = {
            "coresident_pause_s": self._pause_s,
            "coresident_source_check_s": t1 - t0,
            "coresident_copy_s": t2 - t1,
            "coresident_copy_s_max_rank": max(c["seconds"] for c in copies),
            "coresident_unpause_s": t3 - t2,
            "coresident_engine_paused_s": t3 - self._paused_at,
            "coresident_copy_bytes_per_rank": copies[0]["bytes"],
            "coresident_shared_bytes_per_rank": copies[0]["bytes_shared"],
        }
        print(f"[sm_partition] refit: {stats}", flush=True)
        self._paused_at = None
        self._engine_running = True
        return stats

    def shutdown(self) -> None:
        pass
