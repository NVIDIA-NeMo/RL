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

"""Refit protocol over a fleet of NeMo RL Megatron engines behind Dynamo.

Each engine is one ``MegatronGeneration`` (its own worker group, DP = 1). The
channel fans the controller's single calls out over the engines' Ray workers in
record order and reproduces, per engine, the sequence the Megatron weight
synchronizer uses for one dedicated engine: suspend, prepare the weights,
transfer, prepare the KV cache, resume. Rank placement comes from the shared
``RefitMembership`` arithmetic: engine ``i`` starts at the trainers plus the
widths of the engines before it, so a prefill engine at TP 4 may sit next to
decode engines at TP 2.
"""

import dataclasses
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import ray
from ray.actor import ActorHandle

from nemo_rl.models.generation.dynamo.protocols import DynamoWorkerRecord
from nemo_rl.weight_sync.membership import RefitMembership, desired_membership

if TYPE_CHECKING:
    from nemo_rl.models.generation.megatron.megatron_generation import (
        MegatronGeneration,
    )


@dataclasses.dataclass(frozen=True)
class MegatronDynamoWorkerClient:
    """Non-owning handle on one Megatron engine's Ray workers.

    Holds only actor handles. ``MegatronGeneration`` (through its ``Policy``)
    and the ``RayVirtualCluster`` behind its worker group both shut the engine
    down when garbage collected, so any process that unpickled either and then
    dropped it would kill the engine. This handle never does; the worker pool
    that built the ``MegatronGeneration`` owns its lifetime.

    Attributes:
        workers: Every worker of the engine, in rank order.
        leader: The worker leading the engine's single DP shard.
    """

    workers: tuple[ActorHandle, ...]
    leader: ActorHandle

    @classmethod
    def of(cls, generation: "MegatronGeneration") -> "MegatronDynamoWorkerClient":
        """The workers of a dedicated, DP = 1 ``MegatronGeneration``."""
        worker_group = generation.worker_group
        workers = tuple(worker_group.workers)
        return cls(
            workers=workers, leader=workers[worker_group.get_dp_leader_worker_idx(0)]
        )

    def prepare_refit_info(self, state_dict_info: Any) -> list[ray.ObjectRef]:
        return [
            worker.prepare_refit_info.remote(state_dict_info=state_dict_info)
            for worker in self.workers
        ]

    def init_collective(
        self,
        ip: str,
        port: int,
        world_size: int,
        *,
        train_world_size: int,
        rank_offset: int,
    ) -> list[ray.ObjectRef]:
        return [
            worker.init_collective.remote(
                ip=ip,
                port=port,
                world_size=world_size,
                train_world_size=train_world_size,
                rank_offset=rank_offset,
            )
            for worker in self.workers
        ]

    def update_weights_from_collective(
        self, refit_timeout_s: float | None
    ) -> list[ray.ObjectRef]:
        return [
            worker.update_weights_from_collective.remote(
                refit_timeout_s=refit_timeout_s
            )
            for worker in self.workers
        ]

    def pause_for_refit(self) -> list[ray.ObjectRef]:
        return [worker.pause_for_refit.remote() for worker in self.workers]

    def suspend_for_refit(self) -> list[ray.ObjectRef]:
        return [worker.suspend_for_refit.remote() for worker in self.workers]

    def resume_after_refit(self) -> list[ray.ObjectRef]:
        return [worker.resume_after_refit.remote() for worker in self.workers]

    def prepare_for_generation(self, *, tags: list[str]) -> list[ray.ObjectRef]:
        return [
            worker.prepare_for_generation.remote(tags=tags) for worker in self.workers
        ]

    def pinned_handoff_block_count(self) -> ray.ObjectRef:
        return self.leader.pinned_handoff_block_count.remote()


class PinnedHandoffTimeout(TimeoutError):
    """Prefill engines still held handoff blocks when the pause wait expired."""


def _seconds_left(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


class MegatronRefitChannel:
    """``RefitChannel`` implementation for Megatron engines.

    Construction swaps each record's ``MegatronGeneration`` for a
    :class:`MegatronDynamoWorkerClient`, so the channel holds only Ray actor handles:
    every copy, including the SingleController actor's, drives the engines
    without owning them.

    Every control call is bounded: :meth:`pause` and :meth:`resume` by their
    ``timeout_s`` (``control_timeout_s`` when None), one deadline per call, and
    :meth:`prepare_refit_info` by ``control_timeout_s``.
    """

    def __init__(
        self,
        records: Sequence[DynamoWorkerRecord],
        *,
        control_timeout_s: float,
        idle_poll_interval_s: float = 0.2,
    ) -> None:
        if not records:
            raise ValueError("Megatron refit requires at least one engine record")
        for record in records:
            if record.world_size < 1:
                raise ValueError(
                    f"engine {record.instance_id} ({record.role}) has "
                    f"{record.world_size} ranks; every engine must span at least one"
                )
        self._records = tuple(
            dataclasses.replace(
                record, control=MegatronDynamoWorkerClient.of(record.control)
            )
            for record in records
        )
        # Each engine's width is its record's; prefill and decode may differ.
        self._shard_widths = tuple(record.world_size for record in records)
        self._control_timeout_s = control_timeout_s
        self._idle_poll_interval_s = idle_poll_interval_s

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def records(self) -> tuple[DynamoWorkerRecord, ...]:
        """The engine records, each ``control`` a :class:`MegatronDynamoWorkerClient`."""
        return self._records

    @property
    def inference_world_size(self) -> int:
        return sum(self._shard_widths)

    @property
    def num_engines(self) -> int:
        return len(self._records)

    @property
    def shard_widths(self) -> tuple[int, ...]:
        """Ranks each engine spans, in record (refit shard) order."""
        return self._shard_widths

    def client_copy(self) -> "MegatronRefitChannel":
        """The channel holds only actor handles and owns nothing; it is its own copy."""
        return self

    @staticmethod
    def _client(record: DynamoWorkerRecord) -> MegatronDynamoWorkerClient:
        return record.control

    def _prefill_records(self) -> list[DynamoWorkerRecord]:
        return [record for record in self._records if record.role == "prefill"]

    def _non_prefill_records(self) -> list[DynamoWorkerRecord]:
        return [record for record in self._records if record.role != "prefill"]

    @classmethod
    def _call_engines(
        cls,
        records: Sequence[DynamoWorkerRecord],
        call: Callable[[MegatronDynamoWorkerClient], list[ray.ObjectRef]],
        *,
        what: str,
        deadline: float,
    ) -> None:
        """Run ``call`` on every engine at once; raise the first failure.

        The failure keeps its type (callers recover from ``RayActorError``) and
        gains a note naming the engine and the step.
        """
        pending = [(record, call(cls._client(record))) for record in records]
        for record, futures in pending:
            try:
                ray.get(futures, timeout=_seconds_left(deadline))
            except Exception as error:
                error.add_note(f"{what} failed on engine {record.instance_id}")
                raise

    # ------------------------------------------------------------------
    # Refit collective
    # ------------------------------------------------------------------

    def check_full_fleet(self, membership: RefitMembership) -> None:
        """Reject any membership but every engine, in order: the fleet is fixed.

        Engine ``i``'s prefix is the sum of the widths before it, and its width
        must be the one its record reports.
        """
        full_fleet: dict[int, int] = {}
        prefix = 0
        for index, width in enumerate(self._shard_widths):
            full_fleet[index] = prefix
            prefix += width
        if (
            membership.shard_prefixes != full_fleet
            or membership.workers_per_shard != self._shard_widths
        ):
            raise NotImplementedError(
                "Megatron engines behind Dynamo are a fixed fleet; refit membership "
                f"must be every engine {full_fleet} with ranks "
                f"{list(self._shard_widths)}, got {membership.shard_prefixes} with "
                f"{list(membership.workers_per_shard)}."
            )

    def full_fleet_membership(self, train_world_size: int) -> RefitMembership:
        """Every engine, each a DP shard of its own TP x PP ranks after the trainers."""
        return desired_membership(
            absent_shards=[],
            dp_size=self.num_engines,
            total_gen_workers=self.inference_world_size,
            train_world_size=train_world_size,
            shard_widths=self._shard_widths,
        )

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        """Join every engine to the NeMo RL ``model_update_group``."""
        expected = train_world_size + self.inference_world_size
        if world_size != expected:
            raise ValueError(
                f"NCCL world_size={world_size} does not match expected {expected} "
                f"(train={train_world_size} + engines={self.inference_world_size})"
            )
        return self.rebuild_collective(
            self.full_fleet_membership(train_world_size), ip, port
        )

    def rebuild_collective(
        self, membership: RefitMembership, ip: str, port: int
    ) -> list[ray.ObjectRef]:
        """Join every engine to the NeMo RL ``model_update_group`` at its ranks.

        Engine ``i``'s workers are offset to the absolute rank where its shard
        starts: the trainers plus its prefix within the generation block.
        """
        self.check_full_fleet(membership)
        futures: list[ray.ObjectRef] = []
        for index, record in enumerate(self._records):
            futures.extend(
                self._client(record).init_collective(
                    ip,
                    port,
                    membership.world_size,
                    train_world_size=membership.train_world_size,
                    rank_offset=membership.train_world_size
                    + membership.shard_prefixes[index],
                )
            )
        return futures

    def rebuild_nccl_reshard_comm_group(
        self,
        membership: RefitMembership,
        *,
        pp_ips: list[str],
        pp_ports: list[int],
        pp_size: int,
        train_ranks_per_stage: int,
        sub_world_size: int,
    ) -> list[ray.ObjectRef]:
        raise NotImplementedError(
            "Megatron engines behind Dynamo receive weights over the collective "
            "path only; the nccl_reshard transport is not available."
        )

    def prepare_refit_info(self, state_dict_info: dict[str, Any] | None) -> None:
        """Prepare Bridge conversion tasks on every engine."""
        if state_dict_info is None:
            raise ValueError("state_dict_info must not be None for Megatron refit")
        # Stored once; every worker resolves the same object.
        info_ref = ray.put(state_dict_info)
        self._call_engines(
            self._records,
            lambda client: client.prepare_refit_info(info_ref),
            what="prepare_refit_info",
            deadline=time.monotonic() + self._control_timeout_s,
        )

    def prepare_nccl_reshard_refit_info(self, wire_refit_info: dict[str, Any]) -> None:
        raise NotImplementedError(
            "Megatron engines behind Dynamo receive weights over the collective "
            "path only; the nccl_reshard transport is not available."
        )

    def update_weights_from_collective(
        self, refit_timeout_s: float | None, *, refit_with_reload_api: bool
    ) -> list[ray.ObjectRef]:
        """Receive one collective refit on every engine.

        ``refit_with_reload_api`` is accepted for protocol parity and ignored:
        Megatron engines have no reload API, every refit is a packed broadcast
        into the live parameters.
        """
        futures: list[ray.ObjectRef] = []
        for record in self._records:
            futures.extend(
                self._client(record).update_weights_from_collective(refit_timeout_s)
            )
        return futures

    def nccl_reshard_refit(self, refit_timeout_s: float | None) -> list[ray.ObjectRef]:
        raise NotImplementedError(
            "Megatron engines behind Dynamo receive weights over the collective "
            "path only; the nccl_reshard transport is not available."
        )

    # ------------------------------------------------------------------
    # Fence around the transfer
    # ------------------------------------------------------------------

    def _wait_for_prefill_handoffs(
        self, prefill: Sequence[DynamoWorkerRecord], deadline: float
    ) -> None:
        """Wait until no prefill engine holds a pinned handoff.

        A pinned block is prompt KV a decode engine may still be pulling over
        NIXL; suspending the prefill engine while any are pinned could corrupt
        that transfer. Requests may still be in flight (nothing holds new ones
        back), but the prefill engines are paused, so no new handoff starts
        and the count only decreases while the still-running decode engines
        pull and release what is pinned.
        """
        while True:
            pending = [
                (record, self._client(record).pinned_handoff_block_count())
                for record in prefill
            ]
            pinned: dict[str, int] = {}
            for record, future in pending:
                try:
                    pinned[record.instance_id] = int(
                        ray.get(future, timeout=_seconds_left(deadline))
                    )
                except Exception as error:
                    error.add_note(
                        "reading pinned handoff blocks failed on engine "
                        f"{record.instance_id}"
                    )
                    raise
            if not any(pinned.values()):
                return
            if time.monotonic() >= deadline:
                raise PinnedHandoffTimeout(
                    f"prefill engines still hold handoff blocks: {pinned}"
                )
            time.sleep(min(self._idle_poll_interval_s, _seconds_left(deadline)))

    def _restore_engines(
        self,
        records: Sequence[DynamoWorkerRecord],
        *,
        rebuild_kv_cache: bool,
        deadline: float,
    ) -> list[str]:
        """Bring ``records`` back to serving, decode engines first.

        Decode is the reader of prefill KV, so it must be serving before a
        prefill engine can hand anything off again. With ``rebuild_kv_cache``
        each engine first rebuilds its cache view for the new weights (which
        also builds the engine on the first refit); an engine whose rebuild
        fails is not resumed. Every engine is attempted; the failures are
        returned.
        """
        wanted = {record.instance_id for record in records}
        errors: list[str] = []
        for phase in (self._non_prefill_records(), self._prefill_records()):
            to_resume = [record for record in phase if record.instance_id in wanted]
            if rebuild_kv_cache:
                to_resume = self._attempt(
                    to_resume,
                    lambda client: client.prepare_for_generation(tags=["kv_cache"]),
                    what="prepare_for_generation(kv_cache)",
                    deadline=deadline,
                    errors=errors,
                )
            self._attempt(
                to_resume,
                lambda client: client.resume_after_refit(),
                what="resume_after_refit",
                deadline=deadline,
                errors=errors,
            )
        return errors

    @classmethod
    def _attempt(
        cls,
        records: Sequence[DynamoWorkerRecord],
        call: Callable[[MegatronDynamoWorkerClient], list[ray.ObjectRef]],
        *,
        what: str,
        deadline: float,
        errors: list[str],
    ) -> list[DynamoWorkerRecord]:
        """Run ``call`` on every engine at once; return the ones that succeeded."""
        pending = [(record, call(cls._client(record))) for record in records]
        succeeded: list[DynamoWorkerRecord] = []
        for record, futures in pending:
            try:
                ray.get(futures, timeout=_seconds_left(deadline))
            except Exception as error:
                errors.append(f"{record.instance_id}: {what}: {error!r}")
            else:
                succeeded.append(record)
        return succeeded

    def pause(self, *, clear_cache: bool, timeout_s: float | None) -> None:
        """Take every engine down in the order that keeps prefill handoffs whole.

        Prefill engines stop scheduling first, so no new handoff can start,
        then the channel waits until decode has pulled every block prefill still
        pins. Only then are the prefill engines suspended, then the decode
        engines, and every engine prepares its weights for the swap.

        If any step fails or the deadline passes, every engine the pause
        reached is resumed before the error propagates, so a failed pause
        leaves the fleet serving rather than half stopped. Errors from that
        restore are added to the original error as notes.

        Raises:
            NotImplementedError: ``clear_cache`` is set. The engines keep their
                KV caches across the refit (``kv_cache_management_mode:
                persist``), so frozen requests cannot recompute their KV.
            PinnedHandoffTimeout: The prefill pins did not drain in time.
        """
        if clear_cache:
            raise NotImplementedError(
                "Megatron engines behind Dynamo keep their KV caches across a "
                "refit and cannot drop them at pause; set "
                "recompute_kv_cache_after_weight_updates=false."
            )
        timeout_s = self._control_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        prefill = self._prefill_records()
        rest = self._non_prefill_records()
        # Every engine a pause step was sent to, even one whose call failed: the
        # call may have taken effect anyway.
        reached = list(prefill)
        try:
            self._call_engines(
                prefill,
                lambda client: client.pause_for_refit(),
                what="pause_for_refit",
                deadline=deadline,
            )
            self._wait_for_prefill_handoffs(prefill, deadline)
            self._call_engines(
                prefill,
                lambda client: client.suspend_for_refit(),
                what="suspend_for_refit",
                deadline=deadline,
            )
            reached.extend(rest)
            self._call_engines(
                rest,
                lambda client: client.suspend_for_refit(),
                what="suspend_for_refit",
                deadline=deadline,
            )
            self._call_engines(
                self._records,
                lambda client: client.prepare_for_generation(tags=["weights"]),
                what="prepare_for_generation(weights)",
                deadline=deadline,
            )
        except Exception as error:
            # The weights are unchanged, so a plain resume puts each engine back
            # in service. A fresh deadline: the pause may have used up its own.
            restore_errors = self._restore_engines(
                reached,
                rebuild_kv_cache=False,
                deadline=time.monotonic() + self._control_timeout_s,
            )
            if restore_errors:
                error.add_note(
                    "resuming engines after the failed pause also failed: "
                    + "; ".join(restore_errors)
                )
            raise

    def resume(self, *, timeout_s: float | None) -> None:
        """Rebuild each engine's KV cache and resume serving, decode engines first.

        Every engine is attempted even if another fails, so one bad engine does
        not keep the rest of the fleet down.

        Raises:
            RuntimeError: Any engine failed to rebuild its cache or resume.
        """
        timeout_s = self._control_timeout_s if timeout_s is None else timeout_s
        errors = self._restore_engines(
            self._records,
            rebuild_kv_cache=True,
            deadline=time.monotonic() + timeout_s,
        )
        if errors:
            raise RuntimeError(
                "Megatron resume after weight update failed for " + "; ".join(errors)
            )

    def flush_cache(self) -> bool:
        """Nothing to flush: engines persist their caches across the refit."""
        return False
