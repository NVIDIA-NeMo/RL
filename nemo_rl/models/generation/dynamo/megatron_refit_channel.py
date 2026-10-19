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

"""Refit protocol over a fleet of NeMo-RL Megatron engines behind Dynamo.

Each engine is one ``MegatronGeneration`` (its own worker group, DP = 1). The
channel fans the controller's single calls out over them in plan order and
reproduces, per engine, the sequence the Megatron weight synchronizer uses for
one non-colocated engine: suspend, prepare weights, transfer, prepare the KV
cache, resume. Rank offsets come from the plan, never from the engine index.
"""

import time
from collections.abc import Sequence
from typing import Any, Protocol

import ray

from nemo_rl.models.generation.dynamo.role_plan import DisaggRolePlan, EngineRecord
from nemo_rl.models.generation.interfaces import CollectiveSenderSpec


class MegatronEngineControl(Protocol):
    """The subset of ``MegatronGeneration`` the channel drives, per engine."""

    def prepare_refit_info(self, state_dict_info: dict[str, Any] | None) -> None: ...

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]: ...

    def update_weights_from_collective(
        self, refit_timeout_s: float | None = None
    ) -> list[ray.ObjectRef]: ...

    def get_collective_sender_spec(self) -> CollectiveSenderSpec: ...

    def suspend_for_refit(self) -> None: ...

    def resume_after_refit(self) -> None: ...

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool: ...

    def pinned_handoff_block_count(self) -> int: ...


class PinnedHandoffTimeout(TimeoutError):
    """Prefill engines still held handoff blocks when the idle wait expired."""


class MegatronRefitChannel:
    """``EngineRefitChannel`` implementation for Megatron engines."""

    def __init__(
        self,
        records: Sequence[EngineRecord],
        *,
        plan: DisaggRolePlan,
        control_timeout_s: float,
        idle_poll_interval_s: float = 0.2,
    ) -> None:
        if len(records) != len(plan.engines):
            raise ValueError(
                f"{len(records)} engine records for a plan of {len(plan.engines)} engines"
            )
        for record, spec in zip(records, plan.engines, strict=True):
            if record.role != spec.role or record.world_size != spec.world_size:
                raise ValueError(
                    f"engine record {record.instance_id} ({record.role}, "
                    f"{record.world_size} ranks) does not match plan entry {spec.name}"
                )
        self._records = tuple(records)
        self._plan = plan
        self._control_timeout_s = control_timeout_s
        self._idle_poll_interval_s = idle_poll_interval_s

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def records(self) -> tuple[EngineRecord, ...]:
        return self._records

    @property
    def inference_world_size(self) -> int:
        return self._plan.inference_world_size

    @property
    def sender_spec(self) -> CollectiveSenderSpec:
        """Megatron engines all share the trainer-side sender contract."""
        return self._control(self._records[0]).get_collective_sender_spec()

    @staticmethod
    def _control(record: EngineRecord) -> MegatronEngineControl:
        return record.control

    def _prefill_records(self) -> list[EngineRecord]:
        return [record for record in self._records if record.role == "prefill"]

    # ------------------------------------------------------------------
    # Refit collective
    # ------------------------------------------------------------------

    def prepare(self, state_dict_info: dict[str, Any] | None) -> None:
        if state_dict_info is None:
            raise ValueError("state_dict_info must not be None for Megatron refit")
        for record in self._records:
            self._control(record).prepare_refit_info(state_dict_info)

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        expected = train_world_size + self.inference_world_size
        if world_size != expected:
            raise ValueError(
                f"NCCL world_size={world_size} does not match expected {expected} "
                f"(train={train_world_size} + engines={self.inference_world_size})"
            )
        futures: list[ray.ObjectRef] = []
        offsets = self._plan.rank_offsets(train_world_size)
        for record, offset in zip(self._records, offsets, strict=True):
            # MegatronGeneration offsets its ranks by the value it is given as
            # ``train_world_size``; for engine i that is the cumulative offset.
            futures.extend(
                self._control(record).init_collective(
                    ip, port, world_size, train_world_size=offset
                )
            )
        return futures

    def update_weights(
        self, refit_timeout_s: float | None = None
    ) -> list[ray.ObjectRef]:
        futures: list[ray.ObjectRef] = []
        for record in self._records:
            futures.extend(
                self._control(record).update_weights_from_collective(
                    refit_timeout_s=refit_timeout_s
                )
            )
        return futures

    # ------------------------------------------------------------------
    # Fence around the transfer
    # ------------------------------------------------------------------

    def wait_idle(self, timeout_s: float | None = None) -> None:
        """Wait until no prefill engine holds a pinned handoff.

        The controller has already waited for every outstanding request; the
        decode adapter's ``RELEASE_KV`` trails the last token by one background
        RPC, so the pins can lag briefly. Megatron's suspend refuses to run while
        any pin is live, so this turns that hard failure into a short wait.
        """
        budget = self._control_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + budget
        while True:
            pinned = {
                record.instance_id: int(
                    self._control(record).pinned_handoff_block_count()
                )
                for record in self._prefill_records()
            }
            if not any(pinned.values()):
                return
            if time.monotonic() >= deadline:
                raise PinnedHandoffTimeout(
                    "prefill engines still hold handoff blocks after "
                    f"{budget}s: {pinned}"
                )
            time.sleep(self._idle_poll_interval_s)

    def pause(self) -> None:
        """Suspend every engine, decode pool first, then prepare the weight swap.

        Decode engines are the readers of prefill KV, so they go down first;
        with the quiesce done, every engine is idle and the order is defensive.
        """
        for record in self._records:
            self._control(record).suspend_for_refit()
        for record in self._records:
            self._control(record).prepare_for_generation(tags=["weights"])

    def resume(self) -> None:
        """Rebuild each engine's KV cache and resume serving, prefill pool first."""
        for record in reversed(self._records):
            control = self._control(record)
            control.prepare_for_generation(tags=["kv_cache"])
            control.resume_after_refit()

    def flush_cache(self) -> bool:
        """Nothing to flush: engines persist their caches and salt hashes per epoch."""
        return False

    def client_copy(self) -> "MegatronRefitChannel":
        """The records hold Ray handles, which pickle; the channel is its own copy."""
        return self
