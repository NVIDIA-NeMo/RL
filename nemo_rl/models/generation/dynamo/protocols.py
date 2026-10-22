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

"""The engine seam behind ``DynamoGeneration``.

Contracts that separate "what engines are behind the Dynamo frontend" from the
controller-facing adapter. The adapter, the runtime that owns etcd / NATS /
frontend, and the SingleController talk to these protocols only, so a fleet of
Dynamo-managed vLLM workers and a fleet of NeMo RL's Megatron inference engines
are interchangeable behind them. ``DynamoRefitChannel`` (vLLM) already has the
refit shape; ``MegatronRefitChannel`` implements the same protocol.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import ray

from nemo_rl.models.generation.dynamo.config import DynamoWorkerRole
from nemo_rl.weight_sync.membership import RefitMembership


@dataclass(frozen=True)
class DynamoWorkerRecord:
    """One running engine as the refit side sees it.

    ``control`` is opaque above the seam: a driver-side generation handle for a
    Megatron engine, an admin URL for a Dynamo-managed vLLM worker. It must
    pickle, because the SingleController actor receives a copy of the channel.
    """

    role: DynamoWorkerRole
    world_size: int
    instance_id: str
    node_ip: str
    control: Any


@runtime_checkable
class RefitChannel(Protocol):
    """The refit protocol over a fixed engine fleet, as the controller drives it.

    The copy the SingleController actor receives must be able to issue every
    call below; ``client_copy`` strips only driver-owned validators.
    """

    @property
    def inference_world_size(self) -> int: ...

    @property
    def num_engines(self) -> int: ...

    @property
    def shard_widths(self) -> tuple[int, ...]:
        """Ranks each engine spans, in engine (refit shard) order."""
        ...

    def client_copy(self) -> "RefitChannel": ...

    def check_full_fleet(self, membership: RefitMembership) -> None:
        """Reject any membership but every engine: managed fleets are fixed."""
        ...

    def full_fleet_membership(self, train_world_size: int) -> RefitMembership: ...

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]: ...

    def rebuild_collective(
        self, membership: RefitMembership, ip: str, port: int
    ) -> list[ray.ObjectRef]: ...

    def rebuild_nccl_reshard_comm_group(
        self,
        membership: RefitMembership,
        *,
        pp_ips: list[str],
        pp_ports: list[int],
        pp_size: int,
        train_ranks_per_stage: int,
        sub_world_size: int,
    ) -> list[ray.ObjectRef]: ...

    def prepare_refit_info(self, state_dict_info: dict[str, Any] | None) -> None: ...

    def prepare_nccl_reshard_refit_info(
        self, wire_refit_info: dict[str, Any]
    ) -> None: ...

    def update_weights_from_collective(
        self, refit_timeout_s: float | None, *, refit_with_reload_api: bool
    ) -> list[ray.ObjectRef]: ...

    def nccl_reshard_refit(
        self, refit_timeout_s: float | None
    ) -> list[ray.ObjectRef]: ...

    def pause(self, *, clear_cache: bool, timeout_s: float | None) -> None:
        """Make a weight swap safe: no engine may be mid-request afterwards."""
        ...

    def resume(self, *, timeout_s: float | None) -> None: ...

    def flush_cache(self) -> bool: ...
