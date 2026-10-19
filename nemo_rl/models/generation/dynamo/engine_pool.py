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

Two contracts separate "what engines are behind the Dynamo frontend" from the
controller-facing adapter. The adapter, the runtime that owns etcd / NATS /
frontend, and the SingleController never import an engine module; they talk to
these two protocols. A Megatron pool and a vLLM pool are interchangeable behind
them.
"""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import ray

from nemo_rl.models.generation.dynamo.role_plan import DisaggRolePlan, EngineRecord
from nemo_rl.models.generation.interfaces import CollectiveSenderSpec


@runtime_checkable
class DynamoEnginePool(Protocol):
    """Makes the engines exist and registers them with Dynamo."""

    def start(self, plan: DisaggRolePlan) -> list[EngineRecord]:
        """Bring up every engine in plan order and return one record each."""
        ...

    @property
    def records(self) -> list[EngineRecord]:
        """Records returned by :meth:`start`, in engine order."""
        ...

    def expected_endpoints(self) -> dict[str, set[str]]:
        """Dynamo endpoints each component must register before the fleet is ready."""
        ...

    def is_alive(self) -> bool:
        """Whether every engine (and its registration process) is still alive."""
        ...

    def validate(self, expected: Sequence[EngineRecord]) -> list[EngineRecord]:
        """Fail if the live fleet no longer matches ``expected``."""
        ...

    def shutdown(self) -> None:
        """Tear down everything the pool owns. Idempotent."""
        ...


@runtime_checkable
class EngineRefitChannel(Protocol):
    """The refit protocol over a fixed list of engine records.

    The copy the SingleController actor receives must be able to issue every
    call below; ``client_copy`` strips only driver-owned validators.
    """

    @property
    def inference_world_size(self) -> int: ...

    @property
    def sender_spec(self) -> CollectiveSenderSpec: ...

    def prepare(self, state_dict_info: dict[str, Any] | None) -> None: ...

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]: ...

    def update_weights(
        self, refit_timeout_s: float | None = None
    ) -> list[ray.ObjectRef]: ...

    def wait_idle(self, timeout_s: float) -> None:
        """Block until no engine holds state a weight update could corrupt."""
        ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def flush_cache(self) -> bool: ...

    def client_copy(self) -> "EngineRefitChannel": ...
