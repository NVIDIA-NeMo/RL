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

"""Engine-agnostic geometry for a disaggregated prefill/decode engine fleet.

Everything role-aware that the controller side needs lives here: which engines
exist, in which fixed order, how many ranks each one spans, and where each one
starts in the refit collective. Engine pools consume a plan; they never compute
offsets themselves.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

EngineRole = Literal["prefill", "decode"]

# Dynamo component names the frontend expects. Decode registers as the generic
# ``backend`` component so an aggregated worker and a decode worker look alike to
# the router; prefill has its own component.
DYNAMO_COMPONENT_BY_ROLE: dict[str, str] = {"decode": "backend", "prefill": "prefill"}

# Decode engines come first so their ranks follow the trainer's directly. The
# order is also what makes rank offsets stable across restarts: it is derived
# from config, not from placement-group iteration.
ROLE_ORDER: tuple[EngineRole, ...] = ("decode", "prefill")


@dataclass(frozen=True)
class EngineSpec:
    """One engine the pool must bring up."""

    role: EngineRole
    index: int
    world_size: int
    engine_cfg: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.role}-{self.index}"

    @property
    def component(self) -> str:
        return DYNAMO_COMPONENT_BY_ROLE[self.role]


@dataclass(frozen=True)
class EngineRecord:
    """One running engine as the refit channel sees it.

    ``control`` is opaque above the engine seam: a driver-side generation handle
    for Megatron engines, an admin URL for Dynamo-managed vLLM workers. It must
    pickle, because the SingleController actor receives a copy.
    """

    role: EngineRole
    world_size: int
    instance_id: str
    node_ip: str
    control: Any


@dataclass(frozen=True)
class DisaggRolePlan:
    """The fixed engine list and the geometry derived from it."""

    engines: tuple[EngineSpec, ...]

    @classmethod
    def from_role_counts(
        cls,
        *,
        decode_workers: int,
        decode_world_size: int,
        prefill_workers: int,
        prefill_world_size: int,
        decode_cfg: Mapping[str, Any] | None = None,
        prefill_cfg: Mapping[str, Any] | None = None,
    ) -> "DisaggRolePlan":
        for label, value in (
            ("decode_workers", decode_workers),
            ("decode_world_size", decode_world_size),
            ("prefill_workers", prefill_workers),
            ("prefill_world_size", prefill_world_size),
        ):
            if value < 1:
                raise ValueError(f"{label} must be >= 1, got {value}")
        counts = {"decode": decode_workers, "prefill": prefill_workers}
        sizes = {"decode": decode_world_size, "prefill": prefill_world_size}
        cfgs = {"decode": dict(decode_cfg or {}), "prefill": dict(prefill_cfg or {})}
        engines = tuple(
            EngineSpec(
                role=role, index=index, world_size=sizes[role], engine_cfg=cfgs[role]
            )
            for role in ROLE_ORDER
            for index in range(counts[role])
        )
        return cls(engines=engines)

    @property
    def inference_world_size(self) -> int:
        return sum(spec.world_size for spec in self.engines)

    @property
    def max_engine_world_size(self) -> int:
        return max(spec.world_size for spec in self.engines)

    @property
    def expected_components(self) -> dict[str, int]:
        """Dynamo component name -> number of engines the frontend must see."""
        counts: dict[str, int] = {}
        for spec in self.engines:
            counts[spec.component] = counts.get(spec.component, 0) + 1
        return counts

    def engines_with_role(self, role: EngineRole) -> tuple[EngineSpec, ...]:
        return tuple(spec for spec in self.engines if spec.role == role)

    def rank_offsets(self, train_world_size: int) -> list[int]:
        """First collective rank of each engine, in engine order.

        The trainer occupies ranks ``[0, train_world_size)``; engine ``i`` then
        occupies ``world_size_i`` consecutive ranks after every earlier engine.
        """
        offsets: list[int] = []
        cursor = train_world_size
        for spec in self.engines:
            offsets.append(cursor)
            cursor += spec.world_size
        return offsets

    def assert_fits(self, total_gpus: int) -> None:
        """Fail unless the inference allocation is exactly the plan's GPU count."""
        expected = self.inference_world_size
        if total_gpus != expected:
            raise ValueError(
                "Dynamo disaggregation requires the inference GPUs to equal "
                "sum(workers x engine world size) over both roles: "
                f"allocated={total_gpus}, expected={expected} "
                f"({self.describe()})"
            )

    def describe(self) -> str:
        parts = []
        for role in ROLE_ORDER:
            specs = self.engines_with_role(role)
            if specs:
                parts.append(f"{role}: {len(specs)} x {specs[0].world_size} ranks")
        return ", ".join(parts)
