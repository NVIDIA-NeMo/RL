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
"""Engine layout for the SGLang backend: one spec per logical engine.

An SGLang engine wider than a node is driven by one worker actor per node
("slot"); ``SGLangGeneration.all_engines`` is the flat list of slots. Before
prefill/decode disaggregation every engine had the same width, so the slot of
an engine was plain arithmetic on ``nodes_per_engine``. Under disaggregation a
prefill engine and a decode engine may differ in width, so the arithmetic is
replaced by these specs, which also carry each engine's role, replica and
router group.
"""

from dataclasses import dataclass

from nemo_rl.models.generation.disagg_layout import plan_engines
from nemo_rl.models.generation.interfaces import DisaggConfig


@dataclass(frozen=True)
class SGLangEngineSpec:
    """One logical SGLang engine (an HTTP server plus its TP/PP peers)."""

    index: int
    # "aggregated" without disaggregation, else "prefill" or "decode".
    role: str
    # GPUs the engine spans (tp_size * pp_size).
    width: int
    # First GPU of the engine in the placement group's reordered GPU list.
    gpu_offset: int
    # Index of the engine's node-rank-0 actor in ``all_engines``.
    first_slot: int
    # Actors (one per node) that drive the engine.
    num_slots: int
    # Replica this engine belongs to (its DP-shard index without disaggregation).
    replica: int
    # Router group that fronts the engine; always 0 without disaggregation.
    router: int

    @property
    def slots(self) -> range:
        """Indices of this engine's actors in ``all_engines``."""
        return range(self.first_slot, self.first_slot + self.num_slots)

    def slot_gpu_index(self, node_rank: int) -> int:
        """First reordered GPU index of the actor with ``node_rank``."""
        return self.gpu_offset + node_rank * self.slot_width

    @property
    def slot_width(self) -> int:
        """GPUs one actor of this engine owns."""
        return self.width // self.num_slots


def plan_sglang_engines(
    *,
    num_gpus: int,
    gpus_per_node: int,
    layout: DisaggConfig,
    aggregated_width: int,
    prefill_width: int,
    decode_width: int,
    router_scope: str,
) -> tuple[list[SGLangEngineSpec], int, int]:
    """Lay the engines out in GPU order and return ``(specs, num_replicas, num_routers)``.

    Engine order is :func:`plan_engines`'s: without disaggregation one
    aggregated engine per ``aggregated_width`` GPUs; with it, each replica's
    prefill engines followed by its decode engines. An engine wider than a node
    takes whole nodes, as the uniform layout always did.

    ``router_scope`` picks how many sglang routers front the fleet under
    disaggregation: ``"replica"`` gives every replica its own prefill/decode
    pool, ``"global"`` pools every engine behind one router. Without
    disaggregation there is one router either way.
    """
    roles, widths, num_replicas = plan_engines(
        num_gpus,
        layout=layout,
        aggregated_width=aggregated_width,
        prefill_width=prefill_width,
        decode_width=decode_width,
        backend_cfg_key="sglang_cfg",
    )
    engines_per_replica = len(roles) // num_replicas
    per_replica_router = layout.enabled and router_scope == "replica"
    num_routers = num_replicas if per_replica_router else 1

    specs: list[SGLangEngineSpec] = []
    gpu_offset = 0
    slot = 0
    for index, (role, width) in enumerate(zip(roles, widths)):
        if width > gpus_per_node:
            assert width % gpus_per_node == 0, (
                f"an SGLang engine of {width} GPUs must span whole "
                f"{gpus_per_node}-GPU nodes"
            )
        num_slots = max(1, width // gpus_per_node)
        replica = index // engines_per_replica
        specs.append(
            SGLangEngineSpec(
                index=index,
                role=role,
                width=width,
                gpu_offset=gpu_offset,
                first_slot=slot,
                num_slots=num_slots,
                replica=replica,
                router=replica if per_replica_router else 0,
            )
        )
        gpu_offset += width
        slot += num_slots
    return specs, num_replicas, num_routers
