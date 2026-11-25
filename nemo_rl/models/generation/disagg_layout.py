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
"""Backend-agnostic pieces of prefill/decode (PD) disaggregation.

The *layout* of a disaggregated rollout fleet is written in terms every backend
shares (:class:`~nemo_rl.models.generation.interfaces.DisaggConfig`): a replica is
``num_prefill_engines`` prefill engines plus ``num_decode_engines`` decode engines,
and the replica count follows from the inference cluster's size. How a backend
*implements* that layout lives in its own block: a per-role engine override
(``prefill_engine`` / ``decode_engine``) naming only what the two roles may
disagree on.

This module holds what the backends need and must agree on: reading the layout
block as its schema, sizing a replica in GPUs from the per-role engine widths, and
planning the engine list that tiles the cluster. Each backend computes its own
engine widths from its own block.
"""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel

from nemo_rl.models.generation.interfaces import DisaggConfig

Role = Literal["prefill", "decode"]
ROLES: tuple[Role, Role] = ("prefill", "decode")


def as_dict(raw: Any) -> dict[str, Any]:
    """A config block as a plain dict: ``None`` is empty, a model dumps its set fields."""
    if raw is None:
        return {}
    if isinstance(raw, BaseModel):
        return raw.model_dump(exclude_unset=True)
    return dict(raw)


def resolve_disagg_layout(config: Mapping[str, Any]) -> DisaggConfig:
    """``generation.disaggregation`` as its schema.

    Absent is the same as present-and-disabled: every field carries its default
    on :class:`DisaggConfig`, so callers read attributes unconditionally instead
    of re-deriving a default per key at the call site.
    """
    return DisaggConfig.model_validate(as_dict(config.get("disaggregation")))


def replica_width_from(
    layout: DisaggConfig, *, prefill_width: int, decode_width: int
) -> int:
    """GPUs one prefill/decode replica spans: its prefill engines plus its decode engines.

    For TRT-LLM, whose frontend serves one replica, the replica must stay inside
    one NVLink domain, since the KV cache crosses from prefill to decode on every
    turn. Behind Dynamo the router pairs prefill and decode engines across the
    whole fleet, so there only one engine needs NVLink.
    """
    return (
        layout.num_prefill_engines * prefill_width
        + layout.num_decode_engines * decode_width
    )


def plan_engines(
    world_size: int,
    *,
    layout: DisaggConfig,
    aggregated_width: int,
    prefill_width: int,
    decode_width: int,
    backend_cfg_key: str,
) -> tuple[list[str], list[int], int]:
    r"""Per-engine ``(role, width)``, in engine order, plus the replica count.

    Without disaggregation every engine is an aggregated engine of
    ``aggregated_width`` GPUs. Under disaggregation each replica contributes its
    prefill engines followed by its decode engines::

        [prefill_width x M, decode_width x K,   prefill_width x M, ...]
         \_________ replica 0 __________/  \____ replica 1 ...

    The replica *count* is derived, not configured: it follows from the cluster
    size, the same way the DP-shard count does without disaggregation. Same-role
    engines stay contiguous so each role's GPUs are adjacent.

    Raises:
        ValueError: When the engines do not tile ``world_size`` GPUs.
    """
    if not layout.enabled:
        if world_size % aggregated_width != 0:
            raise ValueError(
                f"Cluster world_size ({world_size}) must be divisible by the "
                f"engine width ({aggregated_width})."
            )
        n = world_size // aggregated_width
        # Without disaggregation a replica *is* an engine: one DP shard, one
        # URL, nothing below it to front.
        return ["aggregated"] * n, [aggregated_width] * n, n

    for role, value in (("prefill", prefill_width), ("decode", decode_width)):
        if value < 1:
            raise ValueError(f"{backend_cfg_key}.{role}_engine must span >= 1 GPU")

    num_prefill = layout.num_prefill_engines
    num_decode = layout.num_decode_engines
    if num_prefill < 1 or num_decode < 1:
        raise ValueError(
            "a replica needs at least one engine of each role, got "
            f"num_prefill_engines={num_prefill}, num_decode_engines={num_decode}"
        )
    width = replica_width_from(
        layout, prefill_width=prefill_width, decode_width=decode_width
    )
    if world_size % width != 0:
        raise ValueError(
            f"replica width {width} GPUs ({num_prefill} prefill x "
            f"{prefill_width} + {num_decode} decode x {decode_width}) does not "
            f"divide the {world_size} inference GPUs."
        )
    num_replicas = world_size // width

    roles = (["prefill"] * num_prefill + ["decode"] * num_decode) * num_replicas
    widths = (
        [prefill_width] * num_prefill + [decode_width] * num_decode
    ) * num_replicas
    return roles, widths, num_replicas
