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
``num_prefill_engines`` prefill engines plus ``num_decode_engines`` decode engines
behind ``num_frontend_workers`` frontends, and the replica count follows from the
inference cluster's size. How a backend *implements* that layout lives in its own
block (``trtllm_cfg`` / ``vllm_cfg``): a per-role engine override
(``prefill_engine`` / ``decode_engine``) naming only what the two roles may
disagree on, plus the backend's frontend and KV-transport settings.

This module holds what both backends need and must agree on:

* reading the layout block as its schema,
* resolving the role blocks into complete per-role engine configs,
* sizing a replica in GPUs and planning the engine list that tiles the cluster,
* mapping engines onto placement-group bundles and onto their owner workers.

The TRT-LLM backend (``trtllm/config.py``) delegates its config helpers here; the
vLLM backend builds on the whole module.
"""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel

from nemo_rl.models.generation.interfaces import DisaggConfig

Role = Literal["prefill", "decode"]
ROLES: tuple[Role, Role] = ("prefill", "decode")

# Role blocks and the frontend/transport blocks are *about* the engines, not
# settings of one engine, so they never flow into a resolved role config.
_ROLE_BLOCK_KEYS: tuple[str, ...] = ("prefill_engine", "decode_engine")


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


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """``update`` over ``base``, nested dicts merged key by key; inputs untouched."""
    merged = dict(base)
    for key, value in update.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_role_overrides(
    config: Mapping[str, Any],
    *,
    backend_cfg_key: str,
    allowed_keys: frozenset[str],
    excluded_keys: frozenset[str],
    kwargs_key: str,
) -> Mapping[str, Any]:
    """Resolve ``<backend_cfg>.{prefill,decode}_engine`` into complete engine configs.

    The one place the "a role wins over the shared value" rule is applied. As
    written, a role block names only what differs (``allowed_keys``). Afterwards
    each is that role's whole engine config: the shared backend block with the
    role's keys on top, and the role's ``kwargs_key`` deep-merged over
    ``generation.<kwargs_key>``, so a role's nested block adds to the shared one
    instead of replacing it. Everything downstream reads the resolved block and
    merges nothing itself.

    A role key outside ``allowed_keys`` is rejected: the rest of the backend block
    describes the model or the engine's contract with the rest of the system, so
    a role differing on one would not serve the same policy as its replica-mate.
    Silently dropping a knob someone wrote down would be worse than refusing to
    start.

    Args:
        config: The generation config.
        backend_cfg_key: ``"trtllm_cfg"`` or ``"vllm_cfg"``.
        allowed_keys: The keys a role block may set.
        excluded_keys: Keys of the backend block that are not engine settings
            (the role blocks themselves, the frontend/transport blocks) and so
            never flow into a resolved role config.
        kwargs_key: The backend's engine-constructor passthrough
            (``"trtllm_kwargs"`` / ``"vllm_kwargs"``), deep-merged per role.

    Returns ``config`` unchanged without disaggregation, where the role blocks
    are inert. Idempotent: a resolved block only repeats the shared values outside
    ``allowed_keys``, and those pass the check.
    """
    if not resolve_disagg_layout(config).enabled:
        # The very object, not a copy: callers and tests rely on "nothing to
        # resolve" leaving the config untouched.
        return config  # type: ignore[return-value]

    backend_cfg = dict(config[backend_cfg_key])
    shared = {k: v for k, v in backend_cfg.items() if k not in excluded_keys}
    resolved = dict(backend_cfg)
    for role in ROLES:
        overrides = dict(backend_cfg.get(f"{role}_engine") or {})
        unsupported = sorted(
            k for k in set(overrides) - allowed_keys if overrides[k] != shared.get(k)
        )
        if unsupported:
            raise ValueError(
                f"{backend_cfg_key}.{role}_engine may only override "
                f"{sorted(allowed_keys)}, got {unsupported}. Settings shared by "
                f"every engine of the replica belong on {backend_cfg_key} itself, "
                f"and engine kwargs shared by both roles on generation.{kwargs_key}."
            )
        resolved[f"{role}_engine"] = {
            **shared,
            **overrides,
            kwargs_key: deep_merge(
                dict(config.get(kwargs_key) or {}),
                dict(overrides.get(kwargs_key) or {}),
            ),
        }
    return {**config, backend_cfg_key: resolved}


def role_engine_config(
    config: Mapping[str, Any], role: Role, *, backend_cfg_key: str
) -> dict[str, Any]:
    """The whole engine config of ``role``. ``config`` must be role-resolved."""
    return config[backend_cfg_key][f"{role}_engine"]


def role_tensor_parallel_size(
    config: Mapping[str, Any], role: Role, *, backend_cfg_key: str
) -> int:
    """TP of one engine of ``role``. ``config`` must be role-resolved."""
    return int(
        role_engine_config(config, role, backend_cfg_key=backend_cfg_key)[
            "tensor_parallel_size"
        ]
    )


def engine_width(engine_cfg: Mapping[str, Any]) -> int:
    """GPUs one engine spans: its TP times its PP.

    ``pipeline_parallel_size`` is optional on the TRT-LLM block, which asserts
    PP == 1 elsewhere; an absent key therefore reads as one stage.
    """
    pp = engine_cfg.get("pipeline_parallel_size")
    return int(engine_cfg["tensor_parallel_size"]) * int(1 if pp is None else pp)


def replica_width(config: Mapping[str, Any], *, backend_cfg_key: str) -> int:
    """GPUs one rollout replica spans.

    Without disaggregation a replica is one engine. Under it, a replica is its
    prefill engines plus its decode engines -- the unit that must stay inside one
    NVLink domain, since the KV cache crosses from prefill to decode on every turn.
    ``config`` must be role-resolved when disaggregation is enabled.
    """
    layout = resolve_disagg_layout(config)
    if not layout.enabled:
        return engine_width(config[backend_cfg_key])
    widths = {
        role: engine_width(
            role_engine_config(config, role, backend_cfg_key=backend_cfg_key)
        )
        for role in ROLES
    }
    return (
        layout.num_prefill_engines * widths["prefill"]
        + layout.num_decode_engines * widths["decode"]
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
    """
    if not layout.enabled:
        assert world_size % aggregated_width == 0, (
            f"Cluster world_size ({world_size}) must be divisible by the engine "
            f"width ({aggregated_width})."
        )
        n = world_size // aggregated_width
        # Without disaggregation a replica *is* an engine: one DP shard, one
        # URL, nothing below it to front. Reporting n rather than 1 keeps
        # "replica" meaning the same thing on both paths.
        return ["aggregated"] * n, [aggregated_width] * n, n

    for role, value in (("prefill", prefill_width), ("decode", decode_width)):
        assert value >= 1, f"{backend_cfg_key}.{role}_engine must span >= 1 GPU"

    num_prefill = layout.num_prefill_engines
    num_decode = layout.num_decode_engines
    assert num_prefill >= 1 and num_decode >= 1, (
        f"a replica needs at least one engine of each role, got "
        f"num_prefill_engines={num_prefill}, num_decode_engines={num_decode}"
    )

    width = num_prefill * prefill_width + num_decode * decode_width
    assert world_size % width == 0, (
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


def engine_owner_indices(engine_widths: list[int]) -> list[int]:
    """Global rank of each engine's model owner, in engine order.

    The worker group hands bundle_indices (the "you build the engine" signal) to
    each tied group's local rank 0 and assigns ranks in bundle order, so the
    owners are the prefix sums of the tied group widths.
    """
    owners: list[int] = []
    offset = 0
    for width in engine_widths:
        owners.append(offset)
        offset += width
    return owners


def engine_key(pg_idx: int, local_bundle_indices: list[int]) -> str:
    """Stable id for the engine occupying these bundles.

    Both sides of the worker boundary compute this from the same
    ``(pg_idx, local_bundle_indices)`` tuple, so a worker can look up its own
    entry in the driver-built ``_engine_roles`` map without per-worker init
    kwargs. Keying on the tuple rather than a global ordinal keeps it correct
    for both the unified-PG and per-node-PG layouts, whose local bundle indices
    mean different things.
    """
    return f"{pg_idx}:" + ",".join(str(i) for i in local_bundle_indices)


def slice_engines_on_unified_pg(
    flat_bundles: list[int], engine_widths: list[int]
) -> list[tuple[int, list[int]]]:
    """Consume a topology-sorted bundle list engine by engine.

    A unified placement group has exactly one PG (index 0); engines of differing
    width each take exactly their own number of consecutive bundles.
    """
    if len(flat_bundles) < sum(engine_widths):
        raise ValueError(
            f"Engine layout needs {sum(engine_widths)} bundles but the unified "
            f"placement group has {len(flat_bundles)}."
        )
    tied_groups: list[tuple[int, list[int]]] = []
    cursor = 0
    for width in engine_widths:
        tied_groups.append((0, flat_bundles[cursor : cursor + width]))
        cursor += width
    return tied_groups


def tile_engines_on_per_node_pgs(
    *,
    pg_bundle_counts: list[int],
    pg_order: list[int],
    engine_widths: list[int],
    replica_width_gpus: int,
) -> list[tuple[int, list[int]]]:
    """Pack engines node by node, in ``pg_order``, never splitting one.

    Every engine here fits a node (the unified-PG path handles the other case).
    Nodes are filled greedily in engine order, which is replica order, so a
    replica's prefill and decode engines land next to each other. A node the
    layout does not tile exactly is a startup error: an engine is one tied
    worker group keyed by a single placement group and cannot straddle two,
    and stopping early would leave GPUs idle and silently run fewer engines
    than the config asks for.
    """
    tied_groups: list[tuple[int, list[int]]] = []
    engine_idx = 0
    for pg_idx in pg_order:
        bundle_count = pg_bundle_counts[pg_idx]
        if bundle_count == 0:
            continue
        cursor = 0
        while engine_idx < len(engine_widths):
            width = engine_widths[engine_idx]
            if cursor + width > bundle_count:
                break
            tied_groups.append((pg_idx, list(range(cursor, cursor + width))))
            cursor += width
            engine_idx += 1
        if cursor != bundle_count:
            free = bundle_count - cursor
            if engine_idx < len(engine_widths):
                why = (
                    f"engine {engine_idx} needs {engine_widths[engine_idx]} GPU(s) "
                    f"but only {free} GPU(s) are left on it"
                )
            else:
                why = f"{free} GPU(s) are left over with no engines to place"
            raise ValueError(
                f"Engine layout does not tile placement group {pg_idx} "
                f"({bundle_count} GPUs): {why}. An engine that fits in a node is "
                f"never split across nodes. Engine widths in order: "
                f"{engine_widths}. Two layouts always tile: give every role a "
                f"width equal to the GPUs per node (one engine per node), or make "
                f"the replica width ({replica_width_gpus}) divide the GPUs per "
                f"node (whole replicas per node). Engines are placed replica by "
                f"replica to keep each replica's KV handoff local, so some "
                f"mixed-width layouts that would fit with a different packing "
                f"are rejected."
            )
    if engine_idx < len(engine_widths):
        raise ValueError(
            f"Only {engine_idx} of {len(engine_widths)} engines fit on the "
            f"placement groups {pg_bundle_counts}."
        )
    return tied_groups
