# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Classify native Megatron parameters for external reshard publishers.

The DTensor MX path uses the generic MX publisher directly because DTensor's
``Placement`` enum gives sharding info in a uniform way. Megatron-Core has no
such uniform API; sharding lives in the wrapper-class identity
(``ColumnParallelLinear``, ``RowParallelLinear``, ``VocabParallelEmbedding``,
fused QKV/MLP, MoE expert layers).

This module:

* Classifies every parameter into a Megatron sharding role by walking the model
  graph and consulting **Megatron-Bridge's authoritative parallelism
  registry** (``megatron.bridge.models.conversion.param_mapping.AutoMapping
  ._MODULE_TYPE_REGISTRY``). Bridge's registry already classifies every
  TE / Inference / Quant variant of column-parallel, row-parallel, and
  replicated modules — using it directly rather than rolling our own
  string-matching means we get correct classification of:
    - ``TEColumnParallelLinear``, ``TELayerNormColumnParallelLinear``,
      ``TEColumnParallelGroupedLinear``, ``InferenceLayerNormColumnParallelLinear``
    - ``TERowParallelLinear``, ``TERowParallelGroupedLinear``,
      ``InferenceRowParallelLinear``
    - ``TENorm``, ``FusedLayerNorm``, ``WrappedTorchNorm``, ``L2Norm``,
      ``InferenceTopKRouter``, ``LinearForLastLayer``
  …without us having to maintain a parallel list. If Bridge is not
  importable, the helper falls back to string-matching against the
  base class names — sufficient for mainline Megatron-Core.
* Extracts the local native shard (no allgather, no Megatron-Bridge
  ``export_hf_weights`` call — the param tensor IS the local shard).
* Builds the descriptor extras consumed by ModelExpress's main-native
  ``refit.reshard.megatron_aliases`` adapter.

Limitations:

* Fused QKV / fused gated MLP detection is currently keyed on common
  Megatron name patterns (``linear_qkv``, ``linear_fc1``). Mainline
  Megatron-Core uses these names; non-mainline forks may need a
  ``megatron_role_overrides`` entry.
* MoE per-expert publishing classifies as ``expert_column`` /
  ``expert_row``; the per-expert axis is assumed to be 0 (the leading
  axis), matching ``detect_moe_expert_layout``'s convention.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Iterator

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("nemo_rl.distributed.mx_megatron_helpers")

QkvGeometry = tuple[int, int, int]
QkvGeometryResolver = Callable[[str, Any, Any], QkvGeometry | None]


# Match ModelExpress's main-native Megatron alias role vocabulary.
ROLE_QKV_COLUMN = "qkv_column"
ROLE_GATED_MLP_COLUMN = "gated_mlp_column"
ROLE_COLUMN = "column"
ROLE_ROW = "row"
ROLE_VOCAB_PARALLEL = "vocab_parallel"
ROLE_REPLICATED = "replicated"
ROLE_EXPERT_COLUMN = "expert_column"
ROLE_EXPERT_ROW = "expert_row"

_TP_SHARDED_ROLES = frozenset(
    {
        ROLE_QKV_COLUMN,
        ROLE_GATED_MLP_COLUMN,
        ROLE_COLUMN,
        ROLE_ROW,
        ROLE_VOCAB_PARALLEL,
        ROLE_EXPERT_COLUMN,
        ROLE_EXPERT_ROW,
    }
)


@dataclass
class MegatronRoleSpec:
    """Per-parameter classification result.

    ``role`` is one of the role string constants. ``descriptor_extras`` carries
    the per-tensor metadata consumed by ModelExpress's Megatron alias builder.
    """

    role: str
    descriptor_extras: dict[str, str] = field(default_factory=dict)
    is_expert: bool = False
    expert_axis: int = 0
    owned_expert_ids: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class MegatronTpShardGeometry:
    """Global TP geometry for one rank-local native Megatron tensor."""

    global_shape: tuple[int, ...]
    shard_axis: int
    local_shard_range: tuple[int, int]


def infer_megatron_tp_shard_geometry(
    *,
    local_shape: tuple[int, ...],
    role: str,
    tp_size: int,
    tp_rank: int,
    expert_tp_size: int | None = None,
    expert_tp_rank: int | None = None,
    descriptor_extras: dict[str, str] | None = None,
) -> MegatronTpShardGeometry | None:
    """Describe the TP shard held by a native Megatron parameter.

    Expert tensors still have TP geometry when inference EP is disabled. The
    previous publisher omitted it for every expert, which forced receivers to
    pull each complete expert tensor and slice locally.
    """
    is_expert = role in {ROLE_EXPERT_COLUMN, ROLE_EXPERT_ROW}
    shard_world_size = (
        int(expert_tp_size)
        if is_expert and expert_tp_size is not None
        else int(tp_size)
    )
    shard_rank = (
        int(expert_tp_rank)
        if is_expert and expert_tp_rank is not None
        else int(tp_rank)
    )
    if not 0 <= shard_rank < shard_world_size:
        raise ValueError(
            f"shard rank {shard_rank} is outside [0, {shard_world_size}) "
            f"for Megatron role {role!r}"
        )
    if role == ROLE_REPLICATED or shard_world_size <= 1:
        return None
    extras = descriptor_extras or {}
    expert_layout = extras.get("expert_layout", "grouped")
    if role in {
        ROLE_COLUMN,
        ROLE_QKV_COLUMN,
        ROLE_GATED_MLP_COLUMN,
        ROLE_VOCAB_PARALLEL,
    }:
        shard_axis = 0
    elif role == ROLE_ROW:
        shard_axis = 1
    elif role == ROLE_EXPERT_COLUMN:
        shard_axis = 1 if expert_layout == "leading_axis" else 0
    elif role == ROLE_EXPERT_ROW:
        shard_axis = 2 if expert_layout == "leading_axis" else 1
    else:
        raise ValueError(f"unsupported Megatron TP shard role {role!r}")
    if shard_axis >= len(local_shape):
        raise ValueError(
            f"Megatron role {role!r} requires shard axis {shard_axis}, "
            f"but local shape is {local_shape}"
        )
    local_extent = int(local_shape[shard_axis])
    global_shape = list(local_shape)
    global_shape[shard_axis] = local_extent * shard_world_size
    return MegatronTpShardGeometry(
        global_shape=tuple(global_shape),
        shard_axis=shard_axis,
        local_shard_range=(
            shard_rank * local_extent,
            (shard_rank + 1) * local_extent,
        ),
    )


# Heuristic name patterns for fused-QKV and fused-gate+up linears in
# mainline Megatron-Core. Callers can provide role overrides for forks that use
# different names.
_DEFAULT_FUSED_QKV_NAME_PATTERNS = ("linear_qkv", "qkv_proj", "fused_qkv")
_DEFAULT_FUSED_GATED_MLP_PATTERNS = ("linear_fc1", "gate_up_proj")
# Vocab / embedding name pattern.
_DEFAULT_VOCAB_NAME_PATTERNS = (
    "word_embeddings",
    "embedding",
    "lm_head",
    "output_layer",
)


@lru_cache(maxsize=1)
def _bridge_module_type_registry() -> dict[str, frozenset[str]] | None:
    """Return Bridge's authoritative module classifier registry, or None.

    Bridge ships a curated dict of
    ``{"column": {classes...}, "row": {...}, "replicated": {...}}`` covering
    every TE / Inference / Quant variant. Importing it lazily avoids a hard
    dependency: when Bridge is not in the import path (e.g. in unit tests on
    a CPU-only env), the caller falls back to substring matching against
    the base class names, which is correct for mainline Megatron-Core.

    Cached because this is consulted once per parameter while classifying a
    publish set, and re-running the import machinery and copying the registry
    thousands of times per refit buys nothing: the installed Bridge cannot
    change mid-process. Frozen sets keep the cached value from being mutated by
    a caller.
    """
    try:
        from megatron.bridge.models.conversion.param_mapping import (
            AutoMapping as _AM,
        )

        return {
            kind: frozenset(classes)
            for kind, classes in _AM._MODULE_TYPE_REGISTRY.items()
        }
    except Exception:
        return None


def _classify_module_class(mod_class_name: str) -> str | None:
    """Map ``mod.__class__.__name__`` to a Megatron-Bridge parallelism kind.

    Returns one of ``"column"``, ``"row"``, ``"replicated"``, or ``None``
    if the class name doesn't match any known parallelism variant.
    """
    if not mod_class_name:
        return None
    registry = _bridge_module_type_registry()
    if registry is not None:
        # Direct hit on Bridge's curated set (catches every TE / Inference /
        # Quant variant by exact class name).
        for kind, cls_set in registry.items():
            if mod_class_name in cls_set:
                return kind
        # Bridge also has a special-case for the TE-fused
        # LayerNormColumnParallelLinear: classify as column.
        if "LayerNormColumnParallelLinear" in mod_class_name:
            return "column"
    # Fallback: substring match against the base names.
    if "ColumnParallel" in mod_class_name or "VocabParallelEmbedding" in mod_class_name:
        return "column"
    if "RowParallel" in mod_class_name:
        return "row"
    if any(
        needle in mod_class_name
        for needle in (
            "Norm",
            "RMSNorm",
            "L2Norm",
            "TopKRouter",
            "LinearForLastLayer",
            "IdentityOp",
        )
    ):
        return "replicated"
    return None


_PARAM_LEAF_NAMES = {"weight", "bias", "scale", "_extra_state"}


def _is_param_leaf(name_part: str) -> bool:
    """Return True for any trailing name that's a parameter rather than a child module.

    Includes the standard ``weight``/``bias``/``scale``/``_extra_state``
    and the grouped-MoE per-expert convention ``weight0``, ``weight1``,
    ``weight127``, ``bias0``, etc. Megatron-Core's TE-grouped linears
    expose one ``weight<idx>`` ``nn.Parameter`` per local expert.
    """
    if name_part in _PARAM_LEAF_NAMES:
        return True
    for base in ("weight", "bias", "scale"):
        if name_part.startswith(base):
            suffix = name_part[len(base) :]
            if suffix and suffix.isdigit():
                return True
    return False


def _expert_index_from_param(name_part: str) -> int | None:
    """If ``name_part`` is ``weight<N>``/``bias<N>``/etc, return ``N``."""
    for base in ("weight", "bias", "scale"):
        if name_part.startswith(base):
            suffix = name_part[len(base) :]
            if suffix and suffix.isdigit():
                return int(suffix)
    return None


def canonicalize_grouped_expert_name(
    name: str, descriptor_extras: dict[str, str]
) -> str:
    """Replace an EP-local grouped-expert leaf index with its global ID.

    ModelExpress groups reshard sources by tensor name. Leaving every EP rank's
    first local expert named ``weight0`` makes unrelated experts collide before
    the Megatron translator can inspect descriptor extras.
    """
    if descriptor_extras.get("expert_layout") != "grouped":
        return name
    local = descriptor_extras.get("local_expert_id")
    global_id = descriptor_extras.get("expert_id")
    if local is None or global_id is None:
        raise ValueError(
            f"grouped expert tensor {name!r} is missing local/global expert IDs"
        )
    parent, separator, leaf = name.rpartition(".")
    for prefix in ("weight", "bias", "scale"):
        if leaf == f"{prefix}{local}":
            global_leaf = f"{prefix}{global_id}"
            return f"{parent}{separator}{global_leaf}" if separator else global_leaf
    raise ValueError(
        f"grouped expert tensor {name!r} does not end in an indexed "
        "weight/bias/scale leaf"
    )


def _enclosing_module(name: str, model: "torch.nn.Module") -> "torch.nn.Module | None":
    """Walk down model attributes to find the module that owns ``name``.

    ``name`` is a parameter name like
    ``decoder.layers.0.self_attention.linear_qkv.weight`` or
    ``decoder.layers.0.mlp.experts.linear_fc1.weight0`` for grouped-MoE
    per-expert parameters. Return the parent module of the final
    parameter token.
    """
    parts = name.split(".")
    if not parts or not _is_param_leaf(parts[-1]):
        # Fall back to the deepest module — caller will get a leaf.
        cur = model
        for p in parts:
            sub = getattr(cur, p, None)
            if sub is None:
                return None
            cur = sub
        return cur
    cur: Any = model
    for p in parts[:-1]:
        sub = getattr(cur, p, None)
        if sub is None:
            return None
        cur = sub
    return cur


def resolve_qkv_geometry_from_param(
    name: str, _param: Any, model: "torch.nn.Module"
) -> QkvGeometry | None:
    """Read global Q/KV geometry from the live layer-local QKV module.

    Heterogeneous Megatron models attach the resolved per-layer
    ``TransformerConfig`` to ``linear_qkv``. Reading the root model config would
    stamp one geometry on every layer and is therefore only a compatibility
    fallback for callers that cannot resolve the owning module.
    """
    if not _is_fused_qkv_name(name):
        return None
    module = _enclosing_module(name, model)
    config = getattr(module, "config", None)
    if config is None:
        return None
    q_heads = getattr(config, "num_attention_heads", None)
    if q_heads is None:
        return None
    kv_heads = getattr(config, "num_query_groups", None) or q_heads
    head_dim = getattr(config, "kv_channels", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None or int(hidden_size) % int(q_heads):
            return None
        head_dim = int(hidden_size) // int(q_heads)
    return int(q_heads), int(kv_heads), int(head_dim)


def _module_class_name(mod: "torch.nn.Module | None") -> str:
    if mod is None:
        return ""
    return type(mod).__name__


def _is_fused_qkv_name(name: str) -> bool:
    return any(p in name for p in _DEFAULT_FUSED_QKV_NAME_PATTERNS)


def _is_fused_gated_mlp_name(name: str) -> bool:
    return any(p in name for p in _DEFAULT_FUSED_GATED_MLP_PATTERNS)


def _is_vocab_name(name: str) -> bool:
    return any(p in name for p in _DEFAULT_VOCAB_NAME_PATTERNS)


def _is_expert_name(name: str, *, expert_pattern: str) -> bool:
    return expert_pattern in name


def detect_megatron_role(
    name: str,
    param: "torch.Tensor",
    *,
    model: "torch.nn.Module",
    tp_size: int,
    ep_size: int,
    ep_rank: int,
    num_local_experts: int | None = None,
    num_attention_heads: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    qkv_geometry: QkvGeometry | None = None,
    expert_pattern: str | None = None,
    role_overrides: dict[str, str] | None = None,
) -> MegatronRoleSpec:
    """Classify a Megatron parameter into one of seven roles.

    Returns the role plus per-tensor metadata for the alias builder. The
    classifier is conservative: when we can't
    determine sharding from the module class, we fall back to
    ``ROLE_REPLICATED`` (rank 0 publishes, others skip). That's a
    correctness-preserving default — replicated tensors round-trip via the
    receiver's passthrough path.

    Args:
        name: param name from ``model.named_parameters()`` (e.g.
            ``decoder.layers.0.self_attention.linear_qkv.weight``).
        param: the local shard tensor (Megatron stores native shards).
        model: the root model module; used to walk attributes for the
            enclosing module's class.
        tp_size, ep_size, ep_rank: from ``parallel_state``.
        num_attention_heads, num_kv_heads, head_dim: model-wide compatibility
            fallback for ``qkv_column`` metadata.
        qkv_geometry: optional per-tensor ``(global query heads, global KV
            heads, head dimension)``. This takes precedence over model-wide
            values and is required for heterogeneous attention.
        expert_pattern: substring marker for MoE expert tensors; defaults to
            ``"experts"`` and can be overridden with
            ``NRL_MX_EXPERT_TENSOR_PATTERN``.
        role_overrides: optional ``{param_name_substring: role}`` dict
            for forcing a role on a specific tensor (escape hatch for
            non-mainline Megatron forks).
    """
    expert_pattern = expert_pattern or os.environ.get(
        "NRL_MX_EXPERT_TENSOR_PATTERN", "experts"
    )

    # ---- 1. Explicit override wins. ----
    if role_overrides:
        for needle, role in role_overrides.items():
            if needle in name:
                return MegatronRoleSpec(role=role)

    # ---- 2a. Grouped-MoE per-expert tensors (one ``weight<N>``
    # nn.Parameter per local expert, used by TE-grouped linears even
    # when EP=1). The trailing param name carries the expert index.
    if _is_expert_name(name, expert_pattern=expert_pattern):
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        expert_idx = _expert_index_from_param(leaf)
        if expert_idx is not None:
            # Per-expert grouped tensor. Each `weight<N>` is one expert's
            # full local shard; the receiver runs per_expert assembly.
            #
            # `weight<N>` is LOCAL to this EP rank (every rank names its
            # experts 0..num_local-1). Advertise the GLOBAL expert id so a
            # receiver gathering across EP ranks (EP-trainer -> non-EP / lower-EP
            # rollout) can place experts without collision and the EP filter can
            # route by global ownership. global = ep_rank*num_local + local.
            global_idx = expert_idx
            if num_local_experts:
                global_idx = ep_rank * int(num_local_experts) + expert_idx
            mod_class = _module_class_name(_enclosing_module(name, model))
            sub_role = (
                ROLE_EXPERT_ROW if "RowParallel" in mod_class else ROLE_EXPERT_COLUMN
            )
            return MegatronRoleSpec(
                role=sub_role,
                is_expert=True,
                expert_axis=0,
                owned_expert_ids={global_idx},
                descriptor_extras={
                    "expert_axis": "0",
                    "expert_id": str(global_idx),
                    "local_expert_id": str(expert_idx),
                    "expert_layout": "grouped",
                },
            )

    # ---- 2b. EP>1 leading-axis grouped (legacy path: single .weight
    # holds ep_size experts as the leading axis chunk). ----
    if (
        _is_expert_name(name, expert_pattern=expert_pattern)
        and ep_size > 1
        and param.ndim >= 2
    ):
        leading = param.shape[0]
        if leading % ep_size == 0:
            chunk = leading // ep_size
            owned = set(range(ep_rank * chunk, (ep_rank + 1) * chunk))
            sub_role = ROLE_EXPERT_COLUMN
            if _is_fused_gated_mlp_name(name):
                # Per-expert fused gate+up: assembler treats it as
                # gated_mlp_split inside the per-expert routing.
                sub_role = ROLE_EXPERT_COLUMN
            mod_class = _module_class_name(_enclosing_module(name, model))
            if "RowParallel" in mod_class:
                sub_role = ROLE_EXPERT_ROW
            return MegatronRoleSpec(
                role=sub_role,
                is_expert=True,
                expert_axis=0,
                owned_expert_ids=owned,
                descriptor_extras={
                    "expert_axis": "0",
                    "expert_layout": "leading_axis",
                },
            )

    # ---- 3. Walk to the enclosing module + classify against Bridge's
    # AutoMapping._MODULE_TYPE_REGISTRY (or fall back to substring match). ----
    mod = _enclosing_module(name, model)
    mod_class = _module_class_name(mod)
    parallelism = _classify_module_class(mod_class)

    # ---- 4. VocabParallelEmbedding / lm_head sharded along rows. ----
    if mod_class == "VocabParallelEmbedding" or (
        _is_vocab_name(name)
        and tp_size > 1
        and param.ndim >= 2
        and parallelism == "column"
    ):
        return MegatronRoleSpec(role=ROLE_VOCAB_PARALLEL)

    # ---- 5. Column-parallel linears (incl. all TE / Inference / Quant variants). ----
    if parallelism == "column":
        if _is_fused_qkv_name(name):
            extras: dict[str, str] = {"qkv_interleave": "by_head"}
            if qkv_geometry is not None:
                num_attention_heads, num_kv_heads, head_dim = qkv_geometry
            if (
                num_attention_heads is not None
                and num_kv_heads is not None
                and head_dim is not None
            ):
                q_heads = int(num_attention_heads)
                kv_heads = int(num_kv_heads)
                qkv_head_dim = int(head_dim)
                if (
                    q_heads < 1
                    or kv_heads < 1
                    or qkv_head_dim < 1
                    or q_heads % kv_heads
                ):
                    raise ValueError(
                        f"{name}: invalid global Q/KV geometry "
                        f"{(q_heads, kv_heads, qkv_head_dim)}"
                    )
                expected_global_rows = (q_heads + 2 * kv_heads) * qkv_head_dim
                actual_global_rows = int(param.shape[0]) * int(tp_size)
                if expected_global_rows != actual_global_rows:
                    raise ValueError(
                        f"{name}: fused QKV rows {actual_global_rows} disagree "
                        f"with global head geometry {expected_global_rows}"
                    )
                extras.update(
                    {
                        "num_heads": str(q_heads),
                        "num_kv_heads": str(kv_heads),
                        "head_dim": str(qkv_head_dim),
                    }
                )
                # Preserve compatibility with old MX clients only when local
                # head counts are meaningful. Never publish a zero KV count.
                if tp_size > 0 and q_heads % tp_size == 0 and kv_heads % tp_size == 0:
                    extras["num_heads_local"] = str(q_heads // tp_size)
                    extras["num_kv_heads_local"] = str(kv_heads // tp_size)
            return MegatronRoleSpec(role=ROLE_QKV_COLUMN, descriptor_extras=extras)
        if _is_fused_gated_mlp_name(name):
            return MegatronRoleSpec(
                role=ROLE_GATED_MLP_COLUMN,
                descriptor_extras={"gated_mlp_order": "gate_then_up"},
            )
        return MegatronRoleSpec(role=ROLE_COLUMN)

    # ---- 6. Row-parallel linears. ----
    if parallelism == "row":
        return MegatronRoleSpec(role=ROLE_ROW)

    # ---- 7. Replicated (LayerNorms, biases, scalars, routers, etc.). ----
    # Bridge's registry covers TENorm, FusedLayerNorm, WrappedTorchNorm,
    # LayerNorm, RMSNorm, L2Norm, InferenceTopKRouter, IdentityOp,
    # LinearForLastLayer, TopKRouter — anything unclassified here also
    # falls into "replicated" as a safe default (rank 0 publishes; others
    # skip), since misclassifying a sharded tensor as replicated would
    # silently produce wrong logits while misclassifying a replicated
    # tensor stays correct (just wastes one rank's publish bandwidth).
    return MegatronRoleSpec(role=ROLE_REPLICATED)


def collect_megatron_publish_set(
    model: "torch.nn.Module",
    *,
    tp_size: int,
    pp_size: int,
    pp_rank: int,
    ep_size: int,
    ep_rank: int,
    tp_rank: int,
    num_local_experts: int | None = None,
    num_attention_heads: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    qkv_geometry_resolver: QkvGeometryResolver | None = None,
    expert_pattern: str | None = None,
    role_overrides: dict[str, str] | None = None,
    target_dtype: "torch.dtype | None" = None,
) -> Iterator[tuple[str, "torch.Tensor", MegatronRoleSpec, dict[str, str]]]:
    """Yield ``(name, local_shard, role_spec, full_extras)`` for the publisher.

    For each parameter:

    * Skips replicated tensors when ``tp_rank != 0``. The MX Megatron receiver
      handles rank-0 replicated model tensors specially; publishing local
      copies from non-zero TP ranks can make vLLM's rank-local loader treat
      them as global tensors and slice past the end.
    * Returns the parameter as-is — Megatron stores native shards, so
      the param tensor IS the local shard. No allgather, no Bridge call.
    * ``full_extras`` is the merged ``{megatron_role, tp_rank, tp_size,
      pp_rank, pp_size, ep_rank, ep_size, ...}`` metadata consumed by
      ``mx_reshard_publisher.build_megatron_alias_inputs``.
    """
    for raw_name, param in model.named_parameters():
        if not param.is_floating_point():
            # Skip non-float buffers (rotary inv_freq, etc.); they aren't
            # weight-refit material.
            continue

        # `model.named_parameters()` returns names with a `module.` prefix
        # when the model is wrapped (DDP-style). Two distinct uses of the
        # name:
        #
        # 1. The model-walking classifier needs the ORIGINAL prefixed
        #    name to descend through `model.module.decoder.layers...` —
        #    stripping the prefix breaks `_enclosing_module` and every
        #    non-expert tensor falls to ROLE_REPLICATED.
        # 2. The PUBLISHED name on the catalog has to match Bridge's
        #    name_map (which uses unprefixed names from
        #    `get_conversion_tasks`) so the receiver's name-map lookup
        #    finds the HF target names.
        #
        # Classify with `raw_name`; publish with the stripped form.
        # (Bug surfaced on Qwen3-MoE-30B-A3B on 2026-06-10: the
        # previous version stripped before classification and the
        # receiver saw only `expert_column` / `replicated` because
        # every TP-sharded role fell through to the default.)
        name = raw_name
        while name.startswith("module."):
            # Megatron can wrap a chunk more than once (for example a local
            # Float16Module around a distributed-data-parallel module). Bridge
            # conversion tasks are keyed from the unwrapped module, so every
            # leading wrapper component must be removed, not only the first.
            name = name[len("module.") :]

        qkv_geometry = (
            qkv_geometry_resolver(raw_name, param, model)
            if qkv_geometry_resolver is not None
            else None
        )
        spec = detect_megatron_role(
            raw_name,
            param,
            model=model,
            tp_size=tp_size,
            ep_size=ep_size,
            ep_rank=ep_rank,
            num_local_experts=num_local_experts,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_geometry=qkv_geometry,
            expert_pattern=expert_pattern,
            role_overrides=role_overrides,
        )
        if spec.is_expert:
            name = canonicalize_grouped_expert_name(name, spec.descriptor_extras)

        if spec.role == ROLE_REPLICATED and tp_rank != 0:
            continue

        local = param.detach()
        if target_dtype is not None and local.dtype != target_dtype:
            local = local.to(target_dtype, non_blocking=True)
        local = local.contiguous()

        full_extras: dict[str, str] = {
            "megatron_role": spec.role,
            "tp_rank": str(tp_rank),
            "tp_size": str(tp_size),
            "pp_rank": str(pp_rank),
            "pp_size": str(pp_size),
            "ep_rank": str(ep_rank),
            "ep_size": str(ep_size),
        }
        full_extras.update(spec.descriptor_extras)

        yield name, local, spec, full_extras


__all__ = [
    "MegatronRoleSpec",
    "MegatronTpShardGeometry",
    "ROLE_COLUMN",
    "ROLE_EXPERT_COLUMN",
    "ROLE_EXPERT_ROW",
    "ROLE_GATED_MLP_COLUMN",
    "ROLE_QKV_COLUMN",
    "ROLE_REPLICATED",
    "ROLE_ROW",
    "ROLE_VOCAB_PARALLEL",
    "canonicalize_grouped_expert_name",
    "collect_megatron_publish_set",
    "detect_megatron_role",
    "infer_megatron_tp_shard_geometry",
]
