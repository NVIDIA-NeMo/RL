# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Megatron-Core publisher helpers for the MX v2 path (Phase A).

The DTensor MX path (FSDP2) uses ``mx_helpers.collect_named_local_shards``
+ ``add_tensor`` directly because DTensor's ``Placement`` enum gives
sharding info in a uniform way. Megatron-Core has no such uniform API;
sharding lives in the wrapper-class identity (``ColumnParallelLinear``,
``RowParallelLinear``, ``VocabParallelEmbedding``, fused QKV/MLP, MoE
expert layers).

This module:

* Classifies every parameter into one of seven Megatron roles
  (see ``temp/NemoRL_Megatron_MX_Design.md`` §3) by walking the model
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
* Builds the ``extra_parameters`` dict the v2 publisher attaches so the
  MX-side slice planner can find the right slices for each receiver.

Receiver-side context (for cross-reference): the receiver translator
(Phase C) is intended to be a thin adapter over Bridge's
``MegatronParamMapping.megatron_to_hf`` calls. With ``mpu`` not
initialised, Bridge's TP-gather / PP-broadcast collectives no-op
(``tp_group / pp_group / ep_group = None``, ``tp_size / pp_size = 1``);
feeding a pre-assembled global tensor (assembled from N trainer ranks
by Phase B's slice planner) into ``mapping.megatron_to_hf(buffer,
megatron_module=None)`` gives back HF-shaped ``{q_proj, k_proj, v_proj}``
/ ``{gate_proj, up_proj}`` / etc. directly. No hand-rolled un-interleave
needed on the receiver. See
``temp/NemoRL_Megatron_MX_Design.md`` §10 for the integration shape.

The downstream MX-side slice planner is implemented in
``modelexpress.nemo_rl_v2`` (``MxV2RefitReceiver.pick_megatron_slice_plans``,
seven Megatron role constants, ``MegatronSourceMeta``,
``MegatronSlicePlan``). See ``ai-dynamo/modelexpress`` PR #421 commits
``12c73a7`` + ``b26e80f``.

Limitations of this Phase A:

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
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("nemo_rl.distributed.mx_megatron_helpers")


# Match modelexpress.nemo_rl_v2.ROLE_MEGATRON_* — keep in sync.
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

    ``role`` is one of the role string constants. ``descriptor_extras`` is
    the per-tensor ``extra_parameters`` payload the publisher will merge
    into MX's ``identity.extra_parameters`` (and the v2 sidecar JSON).
    Keys here MUST match the names ``modelexpress.nemo_rl_v2._extract_megatron_meta``
    reads.
    """

    role: str
    descriptor_extras: dict[str, str] = field(default_factory=dict)
    is_expert: bool = False
    expert_axis: int = 0
    owned_expert_ids: set[int] = field(default_factory=set)


# Heuristic name patterns for fused-QKV and fused-gate+up linears in
# mainline Megatron-Core. Override via ``MxConfig.megatron_role_overrides``
# if your fork uses different names.
_DEFAULT_FUSED_QKV_NAME_PATTERNS = ("linear_qkv", "qkv_proj", "fused_qkv")
_DEFAULT_FUSED_GATED_MLP_PATTERNS = ("linear_fc1", "gate_up_proj")
# Vocab / embedding name pattern.
_DEFAULT_VOCAB_NAME_PATTERNS = (
    "word_embeddings",
    "embedding",
    "lm_head",
    "output_layer",
)


def _bridge_module_type_registry() -> dict[str, set[str]] | None:
    """Return Bridge's authoritative module classifier registry, or None.

    Bridge ships a curated dict of
    ``{"column": {classes...}, "row": {...}, "replicated": {...}}`` covering
    every TE / Inference / Quant variant. Importing it lazily avoids a hard
    dependency: when Bridge is not in the import path (e.g. in unit tests on
    a CPU-only env), the caller falls back to substring matching against
    the base class names, which is correct for mainline Megatron-Core.
    """
    try:
        from megatron.bridge.models.conversion.param_mapping import (
            AutoMapping as _AM,
        )

        return dict(_AM._MODULE_TYPE_REGISTRY)
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


def publish_eagle_draft_weights(
    *,
    publisher: Any,
    draft_model: Any,
    dtype: Any,
) -> int:
    """Publish trainer-owned EAGLE draft weights as replicated MX tensors."""
    if draft_model is None:
        return 0

    from nemo_rl.models.megatron.draft import export_eagle_weights_to_hf

    count = 0
    for name, tensor in export_eagle_weights_to_hf(draft_model):
        if tensor.is_floating_point():
            tensor = tensor.to(dtype, non_blocking=True)
        publisher.add_tensor(
            name=f"draft.{name}",
            tensor=tensor.contiguous(),
            is_expert=False,
            expert_axis=0,
            owned_expert_ids=set(),
            megatron_role=ROLE_REPLICATED,
            megatron_extras={},
        )
        count += 1
    return count


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
    num_attention_heads: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    expert_pattern: str | None = None,
    role_overrides: dict[str, str] | None = None,
) -> MegatronRoleSpec:
    """Classify a Megatron parameter into one of seven roles.

    Returns the role + per-tensor metadata that the publisher should attach
    to ``extra_parameters``. The classifier is conservative: when we can't
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
        num_attention_heads, num_kv_heads, head_dim: required for
            ``qkv_column`` role; derived from the model config. Pass
            ``None`` if unknown — the role still classifies but the
            descriptor will be missing fields and the receiver will
            fall back to its default un-interleave assumptions.
        expert_pattern: substring marker for MoE expert tensors; default
            ``"experts"`` (matches ``MxConfig.NRL_MX_EXPERT_TENSOR_PATTERN``).
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
            mod_class = _module_class_name(_enclosing_module(name, model))
            sub_role = (
                ROLE_EXPERT_ROW if "RowParallel" in mod_class else ROLE_EXPERT_COLUMN
            )
            return MegatronRoleSpec(
                role=sub_role,
                is_expert=True,
                expert_axis=0,
                owned_expert_ids={expert_idx},
                descriptor_extras={
                    "expert_axis": "0",
                    "expert_id": str(expert_idx),
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
            if num_attention_heads is not None and tp_size > 0:
                extras["num_heads_local"] = str(num_attention_heads // tp_size)
            if num_kv_heads is not None and tp_size > 0:
                extras["num_kv_heads_local"] = str(num_kv_heads // tp_size)
            if head_dim is not None:
                extras["head_dim"] = str(head_dim)
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
    num_attention_heads: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
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
      pp_rank, pp_size, ep_rank, ep_size, ...}`` dict the publisher should
      pass straight into ``MxV2TrainingPublisher.add_tensor``'s
      ``extra_parameters`` arg.

    Caller is responsible for invoking ``add_tensor`` and
    ``publish(version=...)`` on the publisher.
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
        name = (
            raw_name[len("module.") :] if raw_name.startswith("module.") else raw_name
        )

        spec = detect_megatron_role(
            raw_name,
            param,
            model=model,
            tp_size=tp_size,
            ep_size=ep_size,
            ep_rank=ep_rank,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            expert_pattern=expert_pattern,
            role_overrides=role_overrides,
        )

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
    "ROLE_COLUMN",
    "ROLE_EXPERT_COLUMN",
    "ROLE_EXPERT_ROW",
    "ROLE_GATED_MLP_COLUMN",
    "ROLE_QKV_COLUMN",
    "ROLE_REPLICATED",
    "ROLE_ROW",
    "ROLE_VOCAB_PARALLEL",
    "collect_megatron_publish_set",
    "detect_megatron_role",
]
