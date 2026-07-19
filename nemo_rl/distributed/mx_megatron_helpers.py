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
"""Classify Megatron-Core parameters for rank-local ModelExpress publication."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    import torch

# Values consumed by ModelExpress's Megatron slice planner.
ROLE_QKV_COLUMN = "qkv_column"
ROLE_GATED_MLP_COLUMN = "gated_mlp_column"
ROLE_COLUMN = "column"
ROLE_ROW = "row"
ROLE_VOCAB_PARALLEL = "vocab_parallel"
ROLE_REPLICATED = "replicated"
ROLE_EXPERT_COLUMN = "expert_column"
ROLE_EXPERT_ROW = "expert_row"


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
# mainline Megatron-Core. Publisher options can override these roles for
# deployments with different parameter names.
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
    except (ImportError, AttributeError):
        return None


def _classify_module(module: "torch.nn.Module | None") -> str | None:
    """Map a module to a Megatron-Bridge parallelism kind.

    Returns one of ``"column"``, ``"row"``, ``"replicated"``, or ``None``
    if no verified rule identifies its placement.
    """
    mod_class_name = _module_class_name(module)
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
    if module is not None:
        tensor_model_parallel = getattr(module, "tensor_model_parallel", None)
        if tensor_model_parallel is False:
            return "replicated"
        if tensor_model_parallel is True:
            partition_dim = getattr(module, "partition_dim", None)
            if partition_dim == 0:
                return "column"
            if partition_dim == 1:
                return "row"
        if mod_class_name == "TELinear":
            parallel_mode = getattr(module, "parallel_mode", None)
            if parallel_mode in ("column", "row"):
                return parallel_mode
            if parallel_mode is None:
                return "replicated"
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


def _expert_index_from_path(name: str) -> int | None:
    """Return the local expert index from a module path when present."""
    parts = name.split(".")
    for marker in ("local_experts", "experts"):
        for index, part in enumerate(parts[:-1]):
            if part == marker and parts[index + 1].isdigit():
                return int(parts[index + 1])
    return None


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
    num_local_experts: int | None = None,
    num_attention_heads: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    expert_pattern: str | None = None,
    role_overrides: dict[str, str] | None = None,
) -> MegatronRoleSpec:
    """Classify a Megatron parameter into one of seven roles.

    Returns the role and per-tensor metadata that the publisher attaches to
    source metadata. Unknown placement at TP greater than one fails before
    publication rather than silently treating a sharded tensor as replicated.

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
        expert_pattern: Substring marker for MoE expert tensors.
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

    # Grouped-MoE parameters use either a ``weight<N>`` leaf or an expert
    # index in the module path. Both indices are local to the EP rank.
    if _is_expert_name(name, expert_pattern=expert_pattern):
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        expert_idx = _expert_index_from_param(leaf)
        if expert_idx is None:
            expert_idx = _expert_index_from_path(name)
        if expert_idx is not None:
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

    # Some grouped linears store all local experts in one leading-axis tensor.
    # Only classify this layout when the module class and configured local
    # expert count agree; ordinary expert linears also have a leading output
    # dimension and must not be mistaken for grouped storage.
    if (
        _is_expert_name(name, expert_pattern=expert_pattern)
        and ep_size > 1
        and num_local_experts is not None
        and param.ndim >= 2
    ):
        mod_class = _module_class_name(_enclosing_module(name, model))
        if "Grouped" in mod_class and param.shape[0] == num_local_experts:
            first_expert = ep_rank * num_local_experts
            owned = set(range(first_expert, first_expert + num_local_experts))
            sub_role = ROLE_EXPERT_COLUMN
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
    parallelism = _classify_module(mod)

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
    if parallelism == "replicated" or tp_size <= 1:
        return MegatronRoleSpec(role=ROLE_REPLICATED)

    raise ValueError(
        "cannot determine Megatron tensor placement for "
        f"parameter {name!r} owned by module {mod_class!r}; "
        "register the module with Megatron Bridge or provide an explicit "
        "megatron_role_overrides entry"
    )


def collect_megatron_publish_set(
    model: "torch.nn.Module",
    *,
    tp_size: int,
    ep_size: int,
    ep_rank: int,
    tp_rank: int,
    num_local_experts: int | None = None,
    num_attention_heads: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    expert_pattern: str | None = None,
    role_overrides: dict[str, str] | None = None,
    target_dtype: "torch.dtype | None" = None,
) -> Iterator[tuple[str, "torch.Tensor", MegatronRoleSpec]]:
    """Yield ``(name, local_shard, role_spec)`` for the publisher.

    For each parameter:

    * Skips replicated tensors when ``tp_rank != 0``. The MX Megatron receiver
      handles rank-0 replicated model tensors specially; publishing local
      copies from non-zero TP ranks can make vLLM's rank-local loader treat
      them as global tensors and slice past the end.
    * Returns the parameter as-is — Megatron stores native shards, so
      the param tensor IS the local shard. No allgather, no Bridge call.
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
            num_local_experts=num_local_experts,
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

        yield name, local, spec


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
