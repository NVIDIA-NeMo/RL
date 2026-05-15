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
"""Factory that splits a model's parameters into Muon-eligible (linear
weights) and AdamW-eligible (everything else) groups.

The selection mirrors the rule used by Megatron's
``get_megatron_muon_optimizer`` at
``3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/optimizer/muon.py:252-256``
adapted for HF naming conventions: 2D parameter, not an embedding /
output projection, not a normalization weight.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import torch
import torch.nn as nn

from nemo_rl.algorithms.muon.chained import ChainedTorchOptimizer
from nemo_rl.algorithms.muon.dtensor_muon import DTensorMuon

# Modules whose 2D weights must NOT go through Muon. Patterns are matched
# against the dotted parameter name (e.g. "model.embed_tokens.weight").
_NON_MUON_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Embeddings (HF + Megatron + GPT-2 naming)
    re.compile(r"(^|\.)embed_tokens(\.|$)"),
    re.compile(r"(^|\.)wte(\.|$)"),
    re.compile(r"(^|\.)wpe(\.|$)"),
    re.compile(r"(^|\.)word_embeddings(\.|$)"),
    re.compile(r"(^|\.)position_embeddings(\.|$)"),
    # Output projection / LM head
    re.compile(r"(^|\.)lm_head(\.|$)"),
    re.compile(r"(^|\.)output_layer(\.|$)"),
    # Norm layers (RMSNorm / LayerNorm under varied naming schemes)
    re.compile(r"\.norm(\.|$)"),
    re.compile(r"_norm(\.|$)"),
    re.compile(r"layernorm", re.IGNORECASE),
)

# Fused QKV weight produced by Megatron-style attention impls. Separated
# Q/K/V projections (HF default ``q_proj`` / ``k_proj`` / ``v_proj``) are
# normal 2D weights and need no special handling.
_FUSED_QKV_NAME_PATTERN = re.compile(r"\.linear_qkv\.weight$")


def _is_non_muon(name: str) -> bool:
    return any(pattern.search(name) for pattern in _NON_MUON_NAME_PATTERNS)


def _is_fused_qkv(name: str) -> bool:
    return _FUSED_QKV_NAME_PATTERN.search(name) is not None


def _derive_qkv_split_shapes(
    model: nn.Module,
) -> tuple[int, int, int] | None:
    """Read ``num_attention_heads`` / ``num_key_value_heads`` / head_dim from
    ``model.config`` and produce the per-head ``(q, k, v)`` row-count tuple.

    Mirrors Megatron's derivation in
    ``3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/optimizer/muon.py:230-238``
    but with HF naming. Returns ``None`` if the config does not expose enough
    information; callers should treat that as "no fused QKV expected".
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    num_attention_heads = getattr(cfg, "num_attention_heads", None)
    num_key_value_heads = getattr(
        cfg,
        "num_key_value_heads",
        getattr(cfg, "num_query_groups", None),
    )
    kv_channels = getattr(cfg, "head_dim", None) or getattr(cfg, "kv_channels", None)
    if kv_channels is None and num_attention_heads is not None:
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
        if hidden is not None:
            kv_channels = hidden // num_attention_heads
    if not (num_attention_heads and num_key_value_heads and kv_channels):
        return None
    return (
        num_attention_heads // num_key_value_heads * kv_channels,
        kv_channels,
        kv_channels,
    )


def split_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], set[int]]:
    """Return ``(muon_params, adamw_params, fused_qkv_param_ids)``.

    The Muon group is restricted to 2D trainable weights whose names do not
    match any embedding / norm / output-projection pattern; everything else
    (1D weights, biases, embeddings, norms, lm_head) goes to AdamW. Names
    matching the fused-QKV pattern (``*.linear_qkv.weight``) are still
    routed to Muon but their ``id`` is reported so the caller can mark them
    for the per-head split-orthogonalize path.
    """
    muon: list[nn.Parameter] = []
    adamw: list[nn.Parameter] = []
    fused_qkv_ids: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and not _is_non_muon(name):
            muon.append(param)
            if _is_fused_qkv(name):
                fused_qkv_ids.add(id(param))
        else:
            adamw.append(param)
    return muon, adamw, fused_qkv_ids


def build_dtensor_muon(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float = 0.01,
    muon_momentum: float = 0.95,
    muon_use_nesterov: bool = False,
    muon_weight_decay_method: str = "decoupled",
    muon_fp32_matmul_prec: str = "medium",
    muon_coefficient_type: str = "quintic",
    muon_num_ns_steps: int = 5,
    muon_scale_mode: str = "spectral",
    muon_extra_scale_factor: float = 1.0,
    muon_tp_mode: str = "duplicated",
    muon_split_qkv: bool = True,
    qkv_split_shapes: tuple[int, int, int] | None = None,
    adamw_kwargs: dict[str, Any] | None = None,
) -> ChainedTorchOptimizer:
    """Construct a Muon-for-linear, AdamW-for-rest chained optimizer.

    Mirrors Megatron's ``get_megatron_muon_optimizer`` parameter selection
    rules but uses HF-style module name patterns instead of the
    ``is_embedding_or_output_parameter`` marker that Megatron sets on its
    own modules. Fused QKV weights (``*.linear_qkv.weight``) trigger the
    per-head split-orthogonalize path when ``muon_split_qkv`` is true and a
    ``qkv_split_shapes`` tuple can either be derived from ``model.config``
    or is supplied explicitly. HF-style separated Q/K/V projections are
    just normal 2D weights and need no extra wiring.
    """
    muon_params, adamw_params, fused_qkv_ids = split_params(model)
    if not muon_params and not adamw_params:
        raise ValueError(
            "build_dtensor_muon: model has no trainable parameters to optimize"
        )

    if muon_split_qkv and fused_qkv_ids:
        if qkv_split_shapes is None:
            qkv_split_shapes = _derive_qkv_split_shapes(model)
        if qkv_split_shapes is None:
            raise ValueError(
                "build_dtensor_muon: model has fused linear_qkv.weight params "
                "but qkv_split_shapes could not be derived from model.config; "
                "pass qkv_split_shapes=(q, k, v) explicitly."
            )
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = (
            lambda p, ids=fused_qkv_ids: id(p) in ids
        )
    else:
        muon_split_qkv = False
        is_qkv_fn = None

    optimizers: list[torch.optim.Optimizer] = []
    if muon_params:
        optimizers.append(
            DTensorMuon(
                muon_params,
                lr=lr,
                momentum=muon_momentum,
                weight_decay=weight_decay,
                nesterov=muon_use_nesterov,
                weight_decay_method=muon_weight_decay_method,  # type: ignore[arg-type]
                fp32_matmul_prec=muon_fp32_matmul_prec,  # type: ignore[arg-type]
                coefficient_type=muon_coefficient_type,  # type: ignore[arg-type]
                num_ns_steps=muon_num_ns_steps,
                scale_mode=muon_scale_mode,  # type: ignore[arg-type]
                extra_scale_factor=muon_extra_scale_factor,
                tp_mode=muon_tp_mode,  # type: ignore[arg-type]
                split_qkv=muon_split_qkv,
                is_qkv_fn=is_qkv_fn,
                qkv_split_shapes=qkv_split_shapes,
            )
        )
    if adamw_params:
        optimizers.append(
            torch.optim.AdamW(
                adamw_params,
                lr=lr,
                weight_decay=weight_decay,
                **(adamw_kwargs or {}),
            )
        )
    return ChainedTorchOptimizer(optimizers)


# Tell build_optimizer_from_cfg to hand us the full module rather than
# model.parameters(); we need the named-parameter walk for the Muon vs
# AdamW split.
build_dtensor_muon._builds_optimizer_from_model = True  # type: ignore[attr-defined]
