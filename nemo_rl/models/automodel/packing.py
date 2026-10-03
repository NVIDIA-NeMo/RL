# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Helpers for translating HF FA2 packing metadata to Automodel native args."""

from __future__ import annotations

import inspect
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from torch import nn

_NATIVE_PACKING_FORWARD_KEYS = frozenset({"cu_seqlens", "cu_seqlens_q", "cu_seqlens_kv"})
_NEMOTRON_NAME_MARKERS = ("nemotron",)


def _flash_attn_kwargs_as_mapping(flash_attn_kwargs: Any) -> dict[str, Any]:
    """Normalize nested flash-attn packing metadata to a plain dict."""
    if flash_attn_kwargs is None:
        return {}
    if isinstance(flash_attn_kwargs, Mapping):
        return dict(flash_attn_kwargs)
    if is_dataclass(flash_attn_kwargs):
        return asdict(flash_attn_kwargs)

    mapping: dict[str, Any] = {}
    for key in (
        "cu_seqlens",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "cu_seqlens_kv",
        "max_seqlen",
        "max_seqlen_q",
        "max_seqlen_k",
    ):
        if hasattr(flash_attn_kwargs, key):
            mapping[key] = getattr(flash_attn_kwargs, key)
    return mapping


def _model_expects_native_packing_args(model: nn.Module) -> bool:
    """Detect backends that consume top-level cu_seqlens packing args.

    Automodel Nemotron TE attention reads ``cu_seqlens`` / ``qkv_format`` from
    top-level forward kwargs (see Automodel ``preprocess_args_and_kwargs_for_attn``).
    HF FA2 models instead consume nested ``flash_attn_kwargs``. Prefer an explicit
    forward signature, then fall back to Nemotron model_type / class markers.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if isinstance(model_type, str) and any(
        marker in model_type.lower() for marker in _NEMOTRON_NAME_MARKERS
    ):
        return True

    for cls in type(model).mro():
        name = (getattr(cls, "__name__", "") or "").lower()
        module = (getattr(cls, "__module__", "") or "").lower()
        if any(marker in name for marker in _NEMOTRON_NAME_MARKERS):
            return True
        if "nemo_automodel" in module and any(
            marker in module for marker in _NEMOTRON_NAME_MARKERS
        ):
            return True

    forward = getattr(model, "forward", None)
    if forward is None:
        return False
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False

    param_names = set(parameters)
    if param_names & _NATIVE_PACKING_FORWARD_KEYS:
        return True

    has_var_keyword = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    # HF-style forwards that only document nested flash_attn_kwargs (no **kwargs)
    # must keep the nested packing contract.
    if "flash_attn_kwargs" in param_names and not has_var_keyword:
        return False

    return False


def _promote_flash_attn_kwargs_to_native_packing(
    model_batch: dict[str, Any],
) -> None:
    """Promote nested flash_attn_kwargs to Automodel TE top-level packing args.

    Mutates ``model_batch`` in place: sets ``cu_seqlens`` (and q/kv aliases),
    ``max_seqlen`` aliases, and ``qkv_format="thd"``, then drops nested
    ``flash_attn_kwargs``. Token layout stays ``[1, T]``; Automodel Nemotron
    squeezes to THD when ``qkv_format == "thd"`` and the TE backend is active.
    """
    flash_attn_kwargs = model_batch.get("flash_attn_kwargs")
    if not flash_attn_kwargs:
        return

    flash_dict = _flash_attn_kwargs_as_mapping(flash_attn_kwargs)
    cu_seqlens = flash_dict.get("cu_seqlens")
    if cu_seqlens is None:
        cu_seqlens = flash_dict.get("cu_seqlens_q")
    if cu_seqlens is None:
        return

    cu_seqlens_kv = flash_dict.get("cu_seqlens_kv")
    if cu_seqlens_kv is None:
        cu_seqlens_kv = flash_dict.get("cu_seqlens_k", cu_seqlens)

    # Top-level cu_seqlens is what Automodel TE + Mamba seq_idx construction read.
    model_batch["cu_seqlens"] = cu_seqlens
    model_batch["cu_seqlens_q"] = flash_dict.get("cu_seqlens_q", cu_seqlens)
    model_batch["cu_seqlens_kv"] = cu_seqlens_kv

    max_seqlen = flash_dict.get("max_seqlen")
    if max_seqlen is None:
        max_seqlen = flash_dict.get("max_seqlen_q")
    if max_seqlen is not None:
        model_batch["max_seqlen"] = max_seqlen
        model_batch["max_seqlen_q"] = flash_dict.get("max_seqlen_q", max_seqlen)
        model_batch["max_seqlen_kv"] = flash_dict.get(
            "max_seqlen_k", flash_dict.get("max_seqlen_kv", max_seqlen)
        )

    model_batch["qkv_format"] = "thd"
    # Nested kwargs remain the HF FA2 path; native backends ignore them, so drop
    # after promotion to avoid two competing packing interfaces.
    model_batch.pop("flash_attn_kwargs", None)
