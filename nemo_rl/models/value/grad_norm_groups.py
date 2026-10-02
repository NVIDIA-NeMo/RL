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
"""Pre-clip critic gradient norm split by network part (``critic/gnorm/*``).

Port of legacy megatron_value_worker._pre_clip_grad_norms_by_group
(jiaqiz/ppo-dev). clip_grad applies ONE global norm across the whole critic, so
when the total sits far above the threshold this shows which part (attention /
mamba / moe / mlp / value_head / embedding / other) sets the clip factor for
everything else.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Optional

import torch

# Block type -> metric name, keyed by Nemotron-H's hybrid layer pattern
# ("MEMEMEM*EMEM..."), which is exact where parameter-name heuristics are not:
# every block is decoder.layers.N.mixer.*, and only the pattern says whether
# that mixer is Mamba, MoE, or attention.
_HYBRID_BLOCK_NAMES = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}
# Fixed, rank-independent ordering for the cross-rank reduction.
# grad_norm_group_of must only ever return names from this tuple.
GRAD_NORM_GROUP_ORDER = (
    "attention",
    "mamba",
    "moe",
    "mlp",
    "value_head",
    "embedding",
    "other",
)
_LAYER_IDX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# mcore renamed hybrid_override_pattern -> hybrid_layer_pattern; try both,
# newest first (reading only the old name silently dumps every layer in "other").
_HYBRID_PATTERN_ATTRS = ("hybrid_layer_pattern", "hybrid_override_pattern")
# Fallback when no pattern is exposed: Nemotron-H mixer submodule names.
_NAME_HINTS = (
    (
        "attention",
        ("q_proj", "k_proj", "v_proj", "o_proj", "linear_qkv", "self_attention"),
    ),
    ("mamba", ("a_log", "conv1d", "dt_bias", "in_proj", "mixer.d")),
    ("moe", ("experts", "router", ".gate.", "latent_proj", "linear_fc")),
)


def hybrid_pattern_of(model: Any) -> Optional[str]:
    """Best-effort lookup of the hybrid layer pattern across mcore versions."""
    chunks = model if isinstance(model, list) else [model]
    for chunk in chunks:
        inner = getattr(chunk, "module", None)  # DDP / Float16Module wrapper
        for obj in (
            getattr(chunk, "config", None),
            chunk,
            getattr(inner, "config", None),
            inner,
        ):
            for attr in _HYBRID_PATTERN_ATTRS:
                pat = getattr(obj, attr, None)
                if isinstance(pat, str) and pat:
                    return pat
    return None


def grad_norm_group_of(name: str, hybrid_pattern: Optional[str]) -> str:
    """Bucket a parameter by which part of the network it belongs to."""
    if "output_layer" in name:
        # The value head: freshly initialized, so its gradients start far larger
        # than the pretrained backbone's.
        return "value_head"
    if "embedding" in name:
        return "embedding"
    m = _LAYER_IDX_RE.search(name)
    if m and hybrid_pattern:
        idx = int(m.group(1))
        if 0 <= idx < len(hybrid_pattern):
            return _HYBRID_BLOCK_NAMES.get(hybrid_pattern[idx], "other")
    lowered = name.lower()
    for group, hints in _NAME_HINTS:
        if any(h in lowered for h in hints):
            return group
    return "other"


def pre_clip_grad_norms_by_group(
    model: Any, hybrid_pattern: Optional[str], mp_group: Any
) -> dict[str, torch.Tensor]:
    """Per-group PRE-clip gradient norms.

    Must be called BEFORE ``optimizer.step()`` (where Megatron clips). mcore DDP
    accumulates into ``param.main_grad``. Sums squares locally then all-reduces
    SUM over the model-parallel group; tensors replicated across TP ranks are
    counted once (``param_is_not_tensor_parallel_duplicate``, as
    ``get_grad_norm_fp32`` does).

    The reduction is a COLLECTIVE: every rank of ``mp_group`` must call it with
    the same shape, so the vector is always built in the fixed
    ``GRAD_NORM_GROUP_ORDER`` (zero-filled) and this never returns early.
    """
    try:
        from megatron.core.tensor_parallel.layers import (
            param_is_not_tensor_parallel_duplicate,
        )
    except ImportError:  # pragma: no cover - older mcore / CPU tests
        param_is_not_tensor_parallel_duplicate = None

    sums: dict[str, Any] = defaultdict(float)
    chunks = model if isinstance(model, list) else [model]
    for chunk in chunks:
        for name, param in chunk.named_parameters():
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is None:
                continue
            if param_is_not_tensor_parallel_duplicate is not None:
                try:
                    is_duplicate = not param_is_not_tensor_parallel_duplicate(param)
                except (AssertionError, RuntimeError):
                    # Model parallelism not initialized: a single process holds
                    # every tensor exactly once.
                    is_duplicate = False
                if is_duplicate:
                    continue
            group = grad_norm_group_of(name, hybrid_pattern)
            sums[group] = sums[group] + grad.detach().float().pow(2).sum()

    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    stacked = torch.zeros(
        len(GRAD_NORM_GROUP_ORDER), dtype=torch.float32, device=device
    )
    for i, key in enumerate(GRAD_NORM_GROUP_ORDER):
        val = sums.get(key)
        if val is not None:
            stacked[i] = torch.as_tensor(val, device=device).reshape(())
    if (
        mp_group is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.all_reduce(
            stacked, op=torch.distributed.ReduceOp.SUM, group=mp_group
        )
    stacked = stacked.sqrt().cpu()
    return {
        key: stacked[i] for i, key in enumerate(GRAD_NORM_GROUP_ORDER) if stacked[i] > 0
    }
