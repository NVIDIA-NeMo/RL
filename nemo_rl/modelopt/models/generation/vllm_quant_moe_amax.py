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
"""Route per-expert fakequant amax buffers around vLLM's expert weight loader.

The Megatron side exports one ``input_quantizer._amax`` per expert projection,
e.g. ``model.layers.1.mixer.experts.16.up_proj.input_quantizer._amax``. On the
vLLM side the ModelOpt MoE quant module owns a single fused quantizer per
projection group (``w13_input_quantizer`` / ``w2_input_quantizer``), so the
incoming values fan in with ``max``.

vLLM 0.25 loaded these through the model-level parameter dict, which NeMo-RL
patches to include quantizer buffers. vLLM 0.28's ``AutoWeightsLoader`` hands
every ``experts.*`` name to ``RoutedExperts.load_weights`` instead, which
rewrites the checkpoint name with the expert mapping and resolves the result
with a single ``getattr(self, param_name)``. A dotted quantizer-buffer name such
as ``w13_input_quantizer._amax`` cannot resolve that way and the loader raises
``AttributeError: Layer <experts> has no parameter 'w13_input_quantizer._amax'``.

This module applies the same rewrite vLLM does (replace the mapping's
``weight_name`` with its ``param_name``, strip the layer prefix) and then walks
the dotted path on the expert module itself, so the buffers are updated before
the remaining weights reach vLLM's loader.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import reduce
from typing import Any

import torch

INPUT_QUANTIZER_AMAX_SUFFIX = "input_quantizer._amax"


def _expert_modules(model: torch.nn.Module) -> list[Any]:
    """Return every module that owns an expert mapping and a vLLM layer name."""
    return [
        module
        for _, module in model.named_modules()
        if callable(getattr(module, "get_expert_mapping", None))
        and isinstance(getattr(module, "layer_name", None), str)
    ]


def _resolve_dotted(module: Any, path: str) -> Any | None:
    try:
        return reduce(getattr, path.split("."), module)
    except AttributeError:
        return None


def route_moe_input_quantizer_amax(
    model: torch.nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    reduce_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.max,
    mapper: Any | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Fan per-expert ``input_quantizer._amax`` values into the fused quantizers.

    Returns the weights that were *not* consumed, in their original order, so
    the caller can hand them to vLLM's own loader unchanged. Names that end
    with the amax suffix but do not belong to an expert module (dense layers,
    attention quantizers) pass through untouched.

    ``mapper`` is the model's ``hf_to_vllm_mapper`` (a vLLM ``WeightsMapper``),
    when it has one. Refit sends checkpoint names and vLLM's ``layer_name`` /
    expert mapping use vLLM names; ``load_weights`` applies the mapper before
    its own matching, so the same rename has to happen here or e.g. Nemotron-H's
    ``backbone.layers.N.mixer.experts.*`` never matches the module's
    ``model.layers.N.mixer.experts`` prefix and falls through to the loader
    that cannot resolve it. Names the mapper drops (returns ``None``) are
    passed through untouched.

    Raises:
        KeyError: an expert amax name matched an expert module's mapping but
            the rewritten quantizer path does not exist on that module. This
            is the same condition vLLM would have raised on, surfaced with the
            resolved path so the layout mismatch is visible.
    """
    experts = _expert_modules(model)
    if not experts:
        return list(weights)

    # (layer prefix, module, [(param_name, weight_name)]) — computed once.
    routes = []
    for module in experts:
        mapping = module.get_expert_mapping()
        pairs = [(entry[0], entry[1]) for entry in mapping]
        routes.append((f"{module.layer_name}.", module, pairs))

    remaining: list[tuple[str, torch.Tensor]] = []
    for name, tensor in weights:
        if not name.endswith(INPUT_QUANTIZER_AMAX_SUFFIX):
            remaining.append((name, tensor))
            continue
        # Match on the vLLM-side name, exactly as AutoWeightsLoader will.
        vllm_name = mapper._map_name(name) if mapper is not None else name
        if vllm_name is None:
            remaining.append((name, tensor))
            continue
        handled = False
        for prefix, module, pairs in routes:
            if not vllm_name.startswith(prefix):
                continue
            for param_name, weight_name in pairs:
                if weight_name not in vllm_name:
                    continue
                target_path = vllm_name.replace(weight_name, param_name).removeprefix(
                    prefix
                )
                buf = _resolve_dotted(module, target_path)
                if not isinstance(buf, torch.Tensor):
                    raise KeyError(
                        f"Expert module {module.layer_name!r} has no quantizer "
                        f"buffer {target_path!r} for incoming amax {name!r} "
                        f"(vLLM name {vllm_name!r})"
                    )
                with torch.no_grad():
                    buf.copy_(reduce_fn(buf, tensor.to(buf.device, buf.dtype)))
                handled = True
                break
            if handled:
                break
        if not handled:
            remaining.append((name, tensor))
    return remaining
