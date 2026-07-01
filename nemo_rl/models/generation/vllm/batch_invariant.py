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

"""vLLM BF16 true-on-policy runtime patches used by generation workers."""

from __future__ import annotations

import math
import os
from typing import Any

import torch

G_PATCH_MARKER_ATTR = "_nemo_rl_megatron_style_rmsnorm_patch"
G_ORIGINAL_FORWARD_ATTR = "_nemo_rl_original_forward_cuda"
G_MEGATRON_ROPE_PATCH_MARKER_ATTR = "_nemo_rl_megatron_style_rope_patch"
G_MEGATRON_SWIGLU_PATCH_MARKER_ATTR = "_nemo_rl_megatron_style_swiglu_patch"
G_MEGATRON_ROPE_CACHE_ATTR = "_nemo_rl_megatron_style_cos_sin_cache"
G_TRUE_ON_POLICY_COMPONENTS_ENV = "NEMO_RL_VLLM_TRUE_ON_POLICY_PATCH_COMPONENTS"
G_TRUE_ON_POLICY_BF16_PATCH_COMPONENTS = ("rmsnorm", "rope", "swiglu")
G_TRUE_ON_POLICY_BF16_PATCH_COMPONENT_SET = frozenset(
    G_TRUE_ON_POLICY_BF16_PATCH_COMPONENTS
)


def _rebind_custom_op_forward_methods(
    model: torch.nn.Module,
    target_cls: type,
) -> int:
    rebound_count = 0
    for module in model.modules():
        if isinstance(module, target_cls):
            # CustomOp binds _forward_method at construction, so a class-level
            # patch needs to be rebound onto existing module instances.
            module._forward_method = module.forward_cuda
            rebound_count += 1
    return rebound_count


def install_megatron_style_rmsnorm_patch(
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Route vLLM RMSNorm through Megatron's BI RMSNorm implementation."""
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        BatchInvariantRMSNormFn,
    )
    from vllm.model_executor.layers.layernorm import RMSNorm

    current_forward = RMSNorm.forward_cuda
    original_forward = getattr(
        current_forward,
        G_ORIGINAL_FORWARD_ATTR,
        current_forward,
    )
    already_installed = bool(getattr(current_forward, G_PATCH_MARKER_ATTR, False))

    if not already_installed:

        def patched_forward_cuda(self, x, residual=None):
            if self.variance_size_override is not None or not getattr(
                self, "has_weight", True
            ):
                return original_forward(self, x, residual)

            if residual is not None:
                residual_out = x + residual
                return (
                    BatchInvariantRMSNormFn.apply(
                        residual_out,
                        self.weight.data,
                        self.variance_epsilon,
                        False,
                    ),
                    residual_out,
                )

            return BatchInvariantRMSNormFn.apply(
                x,
                self.weight.data,
                self.variance_epsilon,
                False,
            )

        setattr(patched_forward_cuda, G_PATCH_MARKER_ATTR, True)
        setattr(patched_forward_cuda, G_ORIGINAL_FORWARD_ATTR, original_forward)
        RMSNorm.forward_cuda = patched_forward_cuda

    rebound_count = _rebind_custom_op_forward_methods(model, RMSNorm)

    return {
        "already_installed": already_installed,
        "rebound_count": rebound_count,
    }


def _megatron_style_rotate_half(
    x: torch.Tensor,
    *,
    rotary_interleaved: bool,
) -> torch.Tensor:
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.flatten(-2)


def _megatron_style_duplicate_freqs(
    values: torch.Tensor,
    *,
    rotary_interleaved: bool,
) -> torch.Tensor:
    if not rotary_interleaved:
        return torch.cat((values, values), dim=-1)
    return torch.stack((values, values), dim=-1).flatten(-2)


def _get_megatron_style_cos_sin_cache(module, device: torch.device) -> torch.Tensor:
    cache = getattr(module, G_MEGATRON_ROPE_CACHE_ATTR, None)
    if cache is None or cache.device != device:
        cache = module._compute_cos_sin_cache().to(device=device)
        setattr(module, G_MEGATRON_ROPE_CACHE_ATTR, cache)
    return cache


def _get_megatron_style_inv_freq_from_module_attrs(
    module,
    device: torch.device,
) -> torch.Tensor | None:
    base = getattr(module, "base", None)
    rotary_dim = getattr(module, "rotary_dim", None)
    if base is None or rotary_dim is None:
        return None

    inv_freq = 1.0 / (
        float(base)
        ** (
            torch.arange(0, int(rotary_dim), 2, dtype=torch.float32, device=device)
            / int(rotary_dim)
        )
    )

    class_name = type(module).__name__
    is_llama3_rope = class_name == "Llama3RotaryEmbedding" or all(
        hasattr(module, attr)
        for attr in (
            "scaling_factor",
            "low_freq_factor",
            "high_freq_factor",
            "old_context_len",
        )
    )
    if not is_llama3_rope:
        return inv_freq

    factor = float(getattr(module, "scaling_factor", 8.0))
    low_freq_factor = float(getattr(module, "low_freq_factor", 1.0))
    high_freq_factor = float(getattr(module, "high_freq_factor", 4.0))
    old_context_len = float(getattr(module, "old_context_len", 8192.0))

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    scaled_inv_freq = torch.where(
        wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
    )
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * scaled_inv_freq / factor + smooth_factor * scaled_inv_freq
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_inv_freq, scaled_inv_freq)


def _get_megatron_style_inv_freq(module, device: torch.device) -> torch.Tensor | None:
    inv_freq = _get_megatron_style_inv_freq_from_module_attrs(module, device)
    if inv_freq is not None:
        return inv_freq

    compute_inv_freq = getattr(module, "_compute_inv_freq", None)
    base = getattr(module, "base", None)
    inv_freq = None
    if callable(compute_inv_freq) and base is not None:
        try:
            inv_freq = compute_inv_freq(base)
        except TypeError:
            inv_freq = compute_inv_freq()

    if inv_freq is None:
        inv_freq = getattr(module, "inv_freq", None)
    if inv_freq is None:
        return None
    return inv_freq.to(device=device)


def _get_megatron_style_freqs_half(
    *,
    module,
    positions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    inv_freq = _get_megatron_style_inv_freq(module, device)
    if inv_freq is None:
        return None

    positions = positions.to(device=device, dtype=inv_freq.dtype)
    class_name = type(module).__name__
    if class_name == "LinearScalingRotaryEmbedding" and hasattr(
        module, "scaling_factor"
    ):
        positions = positions / module.scaling_factor

    return torch.outer(positions, inv_freq)


def _apply_megatron_style_rope(
    *,
    module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    original_shape = tensor.shape
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    tensor = tensor.view(num_tokens, -1, module.head_size)
    tensor_rot = tensor[..., : module.rotary_dim]
    tensor_pass = tensor[..., module.rotary_dim :]

    rotary_interleaved = not module.is_neox_style
    freqs_half = _get_megatron_style_freqs_half(
        module=module,
        positions=positions,
        device=tensor.device,
    )
    if freqs_half is None:
        cos_sin_cache = _get_megatron_style_cos_sin_cache(module, tensor.device)
        cos_sin = cos_sin_cache.index_select(0, positions.to(device=tensor.device))
        cos_half, sin_half = cos_sin.chunk(2, dim=-1)
        cos = _megatron_style_duplicate_freqs(
            cos_half,
            rotary_interleaved=rotary_interleaved,
        ).to(tensor_rot.dtype)
        sin = _megatron_style_duplicate_freqs(
            sin_half,
            rotary_interleaved=rotary_interleaved,
        ).to(tensor_rot.dtype)
    else:
        freqs = _megatron_style_duplicate_freqs(
            freqs_half,
            rotary_interleaved=rotary_interleaved,
        )
        cos = torch.cos(freqs).to(tensor_rot.dtype)
        sin = torch.sin(freqs).to(tensor_rot.dtype)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    tensor_rot = (tensor_rot * cos) + (
        _megatron_style_rotate_half(
            tensor_rot,
            rotary_interleaved=rotary_interleaved,
        )
        * sin
    )
    return torch.cat((tensor_rot, tensor_pass), dim=-1).reshape(original_shape)


def install_megatron_style_rope_patch(model: torch.nn.Module) -> dict[str, Any]:
    """Route vLLM base RoPE through Megatron's unfused RoPE formula."""
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    current_forward = RotaryEmbedding.forward_cuda
    original_forward = getattr(
        current_forward,
        G_ORIGINAL_FORWARD_ATTR,
        current_forward,
    )
    already_installed = bool(
        getattr(current_forward, G_MEGATRON_ROPE_PATCH_MARKER_ATTR, False)
    )

    if not already_installed:

        def patched_forward_cuda(self, positions, query, key=None):
            if self.use_flashinfer:
                return original_forward(self, positions, query, key)

            query = _apply_megatron_style_rope(
                module=self,
                positions=positions,
                tensor=query,
            )
            if key is not None:
                key = _apply_megatron_style_rope(
                    module=self,
                    positions=positions,
                    tensor=key,
                )
            return query, key

        setattr(patched_forward_cuda, G_MEGATRON_ROPE_PATCH_MARKER_ATTR, True)
        setattr(patched_forward_cuda, G_ORIGINAL_FORWARD_ATTR, original_forward)
        RotaryEmbedding.forward_cuda = patched_forward_cuda

    rebound_count = _rebind_custom_op_forward_methods(model, RotaryEmbedding)

    return {
        "already_installed": already_installed,
        "rebound_count": rebound_count,
    }


def install_megatron_style_swiglu_patch(model: torch.nn.Module) -> dict[str, Any]:
    """Route vLLM ``SiluAndMul`` through Megatron's unfused SwiGLU path."""
    import torch.nn.functional as F
    from vllm.model_executor.layers.activation import SiluAndMul

    current_forward = SiluAndMul.forward_cuda
    original_forward = getattr(
        current_forward,
        G_ORIGINAL_FORWARD_ATTR,
        current_forward,
    )
    already_installed = bool(
        getattr(current_forward, G_MEGATRON_SWIGLU_PATCH_MARKER_ATTR, False)
    )

    if not already_installed:

        def patched_forward_cuda(self, x):
            x_glu, x_linear = torch.chunk(x, 2, dim=-1)
            return F.silu(x_glu) * x_linear

        setattr(patched_forward_cuda, G_MEGATRON_SWIGLU_PATCH_MARKER_ATTR, True)
        setattr(patched_forward_cuda, G_ORIGINAL_FORWARD_ATTR, original_forward)
        SiluAndMul.forward_cuda = patched_forward_cuda

    rebound_count = _rebind_custom_op_forward_methods(model, SiluAndMul)

    return {
        "already_installed": already_installed,
        "rebound_count": rebound_count,
    }


def install_true_on_policy_patch_components(
    model: torch.nn.Module,
    components: tuple[str, ...],
) -> dict[str, Any]:
    """Install selected BF16 true-on-policy vLLM patches for diagnostics."""
    requested_components = set(components)
    unknown_components = (
        requested_components - G_TRUE_ON_POLICY_BF16_PATCH_COMPONENT_SET
    )
    if unknown_components:
        raise ValueError(
            "Unknown vLLM true-on-policy patch components: "
            f"{sorted(unknown_components)}. Expected subset of "
            f"{list(G_TRUE_ON_POLICY_BF16_PATCH_COMPONENTS)}."
        )

    results: dict[str, Any] = {}
    if "rmsnorm" in requested_components:
        results["megatron_style_rmsnorm"] = install_megatron_style_rmsnorm_patch(model)
    if "rope" in requested_components:
        results["megatron_style_rope"] = install_megatron_style_rope_patch(model)
    if "swiglu" in requested_components:
        results["megatron_style_swiglu"] = install_megatron_style_swiglu_patch(model)
    return results


def _get_requested_true_on_policy_components() -> tuple[str, ...]:
    raw_components = os.environ.get(G_TRUE_ON_POLICY_COMPONENTS_ENV)
    if raw_components is None:
        return G_TRUE_ON_POLICY_BF16_PATCH_COMPONENTS
    if raw_components.strip() == "":
        return ()
    return tuple(
        component.strip().lower()
        for component in raw_components.split(",")
        if component.strip()
    )


def install_true_on_policy_patches(
    model: torch.nn.Module,
    *,
    bf16_true_on_policy: bool,
) -> dict[str, Any]:
    """Install vLLM BF16 true-on-policy patches controlled by policy flags."""
    if not bf16_true_on_policy:
        return {}

    components = _get_requested_true_on_policy_components()
    results: dict[str, Any] = {"bf16_components": components}
    results.update(
        install_true_on_policy_patch_components(
            model,
            components,
        )
    )
    return results
