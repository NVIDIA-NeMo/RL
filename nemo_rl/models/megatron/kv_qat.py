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

"""KV-cache QAT matching vLLM's ``fp8_per_token_head`` cache.

vLLM stores each token's K and V per KV head in E4M3 with a dynamic FP32 scale
``max(amax(head_dim) / 448, 1e-6)`` and attends over the dequantized values with
BF16 queries. The trainer reproduces that rounding: a forward pre-hook on every
Megatron ``core_attention`` (called after QK norm and RoPE) quantizes and
dequantizes K and V the same way, with a straight-through gradient. Q and the
attention arithmetic stay in BF16.
"""

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from nemo_rl.models.policy import PolicyConfig

E4M3_MAX = 448.0


class _PerTokenHeadFP8(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:
        # Mirrors vLLM's Triton cache writer: amax over head_dim only, FP32
        # scale with a 1e-6 floor, multiply by the reciprocal, saturate.
        x = value.float()
        scale = (x.abs().amax(dim=-1, keepdim=True) / E4M3_MAX).clamp_min(1e-6)
        quantized = (x * scale.reciprocal()).clamp(-E4M3_MAX, E4M3_MAX)
        return (quantized.to(torch.float8_e4m3fn).float() * scale).to(value.dtype)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        return grad


def fake_quantize_per_token_head(value: torch.Tensor) -> torch.Tensor:
    """E4M3 quantize-dequantize over the last (head_dim) axis with an identity STE."""
    return _PerTokenHeadFP8.apply(value)


def validate_kv_fp8_qat_config(config: "PolicyConfig") -> None:
    """Require the rollout cache that this QAT reproduces."""
    vllm_cfg = config["generation"]["vllm_cfg"]
    vllm_kwargs = config["generation"].get("vllm_kwargs") or {}
    attention = vllm_kwargs.get("attention_config") or {}
    if (
        config["generation"]["backend"] != "vllm"
        or vllm_cfg["kv_cache_dtype"] != "fp8_per_token_head"
        or attention.get("backend") != "TRITON_ATTN"
    ):
        raise ValueError(
            "megatron_cfg.kv_fp8_qat requires vLLM generation with "
            "kv_cache_dtype=fp8_per_token_head and "
            "vllm_kwargs.attention_config.backend=TRITON_ATTN"
        )
    if config["megatron_cfg"].get("cuda_graph_impl") not in (None, "none"):
        raise ValueError("megatron_cfg.kv_fp8_qat requires cuda_graph_impl=none")


class KVFP8QAT:
    """Parameter-free K/V quantize-dequantize hooks on every core attention.

    Checkpoint and refit names are unchanged. Set ``enabled=False`` to run the
    original BF16 attention (used for reference-policy log-probabilities).
    """

    def __init__(self, model: torch.nn.Module | list[torch.nn.Module]):
        modules = [
            module
            for chunk in (model if isinstance(model, list) else [model])
            for name, module in chunk.named_modules()
            if name.endswith("self_attention.core_attention")
        ]
        if not modules:
            raise ValueError("KV QAT found no self_attention.core_attention modules")
        self.enabled = True
        self.handles = [
            module.register_forward_pre_hook(self._quantize, with_kwargs=True)
            for module in modules
        ]

    def _quantize(
        self, module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not self.enabled:
            return args, kwargs
        # core_attention(query, key, value, ...): K and V are positional 1 and 2
        # or keyword arguments, depending on the Megatron call site.
        args = list(args)
        kwargs = dict(kwargs)
        for index, name in ((1, "key"), (2, "value")):
            if index < len(args):
                args[index] = fake_quantize_per_token_head(args[index])
            elif name in kwargs:
                kwargs[name] = fake_quantize_per_token_head(kwargs[name])
            else:
                raise ValueError(f"KV QAT requires core_attention argument {name!r}")
        return tuple(args), kwargs
