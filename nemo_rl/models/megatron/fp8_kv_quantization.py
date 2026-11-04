# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training-side FP8 E4M3 K/V fake quantization with per-token-head scales."""

from collections.abc import Mapping
import os
from typing import Any

import torch
from torch import nn

FP8_KV_IN_ATTENTION_ENV_VAR = "NRL_MEGATRON_FP8_KV_IN_ATTENTION"


def fp8_kv_attention_env_vars(config_env: Mapping[str, str] | None) -> dict[str, str]:
    """Forward the opt-in switch to Ray without overriding explicit worker env."""
    env = dict(config_env or {})
    if FP8_KV_IN_ATTENTION_ENV_VAR in os.environ:
        env.setdefault(
            FP8_KV_IN_ATTENTION_ENV_VAR, os.environ[FP8_KV_IN_ATTENTION_ENV_VAR]
        )
    _fp8_kv_in_attention_setting(env)
    return env


def _fp8_kv_in_attention_setting(env: Mapping[str, str]) -> bool:
    setting = env.get(FP8_KV_IN_ATTENTION_ENV_VAR, "0")
    if setting not in ("0", "1"):
        raise ValueError(f"{FP8_KV_IN_ATTENTION_ENV_VAR} must be 0 or 1")
    return setting == "1"


def fp8_kv_in_attention_enabled() -> bool:
    return _fp8_kv_in_attention_setting(os.environ)


def quantize_kv_per_token_head(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E4M3 values and FP32 scales, reducing only the head dimension.

    Accepts Megatron's SBHD and packed THD layouts, including noncontiguous
    views. The formula matches vLLM's per-token-head cache-write kernel.
    Scales are independent for every token, batch item, and KV head.
    """
    if tensor.ndim not in (3, 4) or tensor.shape[-1] == 0:
        raise ValueError("FP8 K/V quantization requires THD or SBHD tensors")
    if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("FP8 K/V quantization requires floating-point K/V inputs")
    values = tensor.detach().to(torch.float32)
    scales = (values.abs().amax(dim=-1, keepdim=True) / 448.0).clamp_min(1e-6)
    normalized = (values * scales.reciprocal()).clamp(-448.0, 448.0)
    return normalized.to(torch.float8_e4m3fn), scales


class _FakeQuantizeKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor) -> torch.Tensor:
        quantized, scales = quantize_kv_per_token_head(tensor)
        return (quantized.to(torch.float32) * scales).to(tensor.dtype).contiguous()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        # Straight-through estimator: do not differentiate amax, scales or casts.
        return grad_output


def fake_quantize_kv_per_token_head(tensor: torch.Tensor) -> torch.Tensor:
    """Apply dynamic E4M3 quantize/dequantize with identity backward."""
    return _FakeQuantizeKV.apply(tensor)


def fp8_kv_attention_pre_hook(
    module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Quantize K/V immediately before core attention; leave Q untouched.

    This runs after Q/K normalization and any rotary embeddings, including
    during activation recomputation and no-grad policy logprob evaluation.
    """
    updated_args = list(args)
    updated_kwargs = dict(kwargs)
    for index, name in ((1, "key"), (2, "value")):
        if len(updated_args) > index:
            updated_args[index] = fake_quantize_kv_per_token_head(updated_args[index])
        elif name in updated_kwargs:
            updated_kwargs[name] = fake_quantize_kv_per_token_head(updated_kwargs[name])
        else:
            raise ValueError(f"Core attention did not receive {name}")
    return tuple(updated_args), updated_kwargs


def install_fp8_kv_quantization(models: list[nn.Module]) -> list[nn.Module]:
    """Install one of two exclusive FP8 KV paths before DDP/FSDP wrapping.

    Context and tensor parallelism do not change the reduction: every local
    token/head has a complete head dimension. Pipeline stages without attention
    legitimately install zero hooks. MLA/cross attention are not supported.
    """
    # Megatron-Core is available only in Megatron policy worker environments.
    from megatron.core.transformer.attention import Attention, SelfAttention

    in_attention = fp8_kv_in_attention_enabled()
    if in_attention:
        from nemo_rl.models.megatron.fp8_kv_attention import (
            FP8KVScaledAttention,
            FP8KVScaledAttentionWithExtraState,
        )

    count = 0
    seen: set[nn.Module] = set()
    replacements: dict[nn.Module, nn.Module] = {}
    for model in models:
        for module in model.modules():
            if isinstance(module, Attention) and not isinstance(module, SelfAttention):
                raise ValueError(
                    "fp8_kv_per_token supports standard Megatron SelfAttention only; "
                    f"got {type(module).__name__}"
                )
            if isinstance(module, SelfAttention):
                core = module.core_attention
                if in_attention and isinstance(core, FP8KVScaledAttention):
                    continue
                if core in seen:
                    if in_attention:
                        module.core_attention = replacements[core]
                    continue
                seen.add(core)
                if in_attention:
                    if any(True for _ in core.parameters()) or any(
                        True for _ in core.buffers()
                    ):
                        raise ValueError(
                            "in-attention FP8 KV cannot replace attention with tensor state"
                        )
                    args = (
                        module.config,
                        module.pg_collection.cp,
                        getattr(core, "softmax_scale", None),
                    )
                    if type(core).get_extra_state is not nn.Module.get_extra_state:
                        replacement = FP8KVScaledAttentionWithExtraState(
                            *args,
                            core.get_extra_state,
                            core.set_extra_state,
                        )
                    else:
                        replacement = FP8KVScaledAttention(*args)
                    replacement.train(core.training)
                    replacements[core] = replacement
                    # The old core (and any legacy dequantization pre-hook) is
                    # not called by this replacement. The paths are exclusive.
                    module.core_attention = replacement
                else:
                    core.register_forward_pre_hook(
                        fp8_kv_attention_pre_hook, with_kwargs=True
                    )
                count += 1
    print(
        "[fp8_kv_per_token] Installed dynamic E4M3 K/V quantization "
        f"on {count} attention modules",
        flush=True,
    )
    if in_attention:
        print(
            f"[{FP8_KV_IN_ATTENTION_ENV_VAR}=1] K scales after QK; V scales on softmax weights; BF16 KV dequantization hook disabled. Experimental chunked PyTorch attention backend.",
            flush=True,
        )
    return models


def fp8_kv_quantization_enabled(megatron_cfg: Mapping[str, Any]) -> bool:
    """Validate and resolve the optional training-side quantization switch."""
    enabled = megatron_cfg.get("fp8_kv_per_token")
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("policy.megatron_cfg.fp8_kv_per_token must be a boolean")
    in_attention = fp8_kv_in_attention_enabled()
    return enabled is True or in_attention
