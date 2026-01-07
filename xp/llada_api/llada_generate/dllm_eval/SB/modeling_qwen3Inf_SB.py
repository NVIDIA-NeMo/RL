# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the
# HuggingFace Inc. team. All rights reserved.
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

from typing import Callable, Optional, Tuple, TypedDict, Union
from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple, logging

# SDPA (PyTorch fused attention)
from transformers.integrations.sdpa_attention import sdpa_attention_forward
# FlashAttention-2
from transformers.integrations.flash_attention import flash_attention_forward
# FlexAttention
from transformers.integrations.flex_attention import flex_attention_forward

flash_attention_2_forward = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

# Compatibility shim for LossKwargs (removed in some transformers versions)
try:
    from transformers.utils import LossKwargs  # type: ignore
except Exception:
    class LossKwargs(TypedDict):  # type: ignore
        ...


# Import config from root model directory (model_path is added to sys.path in chat.py)
from configuration_nvrdiff import NVRDiffConfig

logger = logging.get_logger(__name__)


@torch.inference_mode()
def apply_rotary_pos_emb_ref(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*): Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using RoPE.
    """
    #print(f"DataType of q: {q.dtype}, DataType of k: {k.dtype}, DataType of cos: {cos.dtype}, DataType of sin: {sin.dtype}")
    
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(q.dtype)
    k_embed = k_embed.to(k.dtype)
    #print(f"DataType of q_embed: {q_embed.dtype}, DataType of k_embed: {k_embed.dtype}")
    return q_embed, k_embed


# The following two decorators yield best and comparable performance
# @torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
@torch.compile(mode="default", dynamic=True)
#@torch.inference_mode()
def apply_rotary_pos_emb(q: torch.Tensor,
                        k: torch.Tensor,
                        cos: torch.Tensor,
                        sin: torch.Tensor,
                        unsqueeze_dim: int = 1):
    """
    RoPE in fp32 for stable numerics, then cast results back to q/k dtype (typically bfloat16).
    """
    # ensure compute uses fp32
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    half = q.shape[-1] // 2
    cos = cos.view(*cos.shape[:-1], 2, half)
    sin = sin.view(*sin.shape[:-1], 2, half)

    def apply_rotary(x: torch.Tensor) -> torch.Tensor:
        x0, x1 = x.view(*x.shape[:-1], 2, half).unbind(dim=-2)
        cos0, cos1 = cos[..., 0, :], cos[..., 1, :]
        sin0, sin1 = sin[..., 0, :], sin[..., 1, :]

        out0 = x0 * cos0 - x1 * sin0
        out1 = x1 * cos1 + x0 * sin1
        return torch.stack((out0, out1), dim=-2).reshape_as(x)

    q_embed = apply_rotary(q)
    k_embed = apply_rotary(k)
    q_embed = q_embed.to(q.dtype)
    k_embed = k_embed.to(k.dtype)
    #print(f"DataType of q_embed: {q_embed.dtype}, DataType of k_embed: {k_embed.dtype}")
    return q_embed, k_embed


#@torch.inference_mode()
@torch.compile(dynamic=True, mode="default", fullgraph=False)
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    (batch, num_kv_heads, seqlen, head_dim) -> (batch, num_attn_heads, seqlen, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    return hidden_states.repeat_interleave(n_rep, dim=1)


@use_kernel_forward_from_hub("RMSNorm")
class Qwen3InfRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3InfRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    #@torch.compile(dynamic=True, mode="default", fullgraph=False)
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    @torch.inference_mode()
    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3InfMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    #@torch.compile(dynamic=True, mode="default", fullgraph=False)
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3InfAttention(nn.Module):
    """
    Multi-head self-attention with two cache modes:

      • attn_mode=0 (baseline): standard behavior.
        - use_cache=False → concat past K/V for compute only (no cache mutation)
        - use_cache=True  → commit the entire current window to the KV cache

      • attn_mode=1 (fused): commit only a left prefix of the current window
        (length = committed_prefix_len) and ATTEND over [past | tail] in a single call.
        This is used by the double-block "fused commit+denoise" generation scheme.
    """

    def __init__(self, config: NVRDiffConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout

        # diffusion LMs often prefer no mask; keep a switch handy
        self.diffusion_lm = config.diffusion_lm
        self.is_causal = None if not self.diffusion_lm else False

        # Projections
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # Fused-projection buffers (lazy initialized on first use)
        self.register_buffer("_qkv_weight", torch.empty(0), persistent=False)
        self.register_buffer("_qkv_bias", torch.empty(0), persistent=False)
        self._qkv_needs_refresh = True
        self._q_weight_version = -1
        self._k_weight_version = -1
        self._v_weight_version = -1
        self._q_bias_version = -1 if self.q_proj.bias is not None else None
        self._k_bias_version = -1 if self.k_proj.bias is not None else None
        self._v_bias_version = -1 if self.v_proj.bias is not None else None

        # Reusable buffers to avoid reallocating when merging cached + current KV
        self.register_buffer("_merged_k_buffer", torch.empty(0), persistent=False)
        self.register_buffer("_merged_v_buffer", torch.empty(0), persistent=False)

        # Optional Q/K RMSNorm on the head dim
        if not config.disable_qk_norm:
            self.q_norm = Qwen3InfRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen3InfRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Sliding window (enabled only on some layers)
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None
 
        # Preallocate buffers 
        # Cached min for bf16 additive masks
        self._bf16_min = torch.finfo(torch.bfloat16).min

        self._refresh_qkv_weights(force=True)

    @torch.inference_mode()
    def _refresh_qkv_weights(self, force: bool = False) -> None:
        if self.q_proj.weight.is_meta:
            if force or self._qkv_weight.numel() != 0:
                self._qkv_weight = torch.empty(0)
                self._qkv_bias = torch.empty(0)
            self._qkv_needs_refresh = True
            return
 
        # Helper to safely get tensor version (returns -1 if inference tensor)
        def _safe_version(tensor):
            try:
                return tensor._version
            except RuntimeError:
                # Inference tensors don't track version counter
                return -1

        need_refresh = force or (
            self._qkv_needs_refresh
            or self._qkv_weight.numel() == 0
            or self._q_weight_version != _safe_version(self.q_proj.weight)
            or self._k_weight_version != _safe_version(self.k_proj.weight)
            or self._v_weight_version != _safe_version(self.v_proj.weight)
            or (self.q_proj.bias is not None and self._q_bias_version != _safe_version(self.q_proj.bias))
            or (self.k_proj.bias is not None and self._k_bias_version != _safe_version(self.k_proj.bias))
            or (self.v_proj.bias is not None and self._v_bias_version != _safe_version(self.v_proj.bias))
        )

        if not need_refresh:
            return

        with torch.inference_mode():
            weight_device = self.q_proj.weight.device
            weight_dtype = self.q_proj.weight.dtype
            self._qkv_weight = torch.cat(
                [
                    self.q_proj.weight.to(weight_device),
                    self.k_proj.weight.to(weight_device),
                    self.v_proj.weight.to(weight_device),
                ],
                dim=0,
            ).to(weight_dtype)
            if self.q_proj.bias is not None:
                self._qkv_bias = torch.cat(
                    [
                        self.q_proj.bias.to(weight_device),
                        self.k_proj.bias.to(weight_device),
                        self.v_proj.bias.to(weight_device),
                    ],
                    dim=0,
                ).to(self.q_proj.bias.dtype)
            else:
                self._qkv_bias = torch.empty(0, device=weight_device, dtype=weight_dtype)

        self._qkv_needs_refresh = False
        self._q_weight_version = _safe_version(self.q_proj.weight)
        self._k_weight_version = _safe_version(self.k_proj.weight)
        self._v_weight_version = _safe_version(self.v_proj.weight)
        if self.q_proj.bias is not None:
            self._q_bias_version = _safe_version(self.q_proj.bias)
            self._k_bias_version = _safe_version(self.k_proj.bias)
            self._v_bias_version = _safe_version(self.v_proj.bias)
        else:
            self._q_bias_version = self._k_bias_version = self._v_bias_version = None

        if self.q_proj.bias is None:
            self._qkv_bias = torch.empty(0, device=weight_device, dtype=weight_dtype)
        else:
            self._qkv_bias = self._qkv_bias.to(self.q_proj.bias.dtype)

    @torch.compile(dynamic=True, mode="default", fullgraph=False)
    def _merge_kv(self, past: Optional[torch.Tensor], current: torch.Tensor, buffer_name: str) -> torch.Tensor:
        """
        Merge cached KV with the current window while reusing a preallocated buffer when possible.
        """
        if past is None or past.shape[-2] == 0:
            return current
        if current.shape[-2] == 0:
            return past

        B, H, _, Dh = past.shape
        tail = current.shape[-2]
        total_len = past.shape[-2] + tail

        buffer = getattr(self, buffer_name)
        needs_new_buffer = (
            buffer.numel() == 0
            or buffer.dtype != past.dtype
            or buffer.device != past.device
            or buffer.shape[0] < B
            or buffer.shape[1] < H
            or buffer.shape[3] != Dh
            or buffer.shape[2] < total_len
        )

        if needs_new_buffer:
            buffer = torch.empty((B, H, total_len, Dh), dtype=past.dtype, device=past.device)
        else:
            buffer = buffer[:B, :H, :total_len, :Dh]

        buffer[..., : past.shape[-2], :] = past
        buffer[..., past.shape[-2] :, :] = current
        setattr(self, buffer_name, buffer)
        return buffer

    #@torch.compile(dynamic=True, mode="default", fullgraph=False)
    def _attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        #with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True): 
        '''
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
            enable_gqa=True,
        )
        return attn_output.transpose(1, 2).contiguous(), None
        '''
        return sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            is_causal=False,
            **kwargs,
        )

    @torch.compile(dynamic=True, mode="default", fullgraph=False)
    def _fused_qkv_attention(self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        ):
        """Project hidden states into fused Q/K/V and reshape for attention."""
        B, S, _ = hidden_states.shape
        Hq  = self.num_heads
        Hkv = self.num_key_value_heads
        Dh  = self.head_dim
        assert Hq % Hkv == 0, "num_heads must be divisible by num_key_value_heads"
 
        self._refresh_qkv_weights()
 
        W = self._qkv_weight
        b = self._qkv_bias if self._qkv_bias.numel() else None
 
        # single linear -> [B, S, (Hq + 2*Hkv) * Dh]
        qkv = F.linear(hidden_states, W, b).view(B, S, Hq + 2 * Hkv, Dh)
        q, k, v = torch.split(qkv, [Hq, Hkv, Hkv], dim=2)

        q = self.q_norm(q).transpose(1, 2).contiguous()
        k = self.k_norm(k).transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        return q, k, v

    @torch.inference_mode()
    #@torch.compile(dynamic=True, mode="default", fullgraph=False)
    def forward_fused_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        replace_position: Optional[torch.Tensor] = None,
        is_training: bool = True,
        use_cache: bool = False,  
        position_ids: Optional[torch.LongTensor] = None,               
        sequence_length: Optional[int] = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        input_shape = hidden_states.shape[:-1]

        query_states, key_states, value_states = self._fused_qkv_attention(hidden_states, position_embeddings)

        past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            cache_kwargs={"cache_position": cache_position},
        )
        
        past_k, past_v = past_key_value[self.layer_idx] 
        key_states = past_k[:, :, :sequence_length, :]                
        value_states = past_v[:, :, :sequence_length, :]
        
        attn_output, attn_weights = sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            is_causal=False,
            **kwargs,
        )
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    @torch.compile(dynamic=True, mode="default", fullgraph=False)
    def _separate_qkv_projection(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        input_shape: torch.Tensor,
    ):
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        return query_states, key_states, value_states        
 
    @torch.inference_mode()
    #@torch.compile(dynamic=True, mode="reduce-overhead", fullgraph=False)
    def forward_separate_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        replace_position: Optional[torch.Tensor] = None,
        is_training: bool = True,
        use_cache: bool = False,  
        position_ids: Optional[torch.LongTensor] = None,               
        sequence_length: Optional[int] = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        input_shape = hidden_states.shape[:-1]

        query_states, key_states, value_states = self._separate_qkv_projection(hidden_states, position_embeddings, input_shape)

        # KV cache update cannot be compiled well
        past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            cache_kwargs={"cache_position": cache_position},
        )
        
        past_k, past_v = past_key_value[self.layer_idx] 
        key_states = past_k[:, :, :sequence_length, :]                
        value_states = past_v[:, :, :sequence_length, :]
        
        attn_output, attn_weights = sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            is_causal=False,
            **kwargs,
        )
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    
    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        replace_position: Optional[torch.Tensor] = None,
        is_training: bool = True,
        use_cache: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: Optional[int] = 0,                          # 0 = baseline, 1 = fused commit-prefix
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.forward_separate_qkv(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value,
            cache_position,
            replace_position,
            is_training,
            use_cache,
            position_ids,
            sequence_length,
            **kwargs,
        )

class Qwen3InfDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: NVRDiffConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        attn_class = getattr(config, 'attn_class', Qwen3InfAttention)
        self.layer_idx = layer_idx

        self.self_attn = attn_class(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3InfMLP(config)
        self.input_layernorm = Qwen3InfRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3InfRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
    # Even more perf
    #@torch.compile(dynamic=True, mode="max-autotune-no-cudagraphs", fullgraph=False)
    @torch.compile(dynamic=True, mode="default", fullgraph=False)
    def _ffn_compile(self, residual: torch.Tensor, hidden_states: torch.Tensor):
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        replace_position: Optional[torch.Tensor] = None,
        is_training: bool = True,
        committed_prefix_len: Optional[int] = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            replace_position=replace_position,
            is_training=is_training,
            committed_prefix_len=committed_prefix_len,
            **kwargs,
        )
        
        # Fused improves throughput while sacrificing accuracy
        '''
        hidden_states = self._ffn_compile(residual, hidden_states)
        return (hidden_states,)
        '''
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return (hidden_states,)

        '''
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
        '''


@auto_docstring
class Qwen3InfPreTrainedModel(PreTrainedModel):
    config_class = NVRDiffConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3InfDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3InfRMSNorm):
            module.weight.data.fill_(1.0)


class Qwen3InfRotaryEmbedding(nn.Module):
    def __init__(self, config: NVRDiffConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.inference_mode()
    @dynamic_rope_update  # advanced RoPE types support (e.g., dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32 compute
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class Qwen3InfModel(Qwen3InfPreTrainedModel):
    def __init__(self, config: NVRDiffConfig):
        super().__init__(config)
        self.config = config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3InfDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3InfRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3InfRotaryEmbedding(config=config)

        max_commit_batch = getattr(config, "commit_mask_max_batch", 1) or 1
        max_commit_block = getattr(config, "commit_mask_max_block", 128) or 128
        max_commit_total = getattr(config, "commit_mask_max_total_len", 4096) or 4096

        assert (
            max_commit_batch > 0 and max_commit_block > 0 and max_commit_total > 0
        ), "commit_mask_max_* must be > 0 to preallocate commit-mask buffer"

        mask_dtype = getattr(config, "torch_dtype", None) or torch.bfloat16
        mask_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        commit_shape = (max_commit_batch, 1, max_commit_block, max_commit_total)
        self.register_buffer(
            "_commit_mask_buffer",
            torch.zeros(commit_shape, dtype=mask_dtype, device=mask_device),
            persistent=False,
        )

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        try:
            attn_name = type(self.layers[0].self_attn).__name__
            paradigm = getattr(config, "dlm_paradigm", None)
            impl = getattr(config, "attn_implementation", getattr(config, "_attn_implementation", None))
            print(f"[Qwen3InfModel] Initialized with attention={attn_name}, paradigm={paradigm}, impl={impl}")
        except Exception:
            pass

    @torch.inference_mode()
    def get_input_embeddings(self):
        return self.embed_tokens

    @torch.inference_mode()
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @torch.inference_mode()
    def _generate_commit_mask(self, past_cache_length, committed_prefix_len, block_length, batch_count, device, dtype, past_key_values):
        if committed_prefix_len > 0 and past_key_values is not None:
            # Get past key from first layer to check dimensions
            try:
                past_k, past_v = past_key_values[0]
            except Exception:
                return None

            if past_k is None or past_v is None:
                return None

            if past_k.numel() == 0 or past_v.numel() == 0:
                return None

            # The cache lenght is tracked at the generate function
            # We later must extend the generate function to track that at batch granularity
            past_k_len = past_cache_length
            total_k_len = block_length + past_k_len

            if committed_prefix_len <= block_length and past_k_len < total_k_len:
                B = batch_count
                Q = block_length
                min_val = torch.finfo(dtype).min

                buffer = self._commit_mask_buffer

                if B > buffer.shape[0] or Q > buffer.shape[2] or total_k_len > buffer.shape[3]:
                    raise ValueError(
                        "Commit mask dimensions exceed preallocated buffer. "
                        f"Requested (batch={B}, block={Q}, total={total_k_len}) "
                        f"but buffer max is {buffer.shape}."
                    )

                commit_mask = buffer[:B, :1, :Q, :total_k_len]
                commit_mask.zero_()

                tail_start = past_k_len + committed_prefix_len
                if tail_start < total_k_len:
                    commit_mask[:, :, :committed_prefix_len, tail_start:] = min_val

                return commit_mask
        return None

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        replace_position: Optional[torch.Tensor] = None,
        is_training: bool = True,
        committed_prefix_len: Optional[int] = 0,
        kv_cache_pos_offset: Optional[int] = None,
        past_cache_length: Optional[int] = 0,
        block_length: Optional[int] = 0,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            raise ValueError("`past_key_values` must be provided when `use_cache=True`.")

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
            
        if block_length != inputs_embeds.shape[1]:
            print(f"WARNING: block_length != inputs_embeds.shape[1]: {block_length} != {inputs_embeds.shape[1]}")

        sequence_length = past_cache_length + block_length
        
        max_index = cache_position.max().item()
        max_index = max_index + 1
        
        if max_index != sequence_length:
            print(f"WARNING: max_index != sequence_length: {max_index} != {sequence_length}")
            
        causal_mask = attention_mask
        if causal_mask is None and committed_prefix_len > 0 and past_key_values is not None:
            causal_mask = self._generate_commit_mask(
                past_cache_length=past_cache_length,
                committed_prefix_len=committed_prefix_len,
                block_length=inputs_embeds.shape[1],
                batch_count=inputs_embeds.shape[0],
                device=self._commit_mask_buffer.device,
                dtype=self._commit_mask_buffer.dtype,
                past_key_values=past_key_values,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                replace_position=replace_position,
                is_training=is_training,
                sequence_length=max_index,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_values_output = None
        if use_cache and past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_key_values_output = []
                if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
                    for layer_idx in range(len(past_key_values.key_cache)):
                        past_key_values_output.append((
                            past_key_values.key_cache[layer_idx],
                            past_key_values.value_cache[layer_idx]
                        ))
                else:
                    past_key_values_output = past_key_values
            else:
                past_key_values_output = past_key_values

        # If not returning attentions, pack position embeddings in the attentions field (BC / lightweight)
        attentions_or_embeddings = all_self_attns
        if not output_attentions:
            attentions_or_embeddings = position_embeddings

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values_output,
            hidden_states=all_hidden_states,
            attentions=attentions_or_embeddings,
        )




class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs):
    ...


@auto_docstring
class Qwen3InfForCausalLM(Qwen3InfPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        try:
            if getattr(config, "attn_implementation", None) is not None:
                config._attn_implementation = config.attn_implementation
        except Exception:
            pass

        self.model = Qwen3InfModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @torch.inference_mode()
    def get_input_embeddings(self):
        return self.model.embed_tokens

    @torch.inference_mode()
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @torch.inference_mode()
    def get_output_embeddings(self):
        return self.lm_head

    @torch.inference_mode()
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @torch.inference_mode()
    def set_decoder(self, decoder):
        self.model = decoder

    @torch.inference_mode()
    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        replace_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            replace_position=replace_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The Qwen3Inf Model transformer with a sequence classification head on top (linear layer).
    [`Qwen3InfForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.
    """
)
class Qwen3InfForSequenceClassification(Qwen3InfPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen3InfModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    @torch.inference_mode()
    def get_input_embeddings(self):
        return self.model.embed_tokens

    @torch.inference_mode()
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceClassifierOutputWithPast:

        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`."
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@auto_docstring
class Qwen3InfForTokenClassification(Qwen3InfPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen3InfModel(config)
        classifier_dropout = (
            getattr(config, "classifier_dropout", None)
            if getattr(config, "classifier_dropout", None) is not None
            else getattr(config, "hidden_dropout", 0.1)
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @torch.inference_mode()
    def get_input_embeddings(self):
        return self.model.embed_tokens

    @torch.inference_mode()
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> TokenClassifierOutput:

        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class Qwen3InfForQuestionAnswering(Qwen3InfPreTrainedModel):
    base_model_prefix = "transformer"

    def __init__(self, config):
        super().__init__(config)
        self.transformer = Qwen3InfModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.post_init()

    @torch.inference_mode()
    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    @torch.inference_mode()
    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> QuestionAnsweringModelOutput:

        outputs: BaseModelOutputWithPast = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs.last_hidden_state
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen3InfDiffusionLM(Qwen3InfForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.mask_token_id = 151662  # [MASK]

    @torch.inference_mode()
    def forward_process(self, input_ids, eps=1e-3):
        b, l = input_ids.shape
        t = torch.rand(b, device=input_ids.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)
        masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
        noisy_batch = torch.where(masked_indices, self.mask_token_id, input_ids)
        return noisy_batch, masked_indices, p_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        replace_position: Optional[torch.Tensor] = None,
        eps: float = 1e-3,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:

        if labels is not None:
            if getattr(self.config, "random_length_prob", None) is not None:
                if torch.rand(1) < self.config.random_length_prob:
                    random_length = torch.randint(2, input_ids.shape[1] + 1, (1,))
                    input_ids = input_ids[:, :random_length]
                    labels = labels[:, :random_length]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, :random_length]
                    if position_ids is not None:
                        position_ids = position_ids[:, :random_length]

            noisy_batch, masked_indices, p_mask = self.forward_process(input_ids, eps)
        else:
            noisy_batch = input_ids
            masked_indices = None
            p_mask = None

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=noisy_batch,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            replace_position=replace_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            if getattr(self.config, "dlm_type", None) == 'dream':
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
                masked_indices = masked_indices[:, 1:]
                p_mask = p_mask[:, 1:]
            token_loss = torch.nn.functional.cross_entropy(
                logits[masked_indices], labels[masked_indices], reduction='none'
            ) / p_mask[masked_indices]
            loss = token_loss.sum() / masked_indices.sum()

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Qwen3InfForCausalLM",
    "Qwen3InfForQuestionAnswering",
    "Qwen3InfModel",
    "Qwen3InfPreTrainedModel",
    "Qwen3InfForSequenceClassification",
    "Qwen3InfForTokenClassification",
    "Qwen3InfDiffusionLM",
]

def verify_compile_functions(model):
    ''' This function should be used to verify that every function in the model can be compiled. '''
    ''' This will simplify development and debugging by allowing us to use the compiled model without worrying about errors. '''
    
    try:
        compiled_model = torch.compile(model, fullgraph=True, mode="default").eval()
        print("Model compiled successfully with torch.compile")
    except Exception as e:
        print(f"Warning: torch.compile failed with error: {type(e).__name__}")
        print("Falling back to eager mode execution")
        compiled_model = model.eval()