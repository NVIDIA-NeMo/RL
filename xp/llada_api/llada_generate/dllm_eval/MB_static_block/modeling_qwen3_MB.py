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

# PyTorch native FlexAttention (PyTorch 2.5+)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    create_block_mask = None

# Custom Triton FlashAttention with shared mask support
# NVIDIA Transformer Engine for efficient attention with arbitrary masks
try:
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch import DotProductAttention as TEDotProductAttention
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    TEDotProductAttention = None

flash_attention_2_forward = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

# Direct FlashAttention-2 import for optimized forward
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None
    flash_attn_varlen_func = None

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


def apply_rotary_pos_emb(q: torch.Tensor,
                         k: torch.Tensor,
                         cos: torch.Tensor,
                         sin: torch.Tensor,
                         unsqueeze_dim: int = 1):
    """
    Optimized PyTorch RoPE (Rotary Position Embedding) implementation.
    
    Key optimizations:
    - Uses slice indexing instead of unbind() to avoid copies
    - Pre-allocates output tensor and writes directly to it
    - Avoids torch.stack() which allocates new memory
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    half = q.shape[-1] // 2
    cos = cos.view(*cos.shape[:-1], 2, half)
    sin = sin.view(*sin.shape[:-1], 2, half)
    
    # Pre-extract cos/sin components (views, no copy)
    cos0, cos1 = cos[..., 0, :], cos[..., 1, :]
    sin0, sin1 = sin[..., 0, :], sin[..., 1, :]

    def apply_rotary_optimized(x: torch.Tensor) -> torch.Tensor:
        # Reshape to [*, 2, half] - this is a view if contiguous
        x_reshaped = x.view(*x.shape[:-1], 2, half)
        
        # Use slice indexing instead of unbind (views, no copy)
        x0 = x_reshaped[..., 0, :]
        x1 = x_reshaped[..., 1, :]
        
        # Pre-allocate output tensor (single allocation)
        out = torch.empty_like(x_reshaped)
        
        # Write directly to output slices (no intermediate allocation)
        torch.mul(x0, cos0, out=out[..., 0, :])
        out[..., 0, :].sub_(x1 * sin0)
        
        torch.mul(x1, cos1, out=out[..., 1, :])
        out[..., 1, :].add_(x0 * sin1)
        
        # Reshape back to original shape (view, no copy if contiguous)
        return out.view_as(x)

    q_embed = apply_rotary_optimized(q)
    k_embed = apply_rotary_optimized(k)
    
    # Only convert dtype if necessary (typically already correct)
    if q_embed.dtype != q.dtype:
        q_embed = q_embed.to(q.dtype)
    if k_embed.dtype != k.dtype:
        k_embed = k_embed.to(k.dtype)
    
    return q_embed, k_embed


#@torch.inference_mode()
#@torch.compile(dynamic=True, mode="default", fullgraph=False)
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
    # Class-level flag to control fused kernel usage (can be set via config)
    _global_use_fused = None  # None = auto-detect, True = force fused, False = force manual
    
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3InfRMSNorm using PyTorch's native fused RMSNorm kernel (PyTorch 2.4+).
        
        Falls back to manual implementation if fused kernel is not available.
        Can be disabled via config.use_fused_rmsnorm = False or --no-fused-rmsnorm flag.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        
        # Check if fused RMSNorm is available (PyTorch 2.4+)
        self._fused_available = hasattr(F, 'rms_norm')

    def forward(self, hidden_states):
        # Check config flag at runtime (allows toggling after model init)
        use_fused = self._fused_available
        if Qwen3InfRMSNorm._global_use_fused is not None:
            use_fused = use_fused and Qwen3InfRMSNorm._global_use_fused
        
        if use_fused:
            # Use PyTorch's native fused RMSNorm kernel
            # This is a single fused CUDA kernel - much faster!
            return F.rms_norm(hidden_states, [self.hidden_size], self.weight, self.variance_epsilon)
        else:
            # Fallback: manual implementation with explicit fp32 for numerical precision
            variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
            rsqrt_var = torch.rsqrt(variance + self.variance_epsilon).to(hidden_states.dtype)
            return self.weight * (hidden_states * rsqrt_var)

    @torch.inference_mode()
    def extra_repr(self):
        use_fused = self._fused_available and (Qwen3InfRMSNorm._global_use_fused is not False)
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, fused={use_fused}"


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

    #@torch.inference_mode()
    #@torch.compile(dynamic=True, mode="max-autotune-no-cudagraphs", fullgraph=False)
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3InfAttention(nn.Module):

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

        # NVIDIA Transformer Engine DotProductAttention for efficient arbitrary masks
        self._te_attn = None
        if TE_AVAILABLE:
            try:
                self._te_attn = TEDotProductAttention(
                    num_attention_heads=self.num_heads,
                    kv_channels=self.head_dim,
                    num_gqa_groups=self.num_key_value_heads,  # GQA support
                    attn_mask_type='arbitrary',  # Support arbitrary masks with broadcasting
                    qkv_format='bshd',  # [batch, seq, heads, dim]
                    attention_dropout=0.0,  # No dropout during inference
                    softmax_scale=self.scaling,
                )
            except Exception as e:
                logger.warning(f"Failed to create TE DotProductAttention: {e}")
                self._te_attn = None

        # Preallocate buffers 
        # Cached min for bf16 additive masks
        self._bf16_min = torch.finfo(torch.bfloat16).min
        
        # Pre-allocated inference buffers (lazily initialized on first forward)
        # These eliminate .contiguous() copies in the forward pass
        self._max_batch_size = getattr(config, "max_inference_batch_size", 8)
        self._max_seq_len = getattr(config, "max_inference_seq_len", 256)
        self._buffers_initialized = False
        
        # Register empty buffers - will be resized on first use
        self.register_buffer("_q_buffer", torch.empty(0), persistent=False)
        self.register_buffer("_k_buffer", torch.empty(0), persistent=False)
        self.register_buffer("_v_buffer", torch.empty(0), persistent=False)
        self.register_buffer("_attn_output_buffer", torch.empty(0), persistent=False)
        # RoPE output buffers
        self.register_buffer("_q_rope_buffer", torch.empty(0), persistent=False)
        self.register_buffer("_k_rope_buffer", torch.empty(0), persistent=False)

        self._refresh_qkv_weights(force=True)
    
    def _ensure_inference_buffers(self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Ensure inference buffers are allocated with sufficient size."""
        need_resize = (
            not self._buffers_initialized
            or self._q_buffer.numel() == 0
            or self._q_buffer.shape[0] < batch_size
            or self._q_buffer.shape[2] < seq_len
            or self._q_buffer.device != device
            or self._q_buffer.dtype != dtype
        )
        
        if need_resize:
            max_batch = max(batch_size, self._max_batch_size)
            max_seq = max(seq_len, self._max_seq_len)
            
            # Q buffer: [batch, num_heads, seq, head_dim]
            self._q_buffer = torch.empty(
                (max_batch, self.num_heads, max_seq, self.head_dim),
                dtype=dtype, device=device
            )
            # K buffer: [batch, num_kv_heads, seq, head_dim]
            self._k_buffer = torch.empty(
                (max_batch, self.num_key_value_heads, max_seq, self.head_dim),
                dtype=dtype, device=device
            )
            # V buffer: [batch, num_kv_heads, seq, head_dim]
            self._v_buffer = torch.empty(
                (max_batch, self.num_key_value_heads, max_seq, self.head_dim),
                dtype=dtype, device=device
            )
            # Attention output buffer: [batch, seq, hidden_size]
            self._attn_output_buffer = torch.empty(
                (max_batch, max_seq, self.num_heads * self.head_dim),
                dtype=dtype, device=device
            )
            # RoPE output buffers
            self._q_rope_buffer = torch.empty_like(self._q_buffer)
            self._k_rope_buffer = torch.empty_like(self._k_buffer)
            
            self._buffers_initialized = True
            self._max_batch_size = max_batch
            self._max_seq_len = max_seq

    @torch.inference_mode()
    def _refresh_qkv_weights(self, force: bool = False) -> None:
        if self.q_proj.weight.is_meta:
            if force or self._qkv_weight.numel() != 0:
                self._qkv_weight = torch.empty(0)
                self._qkv_bias = torch.empty(0)
            self._qkv_needs_refresh = True
            return
 
        need_refresh = force or (
            self._qkv_needs_refresh
            or self._qkv_weight.numel() == 0
            or self._q_weight_version != self.q_proj.weight._version
            or self._k_weight_version != self.k_proj.weight._version
            or self._v_weight_version != self.v_proj.weight._version
            or (self.q_proj.bias is not None and self._q_bias_version != self.q_proj.bias._version)
            or (self.k_proj.bias is not None and self._k_bias_version != self.k_proj.bias._version)
            or (self.v_proj.bias is not None and self._v_bias_version != self.v_proj.bias._version)
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
        self._q_weight_version = self.q_proj.weight._version
        self._k_weight_version = self.k_proj.weight._version
        self._v_weight_version = self.v_proj.weight._version
        if self.q_proj.bias is not None:
            self._q_bias_version = self.q_proj.bias._version
            self._k_bias_version = self.k_proj.bias._version
            self._v_bias_version = self.v_proj.bias._version
        else:
            self._q_bias_version = self._k_bias_version = self._v_bias_version = None

        if self.q_proj.bias is None:
            self._qkv_bias = torch.empty(0, device=weight_device, dtype=weight_dtype)
        else:
            self._qkv_bias = self._qkv_bias.to(self.q_proj.bias.dtype)

    #@torch.compile(dynamic=True, mode="default", fullgraph=False)
    def _attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

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

    @torch.compile(dynamic=False, mode="default", fullgraph=False)
    def _separate_qkv_projection(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        input_shape: Tuple[int, ...],
    ):
        """
        Optimized QKV projection that writes directly to pre-allocated buffers.
        Eliminates .contiguous() copies by using the correct memory layout from the start.
        
        Args:
            hidden_states: Input tensor [B, S, hidden_size]
            position_embeddings: Tuple of (cos, sin) tensors for RoPE
            input_shape: Tuple of (B, S) - batch size and sequence length
        """
        B, S = input_shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # Ensure buffers are allocated
        self._ensure_inference_buffers(B, S, device, dtype)
        
        # Get buffer slices
        q_out = self._q_buffer[:B, :, :S, :]  # [B, num_heads, S, head_dim]
        k_out = self._k_buffer[:B, :, :S, :]  # [B, num_kv_heads, S, head_dim]
        v_out = self._v_buffer[:B, :, :S, :]  # [B, num_kv_heads, S, head_dim]
        
        # Project Q and reshape+transpose directly into buffer
        # q_proj output: [B, S, num_heads * head_dim]
        # We need: [B, num_heads, S, head_dim]
        q_proj_out = self.q_proj(hidden_states)  # [B, S, num_heads * head_dim]
        q_reshaped = q_proj_out.view(B, S, self.num_heads, self.head_dim)
        # Apply norm then transpose into buffer
        q_normed = self.q_norm(q_reshaped)  # [B, S, num_heads, head_dim]
        # Copy with transpose: [B, S, H, D] -> [B, H, S, D]
        q_out.copy_(q_normed.permute(0, 2, 1, 3))
        
        # Project K
        k_proj_out = self.k_proj(hidden_states)
        k_reshaped = k_proj_out.view(B, S, self.num_key_value_heads, self.head_dim)
        k_normed = self.k_norm(k_reshaped)
        k_out.copy_(k_normed.permute(0, 2, 1, 3))
        
        # Project V (no norm)
        v_proj_out = self.v_proj(hidden_states)
        v_reshaped = v_proj_out.view(B, S, self.num_key_value_heads, self.head_dim)
        v_out.copy_(v_reshaped.permute(0, 2, 1, 3))
        
        # Apply RoPE using optimized PyTorch inplace implementation with pre-allocated buffers
        cos, sin = position_embeddings
        q_rope, k_rope = self._apply_rotary_pos_emb_inplace(q_out, k_out, cos, sin, B, S)
        
        return q_rope, k_rope, v_out
    
    def _apply_rotary_pos_emb_inplace(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ):
        """
        Apply RoPE using pre-allocated buffers to avoid memory allocation.
        """
        # Get buffer slices
        q_out = self._q_rope_buffer[:batch_size, :, :seq_len, :]
        k_out = self._k_rope_buffer[:batch_size, :, :seq_len, :]
        
        cos = cos.unsqueeze(1)  # [B, 1, S, D]
        sin = sin.unsqueeze(1)
        
        half = q.shape[-1] // 2
        cos = cos.view(*cos.shape[:-1], 2, half)
        sin = sin.view(*sin.shape[:-1], 2, half)
        
        cos0, cos1 = cos[..., 0, :], cos[..., 1, :]
        sin0, sin1 = sin[..., 0, :], sin[..., 1, :]
        
        # Q rotation
        q_r = q.view(*q.shape[:-1], 2, half)
        q_out_r = q_out.view(*q_out.shape[:-1], 2, half)
        
        q0, q1 = q_r[..., 0, :], q_r[..., 1, :]
        torch.mul(q0, cos0, out=q_out_r[..., 0, :])
        q_out_r[..., 0, :].sub_(q1 * sin0)
        torch.mul(q1, cos1, out=q_out_r[..., 1, :])
        q_out_r[..., 1, :].add_(q0 * sin1)
        
        # K rotation
        k_r = k.view(*k.shape[:-1], 2, half)
        k_out_r = k_out.view(*k_out.shape[:-1], 2, half)
        
        k0, k1 = k_r[..., 0, :], k_r[..., 1, :]
        torch.mul(k0, cos0, out=k_out_r[..., 0, :])
        k_out_r[..., 0, :].sub_(k1 * sin0)
        torch.mul(k1, cos1, out=k_out_r[..., 1, :])
        k_out_r[..., 1, :].add_(k0 * sin1)
        
        return q_out, k_out        
 
    @torch.inference_mode()
    #@torch.compile(dynamic=False, mode="reduce-overhead", fullgraph=False)
    def forward_separate_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        replace_position: Optional[torch.Tensor] = None,
        is_training: bool = False,
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
    def forward_te_attention(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        shared_mask: torch.Tensor,  # Shared mask [1, 1, Q, K] for all batches (additive: -inf=masked, 0=attend)
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        replace_position: Optional[torch.Tensor] = None,
        is_training: bool = False,
        use_cache: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: Optional[int] = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass using NVIDIA Transformer Engine DotProductAttention with shared mask.
        
        This uses a single mask [1, 1, Q, K] that broadcasts across all batches,
        enabling efficient fused attention via TE's cuDNN/FlashAttention backend.
        
        Key advantages over SDPA with per-batch masks:
        1. Single mask tensor instead of [B, 1, Q, K] - broadcasts efficiently
        2. Native GQA support - no need to expand K/V heads
        3. Optimized cuDNN/FlashAttention kernels with arbitrary mask support
        """
        B, S, _ = hidden_states.shape
        input_shape = (B, S)
        
        query_states, key_states, value_states = self._separate_qkv_projection(hidden_states, position_embeddings, input_shape)

        # Update KV cache
        past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            cache_kwargs={"cache_position": cache_position},
        )
        
        # Retrieve full cached K/V
        past_k, past_v = past_key_value[self.layer_idx]
        key_states = past_k[:, :, :sequence_length, :]
        value_states = past_v[:, :, :sequence_length, :]
        
        # Convert to TE format: [B, H, S, D] -> [B, S, H, D]
        query_states_te = query_states.transpose(1, 2).contiguous()
        key_states_te = key_states.transpose(1, 2).contiguous()
        value_states_te = value_states.transpose(1, 2).contiguous()
        
        # Convert additive mask to boolean mask for TE
        # Input: additive mask where -inf = masked, 0 = attend
        # TE convention: True = MASKED, False = ATTEND
        if shared_mask is not None:
            te_mask = shared_mask < -1e9  # True where masked (-inf), False where attend (0)
        else:
            te_mask = None
        
        # Call TE DotProductAttention
        # Input: Q [B, S_q, num_heads, D], K/V [B, S_k, num_kv_heads, D]
        # Mask: [1, 1, S_q, S_k] broadcasts to all batches/heads
        attn_output = self._te_attn(
            query_states_te,
            key_states_te,
            value_states_te,
            attention_mask=te_mask,
        )
        # Output: [B, S_q, num_heads, head_dim] with qkv_format="bshd"
        
        # Reshape to [B, S, hidden_size] for output projection
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None  # No attention weights

    @torch.inference_mode()
    def forward_flash_attention(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        replace_position: Optional[torch.Tensor] = None,
        is_training: bool = False,
        use_cache: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: Optional[int] = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass using FlashAttention-2 directly.
        
        This path uses FlashAttention with bidirectional attention (no custom masks).
        FlashAttention handles GQA natively without K/V head expansion.
        """
        B, S, _ = hidden_states.shape
        input_shape = (B, S)
        query_states, key_states, value_states = self._separate_qkv_projection(hidden_states, position_embeddings, input_shape)

        # Update KV cache
        past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            cache_kwargs={"cache_position": cache_position},
        )
        
        # Retrieve full cached K/V
        past_k, past_v = past_key_value[self.layer_idx]
        key_states = past_k[:, :, :sequence_length, :]
        value_states = past_v[:, :, :sequence_length, :]
        
        # FlashAttention expects [B, S, H, D] format
        q = query_states.transpose(1, 2)   # [B, S_q, num_heads, head_dim]
        k = key_states.transpose(1, 2)     # [B, S_kv, num_kv_heads, head_dim]
        v = value_states.transpose(1, 2)   # [B, S_kv, num_kv_heads, head_dim]
        
        # FlashAttention handles GQA natively - no K/V expansion needed!
        # Bidirectional attention (causal=False) for diffusion models
        attn_output = flash_attn_func(
            q, k, v,
            softmax_scale=self.scaling,
            causal=False,  # Bidirectional attention - no custom masks
        )
        # attn_output: [B, S_q, num_heads, head_dim]
        
        # Reshape to [B, S, hidden_size]
        attn_output = attn_output.reshape(B, S, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None  # No attention weights
    
    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        replace_position: Optional[torch.Tensor] = None,
        is_training: bool = False,
        use_cache: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: Optional[int] = 0,                          # 0 = baseline, 1 = fused commit-prefix
        use_optimized: bool = True,  # Use optimized forward with pre-allocated buffers
        shared_attn_mask: Optional[torch.Tensor] = None,  # Shared mask for TE attention
        attn_kernel: str = "sdpa",  # Attention kernel: "sdpa" (default), "te", "flash"
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with selectable attention kernel.
        
        Args:
            attn_kernel: Which attention kernel to use:
                - "sdpa" (default): PyTorch SDPA with custom mask support
                - "te": NVIDIA Transformer Engine DotProductAttention (requires shared mask)
                - "flash": FlashAttention-2 (bidirectional only, no custom masks)
        """
        # Dispatch based on attn_kernel selection
        if attn_kernel == "te":
            # NVIDIA Transformer Engine with shared mask
            if self._te_attn is None:
                raise ValueError("TE attention requested but NVIDIA Transformer Engine is not available")
            return self.forward_te_attention(
                hidden_states,
                position_embeddings,
                shared_attn_mask,
                past_key_value,
                cache_position,
                replace_position,
                is_training,
                use_cache,
                position_ids,
                sequence_length,
                **kwargs,
            )
        elif attn_kernel == "flash":
            # FlashAttention-2 - bidirectional, no custom masks supported
            if not FLASH_ATTN_AVAILABLE or flash_attn_func is None:
                raise ValueError("FlashAttention requested but flash_attn is not available. "
                               "Install with: pip install flash-attn --no-build-isolation")
            return self.forward_flash_attention(
                hidden_states,
                position_embeddings,
                past_key_value,
                cache_position,
                replace_position,
                is_training,
                use_cache,
                position_ids,
                sequence_length,
                **kwargs,
            )
        else:  # "sdpa" (default)
            # SDPA with custom mask support
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
    #@torch.compile(dynamic=True, mode="default", fullgraph=False)
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
        is_training: bool = False,
        committed_prefix_len: Optional[int] = 0,
        shared_attn_mask: Optional[torch.Tensor] = None,  # Shared mask for TE attention
        attn_kernel: str = "sdpa",  # Attention kernel: "sdpa" (default), "te", "flash"
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
            shared_attn_mask=shared_attn_mask,
            attn_kernel=attn_kernel,
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
            # Optimized: use repeat instead of cat to avoid extra allocation
            # emb = torch.cat((freqs, freqs), dim=-1) creates a copy
            # repeat(1, 1, 2) is more memory efficient
            emb = freqs.repeat(1, 1, 2)
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

    def _create_shared_attention_mask(self, past_cache_length, committed_prefix_len, block_length, device, dtype, prefix_bidir=False, distance_bidir=None):
        """
        Create a shared attention mask tensor for efficient Triton FlashAttention.
        
        This creates a single mask [1, 1, Q, K] that broadcasts across all batches,
        enabling efficient fused attention via custom Triton kernel.
        
        IMPORTANT: This requires uniform parameters across batches (use --unified-prefix).
        
        Args:
            past_cache_length: int or Tensor - past cache length (must be uniform)
            committed_prefix_len: int or Tensor - committed prefix length (must be uniform)
            block_length: int - current block length (query length)
            device: torch device
            dtype: torch dtype (should match model dtype, e.g., bfloat16)
            prefix_bidir: bool - if True, committed prefix can attend to uncommitted tokens
            distance_bidir: int or None - if set, limits attention to distance_bidir FUTURE tokens
            
        Returns:
            Shared mask tensor [1, 1, Q, K], or None if no masking needed
        """
        # Extract scalar values (must be uniform for shared mask)
        if isinstance(past_cache_length, torch.Tensor):
            past_len = int(past_cache_length[0].item())
        else:
            past_len = past_cache_length
            
        if isinstance(committed_prefix_len, torch.Tensor):
            commit_len = int(committed_prefix_len[0].item())
        else:
            commit_len = committed_prefix_len
        
        # Check if we need any masking
        need_commit_mask = commit_len > 0 and not prefix_bidir
        need_distance_mask = distance_bidir is not None
        
        if not need_commit_mask and not need_distance_mask:
            return None
        
        # Create additive mask manually (works with both SDPA and TE)
        Q = block_length
        K = past_len + block_length
        min_val = torch.finfo(dtype).min
        
        mask = torch.zeros((1, 1, Q, K), device=device, dtype=dtype)
        
        # COMMIT PREFIX MASKING
        if need_commit_mask:
            tail_start = past_len + commit_len
            q_idx = torch.arange(Q, device=device).view(1, 1, Q, 1)
            k_idx = torch.arange(K, device=device).view(1, 1, 1, K)
            
            is_prefix_query = q_idx < commit_len
            is_uncommitted_key = k_idx >= tail_start
            commit_mask = is_prefix_query & is_uncommitted_key
            mask = torch.where(commit_mask, min_val, mask)
        
        # DISTANCE-BASED MASKING
        if need_distance_mask:
            block_start = past_len
            q_idx = torch.arange(Q, device=device).view(1, 1, Q, 1)
            k_idx = torch.arange(K, device=device).view(1, 1, 1, K)
            
            is_in_block = (k_idx >= block_start) & (k_idx < block_start + Q)
            k_rel = k_idx - block_start
            q_rel = q_idx
            is_future_beyond = (k_rel > q_rel) & (k_rel > q_rel + distance_bidir)
            distance_mask = is_in_block & is_future_beyond
            mask = torch.where(distance_mask, min_val, mask)
        
        return mask

    @torch.inference_mode()
    def _generate_batch_mask(self, past_cache_length, committed_prefix_len, block_length, batch_count, device, dtype, past_key_values, padding_counts=None, prefix_bidir=False, distance_bidir=None):
        """
        Generate attention mask for per-batch isolation and commit prefix handling.
        
        VECTORIZED IMPLEMENTATION - No Python loops, no .item() calls.
        
        SHARED MASK OPTIMIZATION: When all batches have uniform parameters (same
        past_cache_length and committed_prefix_len), returns a [1, 1, Q, K] mask
        that broadcasts to all batches. This enables more efficient attention kernels.
        Use --unified-prefix flag to ensure uniform parameters across batches.
        
        Note: Static block generation does NOT use padding (all prompts same length,
        fixed block size). The padding_counts parameter is kept for API compatibility
        but is expected to be None in static block mode.
        
        This mask ensures:
        1. Each batch only attends to its OWN valid cache positions (per-batch isolation)
        2. Committed prefix tokens don't see uncommitted tokens (commit masking) - unless prefix_bidir=True
        3. Optional distance-based limiting of FUTURE token attention within the block (distance_bidir)
        
        Supports both scalar and per-batch tensor values for past_cache_length and committed_prefix_len.
        
        Args:
            past_cache_length: int or Tensor[B] - past cache length per batch
            committed_prefix_len: int or Tensor[B] - committed prefix length per batch
            block_length: int - current block length (query length)
            batch_count: int - number of batches
            device: torch device
            dtype: torch dtype
            past_key_values: cache object (can be None or empty for prefill)
            padding_counts: Tensor[B] or None - unused in static block mode (kept for API compatibility)
            prefix_bidir: bool - if True, committed prefix can attend to uncommitted tokens (default False)
            distance_bidir: int or None - if set, limits attention to only distance_bidir FUTURE tokens
            
        Returns:
            Tensor[1, 1, Q, K] (shared/unified) or Tensor[B, 1, Q, K] (per-batch) attention mask, or None
        """
        B = batch_count
        
        # Static block mode: no padding, prefill has no mask needed
        # Determine if we're in prefill mode (past_cache_length == 0)
        if isinstance(past_cache_length, torch.Tensor):
            is_prefill = (past_cache_length.max().item() == 0)
        else:
            is_prefill = (past_cache_length == 0)
        
        # During prefill in static block mode: no mask needed (no padding, bidirectional attention)
        if is_prefill:
            return None
        
        # Convert to tensors if scalars for uniform handling
        if isinstance(past_cache_length, torch.Tensor):
            past_k_len = past_cache_length  # [B]
            max_past_k_len = int(past_k_len.max().item())
            min_past_k_len = int(past_k_len.min().item())
        else:
            past_k_len = past_cache_length  # scalar
            max_past_k_len = past_cache_length
            min_past_k_len = past_cache_length
            
        if isinstance(committed_prefix_len, torch.Tensor):
            commit_len = committed_prefix_len  # [B]
            max_committed = int(commit_len.max().item())
            min_committed = int(commit_len.min().item())
            any_committed = max_committed > 0
        else:
            commit_len = committed_prefix_len  # scalar
            max_committed = committed_prefix_len
            min_committed = committed_prefix_len
            any_committed = committed_prefix_len > 0

        total_k_len = block_length + max_past_k_len
        
        # Check if we need a mask:
        # 1. When batches have different past_cache_lengths (need per-batch isolation)
        # 2. When there's a committed prefix (need commit masking) - unless prefix_bidir
        # 3. When distance_bidir is set (need distance-based masking within block)
        batches_differ = (min_past_k_len != max_past_k_len) if isinstance(past_k_len, torch.Tensor) else False
        commits_differ = (min_committed != max_committed) if isinstance(commit_len, torch.Tensor) else False
        need_commit_mask = any_committed and not prefix_bidir
        need_mask = need_commit_mask or batches_differ or (distance_bidir is not None)
        
        if not need_mask:
            return None
        
        # OPTIMIZATION: Use shared mask [1, 1, Q, K] when all batches are uniform
        # This enables more efficient attention kernels
        use_shared_mask = not batches_differ and not commits_differ
            
        if max_committed <= block_length or not any_committed:
            Q = block_length
            min_val = torch.finfo(dtype).min

            buffer = self._commit_mask_buffer
            
            # Use batch dimension 1 for shared mask (broadcasts to all batches)
            mask_B = 1 if use_shared_mask else B

            if B > buffer.shape[0] or Q > buffer.shape[2] or total_k_len > buffer.shape[3]:
                raise ValueError(
                    "Commit mask dimensions exceed preallocated buffer. "
                    f"Requested (batch={B}, block={Q}, total={total_k_len}) "
                    f"but buffer max is {buffer.shape}."
                )

            commit_mask = buffer[:mask_B, :1, :Q, :total_k_len]
            commit_mask.zero_()

            # ============================================================
            # VECTORIZED mask generation (no Python loops, no .item() calls)
            # When use_shared_mask=True, mask_B=1 and mask broadcasts to all batches
            # ============================================================
            
            # Create position index tensors
            q_idx = torch.arange(Q, device=device, dtype=torch.long).view(1, 1, Q, 1)
            k_idx = torch.arange(total_k_len, device=device, dtype=torch.long).view(1, 1, 1, total_k_len)
            
            # For shared mask, use scalar values directly; for per-batch, use tensors
            if use_shared_mask:
                # Scalars -> [1, 1, 1, 1] for uniform broadcasting
                past_k_len_4d = torch.tensor([[[[max_past_k_len]]]], device=device, dtype=torch.long)
                commit_len_4d = torch.tensor([[[[max_committed]]]], device=device, dtype=torch.long)
                total_k_len_4d = torch.tensor([[[[total_k_len]]]], device=device, dtype=torch.long)
            else:
                # Convert scalars to tensors for uniform vectorized handling
                if not isinstance(past_k_len, torch.Tensor):
                    past_k_len_t = torch.full((B,), past_k_len, device=device, dtype=torch.long)
                else:
                    past_k_len_t = past_k_len
                    
                if not isinstance(commit_len, torch.Tensor):
                    commit_len_t = torch.full((B,), commit_len, device=device, dtype=torch.long)
                else:
                    commit_len_t = commit_len
                
                # Reshape for broadcasting: [B] -> [B, 1, 1, 1]
                past_k_len_4d = past_k_len_t.view(B, 1, 1, 1)
                commit_len_4d = commit_len_t.view(B, 1, 1, 1)
                total_k_len_4d = block_length + past_k_len_4d  # per-batch total length
            
            # 1. PER-BATCH RANGE MASKING (only needed if batches differ)
            # Mask out positions beyond this batch's valid range
            # This prevents batch b from attending to KV positions it hasn't written to
            if batches_differ:
                invalid_range_mask = k_idx >= total_k_len_4d  # [B, 1, 1, K]
                commit_mask.masked_fill_(invalid_range_mask, min_val)
            
            # 2. COMMIT PREFIX MASKING (vectorized)
            # Prevent committed prefix tokens from seeing uncommitted tokens
            if any_committed and not prefix_bidir:
                # Query positions that are in the prefix: q_idx < commit_len
                is_prefix_query = q_idx < commit_len_4d  # [mask_B, 1, Q, 1]
                
                # Key positions that are uncommitted: k_idx >= (past_k_len + commit_len)
                tail_start_4d = past_k_len_4d + commit_len_4d  # [mask_B, 1, 1, 1]
                is_uncommitted_kv = k_idx >= tail_start_4d  # [mask_B, 1, 1, K]
                
                # Also ensure we're within valid range (k_idx < total_k_len)
                is_valid_kv = k_idx < total_k_len_4d  # [mask_B, 1, 1, K]
                
                # Combined commit mask: prefix queries can't see uncommitted (but valid) keys
                commit_prefix_mask = is_prefix_query & is_uncommitted_kv & is_valid_kv  # [mask_B, 1, Q, K]
                commit_mask.masked_fill_(commit_prefix_mask, min_val)
            
            # 3. DISTANCE-BASED MASKING (vectorized)
            # Limits each token to attend only to distance_bidir FUTURE tokens within the block
            if distance_bidir is not None:
                # Block start/end in KV space
                block_start_4d = past_k_len_4d  # [mask_B, 1, 1, 1]
                block_end_4d = past_k_len_4d + Q  # [mask_B, 1, 1, 1]
                
                # Check if k is in the current block
                is_in_block = (k_idx >= block_start_4d) & (k_idx < block_end_4d)  # [mask_B, 1, 1, K]
                
                # Relative position of k within the block
                k_rel = k_idx - block_start_4d  # [mask_B, 1, 1, K]
                q_rel = q_idx  # [1, 1, Q, 1] - query position is already relative to block
                
                # Mask if: k_rel > q_rel + distance_bidir AND k_rel > q_rel (future beyond distance)
                is_future_beyond_distance = (k_rel > q_rel) & (k_rel > q_rel + distance_bidir)  # [mask_B, 1, Q, K]
                
                # Only apply to positions within the current block
                distance_mask = is_in_block & is_future_beyond_distance  # [mask_B, 1, Q, K]
                commit_mask.masked_fill_(distance_mask, min_val)

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
        is_training: bool = False,
        committed_prefix_len: Optional[torch.Tensor | int] = 0,  # Can be int or Tensor[B]
        kv_cache_pos_offset: Optional[int] = None,
        past_cache_length: Optional[torch.Tensor | int] = 0,  # Can be int or Tensor[B]
        block_length: Optional[int] = 0,
        padding_counts: Optional[torch.Tensor] = None,  # UNUSED in static block mode (kept for API compat)
        prefix_bidir: bool = False,  # If True, committed prefix can attend to uncommitted tokens
        distance_bidir: Optional[int] = None,  # If set, limits attention to only distance_bidir FUTURE tokens
        attn_kernel: str = "sdpa",  # Attention kernel: "sdpa" (default), "te", "flash"
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
            # Handle both 1D [seq_len] and 2D [B, seq_len] cache_position
            if cache_position.dim() == 1:
                position_ids = cache_position.unsqueeze(0)  # [seq_len] -> [1, seq_len]
            else:
                position_ids = cache_position  # [B, seq_len] -> use directly
            
        if block_length != inputs_embeds.shape[1]:
            print(f"WARNING: block_length != inputs_embeds.shape[1]: {block_length} != {inputs_embeds.shape[1]}")

        # Handle both scalar and tensor past_cache_length
        if isinstance(past_cache_length, torch.Tensor):
            max_past_cache_length = int(past_cache_length.max().item())
        else:
            max_past_cache_length = past_cache_length
        sequence_length = max_past_cache_length + block_length
        
        max_index = cache_position.max().item()
        max_index = max_index + 1
        
        if max_index != sequence_length:
            pass  # Expected with per-batch positions - different batches can be at different positions
            
        causal_mask = attention_mask
        shared_attn_mask = None  # Shared mask for TE DotProductAttention
        
        # Generate masks based on attention kernel selection
        # Static block mode: no padding support (all prompts same length, fixed block size)
        if causal_mask is None and (past_key_values is not None or distance_bidir is not None):
            if attn_kernel == "flash":
                # FlashAttention: bidirectional, no custom masks supported
                # Skip all mask generation - attention will be fully bidirectional
                causal_mask = None
                shared_attn_mask = None
            elif attn_kernel == "te":
                # NVIDIA Transformer Engine: use shared mask [1, 1, Q, K]
                if not TE_AVAILABLE:
                    raise ValueError("TE attention kernel requested but NVIDIA Transformer Engine is not available")
                shared_attn_mask = self._create_shared_attention_mask(
                    past_cache_length=past_cache_length,
                    committed_prefix_len=committed_prefix_len,
                    block_length=inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                    prefix_bidir=prefix_bidir,
                    distance_bidir=distance_bidir,
                )
                causal_mask = None
            else:  # "sdpa" (default)
                # SDPA: generate per-batch mask for commit prefix and distance masking
                causal_mask = self._generate_batch_mask(
                    past_cache_length=past_cache_length,
                    committed_prefix_len=committed_prefix_len,
                    block_length=inputs_embeds.shape[1],
                    batch_count=inputs_embeds.shape[0],
                    device=self._commit_mask_buffer.device,
                    dtype=self._commit_mask_buffer.dtype,
                    past_key_values=past_key_values,
                    prefix_bidir=prefix_bidir,
                    distance_bidir=distance_bidir,
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
                shared_attn_mask=shared_attn_mask,
                attn_kernel=attn_kernel,
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

