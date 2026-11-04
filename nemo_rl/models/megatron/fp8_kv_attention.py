# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Opt-in attention with E4M3 K/V scales applied to scores and probabilities.

This experimental PyTorch backend keeps unscaled E4M3 values in the input
dtype for matrix multiplication. It never multiplies K/V by their scales
before attention. Query chunks and activation checkpointing bound score
storage; this is not a replacement for TE's fused-kernel performance.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from nemo_rl.models.megatron.fp8_kv_quantization import quantize_kv_per_token_head

QUERY_CHUNK_SIZE = 64


class _UnscaledE4M3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        quantized, scales = quantize_kv_per_token_head(tensor)
        ctx.save_for_backward(scales)
        ctx.mark_non_differentiable(scales)
        # Every E4M3 value is exactly representable in BF16/FP16. This cast
        # changes storage only: dequantization scales are NOT applied here.
        return quantized.to(tensor.dtype), scales

    @staticmethod
    def backward(ctx, grad_values, grad_scales):
        (scales,) = ctx.saved_tensors
        # STE for quantize(x / scale); do not differentiate amax or scales.
        return (grad_values.float() / scales).to(grad_values.dtype)


class _BmmFP32(torch.autograd.Function):
    """FP32 dot-product outputs, including on torch versions lacking its VJP."""

    @staticmethod
    def forward(ctx, left, right):
        ctx.save_for_backward(left, right)
        if left.is_cuda and left.dtype in (torch.bfloat16, torch.float16):
            return torch.bmm(left, right, out_dtype=torch.float32)
        return torch.bmm(left.float(), right.float())

    @staticmethod
    def backward(ctx, grad):
        left, right = ctx.saved_tensors
        grad_left = torch.bmm(grad.float(), right.float().transpose(1, 2))
        grad_right = torch.bmm(left.float().transpose(1, 2), grad.float())
        return grad_left.to(left.dtype), grad_right.to(right.dtype)


def _attention_chunk(query, key, value, k_scale, v_scale, positions, causal, scale):
    q_len, q_heads, dim = query.shape
    kv_heads = key.shape[1]
    groups = q_heads // kv_heads
    # Flatten grouped Q heads into the row dimension without copying K/V
    # once per Q head. Matrix products accumulate into FP32.
    q = query.reshape(q_len, kv_heads, groups, dim).permute(1, 2, 0, 3)
    q = q.reshape(kv_heads, groups * q_len, dim)
    k = key.permute(1, 2, 0)
    v = value.permute(1, 0, 2)
    scores = _BmmFP32.apply(q, k) * (scale * k_scale.permute(1, 2, 0))
    if causal:
        mask = torch.arange(len(key), device=query.device)[None, :] > positions[:, None]
        scores = scores.reshape(kv_heads, groups, q_len, len(key))
        scores = scores.masked_fill(mask[None, None], -torch.inf)
        scores = scores.reshape(kv_heads, groups * q_len, len(key))
    probabilities = scores.softmax(dim=-1)
    # Match vLLM's placement: multiply P by v_scale, then round the weights
    # to the matmul operand dtype; V remains unscaled E4M3 values.
    weights = (probabilities * v_scale.permute(1, 2, 0)).to(value.dtype)
    output = _BmmFP32.apply(weights, v)
    return (
        output.reshape(kv_heads, groups, q_len, dim)
        .permute(2, 0, 1, 3)
        .reshape(query.shape)
        .to(query.dtype)
    )


def scaled_fp8_kv_attention(
    query, key, value, *, positions=None, causal=True, scale=None
):
    """THD attention with detached scales and STE gradients for original K/V.

    Softmax is normalized per query over all valid keys. vLLM uses an online
    tiled softmax, so scale placement matches but bitwise parity is not promised.
    """
    if query.ndim != 3 or key.ndim != 3 or value.shape != key.shape:
        raise ValueError("in-attention FP8 KV requires THD Q/K/V")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("Q/K/V must have the same dtype")
    if query.shape[-1] != key.shape[-1] or query.shape[1] % key.shape[1]:
        raise ValueError("incompatible head dimensions or GQA head counts")
    if not len(query) or not len(key):
        raise ValueError("in-attention FP8 KV requires nonempty sequences")
    if positions is None:
        if len(query) > len(key):
            raise ValueError("query length must not exceed KV length")
        positions = (
            torch.arange(len(query), device=query.device) + len(key) - len(query)
        )
    if positions.shape != (len(query),):
        raise ValueError("query positions must have one entry per query token")
    if scale is None:
        scale = query.shape[-1] ** -0.5
    k, ks = _UnscaledE4M3.apply(key)
    v, vs = _UnscaledE4M3.apply(value)
    outputs = []
    for start in range(0, len(query), QUERY_CHUNK_SIZE):
        args = (
            query[start : start + QUERY_CHUNK_SIZE],
            k,
            v,
            ks,
            vs,
            positions[start : start + QUERY_CHUNK_SIZE],
            causal,
            scale,
        )
        if torch.is_grad_enabled() and any(
            t.requires_grad for t in (query, key, value)
        ):
            outputs.append(checkpoint(_attention_chunk, *args, use_reentrant=False))
        else:
            outputs.append(_attention_chunk(*args))
    return torch.cat(outputs).contiguous()


def cp_token_positions(lengths, rank, size, device):
    """Global positions for Megatron's per-sequence mirrored CP chunks."""
    result = []
    offset = 0
    for length in lengths:
        if size == 1:
            result.append(torch.arange(offset, offset + length, device=device))
        else:
            if length % (2 * size):
                raise ValueError(
                    "packed sequence lengths must be divisible by 2 * CP size"
                )
            chunk = length // (2 * size)
            for part in (rank, 2 * size - rank - 1):
                result.append(
                    torch.arange(
                        offset + part * chunk,
                        offset + (part + 1) * chunk,
                        device=device,
                    )
                )
        offset += length
    return torch.cat(result)


class FP8KVScaledAttention(nn.Module):
    """Megatron core-attention replacement; no weights or checkpoint state keys."""

    def __init__(self, config: Any, cp_group=None, scale=None):
        super().__init__()
        self.config = config
        self.cp_group = cp_group
        self.scale = scale
        if config.cuda_graph_impl != "none" or config.enable_cuda_graph:
            raise ValueError("in-attention FP8 KV does not support CUDA graph capture")
        if config.apply_query_key_layer_scaling:
            raise ValueError(
                "in-attention FP8 KV does not support query/key layer scaling"
            )
        if config.attention_dropout != 0:
            raise ValueError(
                "in-attention FP8 KV currently requires attention_dropout=0"
            )
        if config.window_size is not None:
            raise ValueError(
                "in-attention FP8 KV does not support sliding-window attention"
            )
        if (
            config.softmax_type != "vanilla"
            or config.attn_logit_softcapping is not None
        ):
            raise ValueError(
                "in-attention FP8 KV requires vanilla softmax without softcap"
            )
        if config.qk_clip or config.log_max_attention_logit:
            raise ValueError(
                "in-attention FP8 KV does not support QK clipping/logit statistics"
            )

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        num_splits=None,
        bf16_backward=None,
    ):
        if attention_mask is not None or attention_bias is not None:
            raise ValueError(
                "in-attention FP8 KV requires implicit causal/no-mask attention without bias"
            )
        if num_splits is not None or bf16_backward is not None:
            raise ValueError(
                "TE-specific attention overrides are unsupported by in-attention FP8 KV"
            )
        mask_name = attn_mask_type.name if attn_mask_type is not None else "causal"
        if mask_name not in (
            "causal",
            "causal_bottom_right",
            "padding_causal",
            "no_mask",
            "padding",
        ):
            raise ValueError(f"unsupported in-attention FP8 KV mask: {mask_name}")
        causal = mask_name in ("causal", "causal_bottom_right", "padding_causal")
        group = self.cp_group
        if packed_seq_params is not None:
            from megatron.core.packed_seq_params import resolve_cp_group

            group = resolve_cp_group(group, packed_seq_params)
        size = torch.distributed.get_world_size(group) if group is not None else 1
        rank = torch.distributed.get_rank(group) if group is not None else 0
        expected_cp = self.config.context_parallel_size
        if (
            packed_seq_params is not None
            and packed_seq_params.local_cp_size is not None
        ):
            expected_cp = packed_seq_params.local_cp_size
        if size != expected_cp:
            raise ValueError(
                "context-parallel group does not match attention configuration"
            )
        original_shape = query.shape
        if packed_seq_params is None:
            if (
                query.ndim != 4
                or key.ndim != 4
                or value.shape != key.shape
                or query.shape[1] != key.shape[1]
            ):
                raise ValueError("unpacked in-attention FP8 KV requires SBHD tensors")
            q_len, batch = query.shape[:2]
            k_len = key.shape[0]
            query, key, value = (
                t.permute(1, 0, 2, 3).flatten(0, 1).contiguous()
                for t in (query, key, value)
            )
            q_lengths, k_lengths = [q_len * size] * batch, [k_len * size] * batch
            q_valid, k_valid = q_lengths, k_lengths
        else:
            if packed_seq_params.qkv_format != "thd" or query.ndim != 3:
                raise ValueError("packed in-attention FP8 KV requires THD tensors")

            def lengths(cu):
                boundaries = cu.detach().cpu().tolist()
                if boundaries[0] != 0 or any(
                    b <= a for a, b in zip(boundaries, boundaries[1:])
                ):
                    raise ValueError(
                        "packed attention boundaries must increase from zero"
                    )
                return [b - a for a, b in zip(boundaries, boundaries[1:])]

            q_valid = lengths(packed_seq_params.cu_seqlens_q)
            k_valid = lengths(packed_seq_params.cu_seqlens_kv)
            q_padded = packed_seq_params.cu_seqlens_q_padded
            k_padded = packed_seq_params.cu_seqlens_kv_padded
            q_lengths = lengths(q_padded) if q_padded is not None else q_valid
            k_lengths = lengths(k_padded) if k_padded is not None else k_valid
        if (
            len(q_lengths) != len(k_lengths)
            or len(q_lengths) != len(q_valid)
            or len(k_lengths) != len(k_valid)
        ):
            raise ValueError("Q/KV packed sequence counts must match")
        if sum(q_lengths) != len(query) * size or sum(k_lengths) != len(key) * size:
            raise ValueError("packed sequence lengths do not match CP-local tensors")
        if any(
            q > qp or k > kp or q > k
            for q, qp, k, kp in zip(q_valid, q_lengths, k_valid, k_lengths)
        ):
            raise ValueError("invalid packed sequence padding or query/KV lengths")
        positions = cp_token_positions(q_lengths, rank, size, query.device)
        if size > 1:
            if q_lengths != k_lengths or q_valid != k_valid:
                raise ValueError(
                    "context-parallel in-attention FP8 KV requires self-attention with matching Q/KV lengths"
                )
            from torch.distributed.nn.functional import all_gather

            order = torch.cat(
                [
                    cp_token_positions(k_lengths, r, size, key.device)
                    for r in range(size)
                ]
            ).argsort()
            # One collective gives a consistent backward order even on ranks
            # whose queries are entirely padding.
            gathered = torch.cat(
                all_gather(torch.stack((key, value)), group=group), dim=1
            )
            key, value = gathered.index_select(1, order).unbind(0)
        outputs = []
        local_start = q_start = k_start = 0
        for q_physical, k_physical, q_count, k_count in zip(
            q_lengths, k_lengths, q_valid, k_valid
        ):
            local_length = q_physical // size
            local_q = query[local_start : local_start + local_length]
            relative_positions = (
                positions[local_start : local_start + local_length] - q_start
            )
            valid_indices = (relative_positions < q_count).nonzero().flatten()
            out = local_q.new_zeros(local_q.shape)
            if len(valid_indices):
                out = out.index_copy(
                    0,
                    valid_indices,
                    scaled_fp8_kv_attention(
                        local_q.index_select(0, valid_indices),
                        key[k_start : k_start + k_count],
                        value[k_start : k_start + k_count],
                        positions=relative_positions.index_select(0, valid_indices)
                        + k_count
                        - q_count,
                        causal=causal,
                        scale=self.scale,
                    ),
                )
            outputs.append(out)
            local_start += local_length
            q_start += q_physical
            k_start += k_physical
        result = torch.cat(outputs)
        if size > 1:
            # Padding-only query ranks must still reduce zero gradients to K/V.
            result = result + (query.sum() + key.sum() + value.sum()) * 0
        if packed_seq_params is None:
            result = result.reshape(batch, q_len, *original_shape[-2:]).permute(
                1, 0, 2, 3
            )
            return result.flatten(-2).contiguous()
        return result.flatten(-2).contiguous()

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        # As in TEDotProductAttention with vanilla softmax, there are no
        # learned tensors to put into Megatron's distributed checkpoint.
        return {}


class FP8KVScaledAttentionWithExtraState(FP8KVScaledAttention):
    """Retain TE's ordinary PyTorch extra state without calling TE attention."""

    def __init__(
        self,
        config,
        cp_group,
        scale,
        extra_state_getter: Callable[[], Any],
        extra_state_setter: Callable[[Any], None],
    ):
        super().__init__(config, cp_group, scale)
        self.extra_state_getter = extra_state_getter
        self.extra_state_setter = extra_state_setter

    def get_extra_state(self):
        return self.extra_state_getter()

    def set_extra_state(self, state):
        self.extra_state_setter(state)
