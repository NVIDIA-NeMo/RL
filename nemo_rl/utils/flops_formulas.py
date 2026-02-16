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

from dataclasses import dataclass
from typing import List, Optional, Union


# lifted from NeMo/nemo/utils/flops_formulas.py
@dataclass
class FLOPSConfig:
    """Contains the model hparams needed for FLOPS computations."""

    gbs: int
    enc_seq_len: Optional[int] = None
    hs: Optional[int] = None
    layers: Optional[int] = None
    ffn_hs: Optional[int] = None
    attention_heads: Optional[int] = None
    moe_router_topk: Optional[int] = None
    query_groups: Optional[int] = None
    img_seq_len: Optional[int] = None
    img_h: Optional[int] = None
    img_w: Optional[int] = None
    in_channels: Optional[int] = None
    patch_dim: Optional[int] = None
    class_token_len: Optional[int] = None
    projector_type: Optional[str] = None
    inp_s: Optional[int] = None
    model_pattern: Optional[str] = None
    vocab_size: Optional[int] = None
    model_channels: Optional[int] = None
    vec_in_dim: Optional[int] = None
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_head_dim: Optional[int] = None
    qk_pos_emb_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    moe_layer_freq: Optional[Union[int, List[int]]] = None
    moe_shared_expert_intermediate_size: Optional[int] = None
    moe_ffn_hidden_size: Optional[int] = None
    mtp_num_layers: Optional[int] = None
    causal_self_attn: Optional[bool] = None
    is_hybrid_model: bool = False
    hybrid_override_pattern: Optional[str] = None
    mamba_state_dim: Optional[int] = None
    mamba_head_dim: Optional[int] = None
    mamba_num_groups: Optional[int] = None
    mamba_num_heads: Optional[int] = None

    # GPT-OSS / Sliding Window Attention specific fields
    swa_window_size: Optional[int] = None
    """Sliding window attention window size."""

    window_attn_skip_freq: Optional[int] = None
    """Frequency of full attention layers. If N, every Nth layer uses full attention,
    others use sliding window attention. None means all layers use SWA if window_size is set."""

    kv_channels: Optional[int] = None
    """Key/Value channels per attention head. Defaults to hidden_size / num_attention_heads."""

    gated_linear_unit: bool = True
    """Whether MLP uses gated linear unit (e.g., SwiGLU). Default True for modern architectures."""


def gpt3(config: FLOPSConfig):
    """Model FLOPs for GPT3 family."""
    return (
        24 * config.gbs * config.enc_seq_len * config.hs * config.hs
        + 4 * config.gbs * config.enc_seq_len * config.enc_seq_len * config.hs
    ) * (3 * config.layers) + (
        6 * config.gbs * config.enc_seq_len * config.hs * config.vocab_size
    )


def llama(config: FLOPSConfig):
    """Model FLOPs for llama3 family."""
    return (
        config.gbs
        * config.enc_seq_len
        * config.layers
        * config.hs
        * config.hs
        * (
            12
            + (12 * config.query_groups / config.attention_heads)
            + (18 * config.ffn_hs / config.hs)
            + (6 * config.enc_seq_len / config.hs)
            + (6 * config.vocab_size / (config.layers * config.hs))
        )
    )


def nemotron(config: FLOPSConfig):
    """Model FLOPs for nemotron family."""
    return (
        config.gbs
        * config.enc_seq_len
        * config.layers
        * config.hs
        * config.hs
        * (
            12
            + (12 * config.query_groups / config.attention_heads)
            + (12 * config.ffn_hs / config.hs)
            + (12 * config.enc_seq_len / config.hs)
            + (6 * config.vocab_size / (config.layers * config.hs))
        )
    )


def mixtral(config: FLOPSConfig):
    """Model FLOPs for mixtral family."""
    return (
        config.gbs
        * config.enc_seq_len
        * config.layers
        * config.hs
        * config.hs
        * (
            12
            + (12 * config.query_groups / config.attention_heads)
            + (18 * config.moe_router_topk * config.ffn_hs / config.hs)
            + (12 * config.enc_seq_len / config.hs)
            + (6 * config.vocab_size / (config.layers * config.hs))
        )
    )


def qwen2(config: FLOPSConfig):
    """Model FLOPs for Qwen2 family."""
    causal_self_attn = True
    seq_len = config.enc_seq_len
    hidden_size = config.hs
    gated_linear_multiplier = 2

    # attention flops for GQA
    attention_flops = (
        3
        * 2
        * config.gbs
        * config.layers
        * seq_len
        * hidden_size
        * hidden_size
        * (
            (2 + 1)  # QKV gemm
            + (
                seq_len / hidden_size * 2 * (0.5 if causal_self_attn else 1)
            )  # attention
            + 1  # attention proj gemm
        )
    )

    # mlp flops
    mlp_flops = (
        3
        * 2
        * config.gbs
        * config.layers
        * seq_len
        * hidden_size
        * (1 + gated_linear_multiplier)
        * config.ffn_hs
    )

    # vocab flops
    vocab_flops = 3 * 2 * config.gbs * seq_len * hidden_size * config.vocab_size

    return attention_flops + mlp_flops + vocab_flops


def qwen3(config: FLOPSConfig):
    """Model FLOPs for Qwen3 family."""
    causal_self_attn = True
    seq_len = config.enc_seq_len
    hidden_size = config.hs
    gated_linear_multiplier = 2

    # attention flops for GQA
    attention_flops = (
        3
        * 2
        * config.gbs
        * config.layers
        * seq_len
        * hidden_size
        * hidden_size
        * (
            (config.query_groups / config.attention_heads * 2 + 1)  # QKV gemm
            + (
                seq_len / hidden_size * 2 * (0.5 if causal_self_attn else 1)
            )  # attention
            + 1  # attention proj gemm
        )
    )

    # mlp flops
    mlp_flops = (
        3
        * 2
        * config.gbs
        * config.layers
        * seq_len
        * hidden_size
        * (1 + gated_linear_multiplier)
        * (config.moe_ffn_hidden_size * config.moe_router_topk)  # MoE layers
    )

    # vocab flops
    vocab_flops = 3 * 2 * config.gbs * seq_len * hidden_size * config.vocab_size

    return attention_flops + mlp_flops + vocab_flops


def bert(config: FLOPSConfig):
    """Model FLOPs for BERT family."""
    return (
        72
        * config.gbs
        * config.layers
        * config.enc_seq_len
        * config.hs
        * config.hs
        * (
            1
            + (config.enc_seq_len / (6 * config.hs))
            + (config.vocab_size / (12 * config.hs * config.layers))
        )
    )


def transformer(config: FLOPSConfig):
    """Calculate FLOPs for a standard Transformer model.

    Note: This does not cover encoder-decoder models.
    """
    # Extract parameters from config
    batch_size = config.gbs
    hidden_size = config.hs
    seq_length = config.enc_seq_len
    num_layers = config.layers
    num_attention_heads = config.attention_heads
    ffn_hidden_size = config.ffn_hs
    vocab_size = config.vocab_size

    if vocab_size is None:
        raise ValueError("vocab_size is required for transformer FLOPs calculation")

    # Handle optional parameters with reasonable defaults
    query_groups = (
        config.query_groups if config.query_groups is not None else num_attention_heads
    )
    causal_self_attn = (
        config.causal_self_attn if config.causal_self_attn is not None else False
    )
    moe_router_topk = (
        config.moe_router_topk if config.moe_router_topk is not None else 0
    )
    kv_channels = hidden_size // num_attention_heads  # Standard dimension per head

    # Calculate query projection size and ratio
    query_projection_size = kv_channels * num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / hidden_size

    # MoE parameters - simplified for NeMo config
    # In this implementation, we assume all layers are dense if num_experts is None
    if moe_router_topk == 0:
        num_dense_layers = num_layers
        num_moe_layers = 0
        num_experts_routed_to = 0
    else:
        # Simplified MoE handling - assuming uniform distribution of MoE layers
        # This can be expanded based on NeMo's actual MoE implementation
        num_moe_layers = num_layers // 2  # Simplified assumption
        num_dense_layers = num_layers - num_moe_layers
        num_experts_routed_to = moe_router_topk

    # Handle SwiGLU vs standard GELU/ReLU
    # Default to standard activation (no SwiGLU)
    gated_linear_multiplier = 1

    # Define the expansion factor as described in the paper
    # 3x: Each GEMM needs forward pass, backward wgrad, and backward dgrad
    # 2x: GEMMs are stacked twice in standard Transformer architectures
    # 2x: A GEMM of m*n with n*k requires 2mnk floating-point operations
    expansion_factor = 3 * 2 * 2
    # Attention
    if not causal_self_attn:
        attention_component = (
            1
            + (query_groups / num_attention_heads)
            # Only half of the attention matrix is non-zero and needs to be multiplied with V
            + (seq_length / hidden_size)  # If causal self attn -> divide by 2.
        ) * query_projection_to_hidden_size_ratio
    else:
        attention_component = (
            1
            + (query_groups / num_attention_heads)
            # Only half of the attention matrix is non-zero and needs to be multiplied with V
            + (seq_length / hidden_size / 2)  # If causal self attn -> divide by 2.
        ) * query_projection_to_hidden_size_ratio

    # Calculate total FLOPs
    total_flops = (
        expansion_factor
        * batch_size
        * seq_length
        * num_layers
        * hidden_size
        * hidden_size
        * (
            attention_component
            # MLP component
            + (
                (
                    # Dense layers
                    (ffn_hidden_size * num_dense_layers)
                    +
                    # MoE layers
                    (
                        (
                            # Routed experts
                            ffn_hidden_size * num_experts_routed_to
                            # Note: Shared experts are not implemented in this version
                        )
                        * num_moe_layers
                    )
                )
                * gated_linear_multiplier
                / (num_layers * hidden_size)
            )
            # Logit component
            + (vocab_size / (2 * num_layers * hidden_size))
        )
    )

    return total_flops


def flux(config: FLOPSConfig):
    """Model FLOPs for FLUX."""
    hs = config.hs
    seq_len = config.model_channels + config.inp_s
    base_factor = 6 * config.gbs  # common multiplier for most terms

    # Joint layer computations
    joint_layer_flops = (
        base_factor
        * config.layers[0]
        * (
            10 * hs * hs  # hidden size operations
            + 2
            * hs
            * (config.model_channels + config.inp_s)
            * (1 + hs * 7)  # channel and context joint attention
            + 2 * (config.model_channels + config.inp_s) * hs  # final projection
        )
    )

    # Single layer computations
    single_layer_flops = (
        base_factor
        * config.layers[1]
        * seq_len
        * hs
        * (
            3  # linear Y
            + 1  # Modulation
            + 4 * hs  # Linear computations
            + (3 * hs + 2 * seq_len)  # attention operations
            + 5 * hs  # feed-forward
            + 1  # Modulation
        )
    )

    # Embedding and projection layers
    other_flops = base_factor * (
        config.inp_s * config.in_channels * hs  # image embedding
        + config.inp_s * hs * config.model_channels  # text embedding
        + config.vec_in_dim * hs
        + hs * hs  # vector embedding
        + 2 * (config.model_channels * hs + hs * hs)  # guidance + timestep embedding
        + (config.inp_s * config.in_channels * hs) / config.gbs  # final projection
    )

    return joint_layer_flops + single_layer_flops + other_flops


def deepseekv3(config: FLOPSConfig):
    """Model FLOPs for DeepSeek V3."""
    # self-attention flops
    bmm1_flops = (
        0.5
        * (config.qk_head_dim + config.qk_pos_emb_head_dim)
        * config.attention_heads
        * (config.enc_seq_len**2)
    )
    bmm2_flops = (
        0.5 * config.v_head_dim * config.attention_heads * (config.enc_seq_len**2)
    )
    per_input_attention_flops = 6 * (bmm1_flops + bmm2_flops) * config.layers
    if config.mtp_num_layers is not None:
        per_input_attention_flops += (
            6 * (bmm1_flops + bmm2_flops) * config.mtp_num_layers
        )

    # linear layer flops
    # Q projection: check if using MLA (q_lora_rank is set) or standard attention
    if config.q_lora_rank is not None:
        # MLA for Q (e.g., DeepSeek-V3)
        per_layer_mla_params = config.hs * config.q_lora_rank + config.q_lora_rank * (
            (config.qk_head_dim + config.qk_pos_emb_head_dim) * config.attention_heads
        )  # Q
    else:
        # Standard attention for Q (e.g., Moonlight)
        per_layer_mla_params = config.hs * (
            (config.qk_head_dim + config.qk_pos_emb_head_dim) * config.attention_heads
        )  # Q

    per_layer_mla_params += config.hs * config.qk_pos_emb_head_dim  # K^R
    per_layer_mla_params += config.hs * config.kv_lora_rank + config.kv_lora_rank * (
        (config.qk_head_dim + config.v_head_dim) * config.attention_heads
    )  # K^C and V^C
    per_layer_mla_params += (
        config.v_head_dim * config.attention_heads * config.hs
    )  # Proj
    mla_params = per_layer_mla_params * config.layers
    if config.mtp_num_layers is not None:
        mla_params += per_layer_mla_params * config.mtp_num_layers

    dense_layer_ffn_params = config.hs * config.ffn_hs * 3  # gated linear unit
    per_shared_expert_params = (
        config.hs * config.moe_shared_expert_intermediate_size * 3
    )
    per_selected_expert_params = config.hs * config.moe_ffn_hidden_size * 3
    ffn_params = 0

    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [
            1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.layers)
        ]
    else:
        moe_layer_pattern = config.moe_layer_freq
    for i in moe_layer_pattern:
        if i == 0:
            ffn_params += dense_layer_ffn_params
        else:
            ffn_params += per_shared_expert_params + (
                per_selected_expert_params * config.moe_router_topk
            )
    if config.mtp_num_layers is not None:
        for i in range(config.mtp_num_layers):
            ffn_params += per_shared_expert_params + (
                per_selected_expert_params * config.moe_router_topk
            )
    per_input_params = mla_params + ffn_params
    per_input_linear_flops = 6 * per_input_params * config.enc_seq_len

    # vocab flops
    per_input_vocab_flops = 6 * config.vocab_size * config.hs * config.enc_seq_len
    if config.mtp_num_layers is not None:
        for i in range(config.mtp_num_layers):
            per_input_vocab_flops += (
                6 * config.vocab_size * config.hs * config.enc_seq_len
            )
            per_input_vocab_flops += 6 * config.hs * 2 * config.hs * config.enc_seq_len

    return (
        per_input_attention_flops + per_input_linear_flops + per_input_vocab_flops
    ) * config.gbs


def _mlp_layer_flops(config: FLOPSConfig):
    """Model FLOPs for MLP layer."""
    return (
        6
        * config.gbs
        * config.enc_seq_len
        * config.hs
        * config.ffn_hs
        * (2 if config.gated_linear_unit else 1)
    )


def _non_mla_attn_layer_flops(config: FLOPSConfig):
    """Model FLOPs for attention layer."""
    return (
        6
        * config.gbs
        * config.enc_seq_len
        * config.hs
        * (
            config.hs  # Q
            + config.query_groups / config.attention_heads * config.hs * 2  # KV
            + config.enc_seq_len / 2 * 2
            + config.hs
        )
    )


def _mamba_layer_flops(config: FLOPSConfig):
    """Model FLOPs for Mamba layer. We ignore part of the flops of scan because the chunk size is not known from model config."""
    assert config.mamba_state_dim is not None
    assert config.mamba_head_dim is not None

    if config.mamba_num_heads:
        nheads = config.mamba_num_heads
    else:
        nheads = 2 * config.hs // config.mamba_head_dim  # default expand is 2
    d_in = nheads * config.mamba_head_dim
    return (
        (
            6
            * config.gbs
            * config.enc_seq_len
            * config.hs
            * (2 * d_in + 2 * config.mamba_num_groups * config.mamba_state_dim + nheads)
        )
        + (3 * 2 * config.gbs * config.enc_seq_len * d_in * config.mamba_state_dim)
        + (6 * config.gbs * config.enc_seq_len * d_in * config.hs)
    )


def _hybrid_model_flops(config: FLOPSConfig):
    """Model FLOPs for hybrid model."""
    assert config.is_hybrid_model == True
    assert config.hybrid_override_pattern is not None

    num_attn_layers, num_mamba_layers, num_mlp_layers = 0, 0, 0
    for c in config.hybrid_override_pattern:
        if c == "M":
            num_mamba_layers += 1
        elif c == "-":
            num_mlp_layers += 1
        elif c == "*":
            num_attn_layers += 1
    return (
        num_attn_layers * _non_mla_attn_layer_flops(config)
        + num_mamba_layers * _mamba_layer_flops(config)
        + num_mlp_layers * _mlp_layer_flops(config)
        + 6 * config.gbs * config.enc_seq_len * config.hs * config.vocab_size
    )


def nemotronh(config: FLOPSConfig):
    """Model FLOPs for NemotronH."""
    return _hybrid_model_flops(config)


# =============================================================================
# GPT-OSS FLOPS Calculation Utilities
# =============================================================================


def _compute_kv_channels(config: FLOPSConfig) -> int:
    """Compute key/value channels per head, with fallback to hidden_size / num_heads."""
    if config.kv_channels is not None:
        return config.kv_channels
    return config.hs // config.attention_heads


def _is_layer_window_attention(
    window_attn_skip_freq: Optional[int],
    layer_idx: int,
) -> bool:
    """Determine if a layer uses sliding window attention.

    Args:
        window_attn_skip_freq: If N, every Nth layer (1-indexed) uses full attention.
            None means all layers use sliding window attention.
        layer_idx: 0-indexed layer number.

    Returns:
        True if the layer uses sliding window attention, False for full attention.
    """
    if window_attn_skip_freq is None:
        return True
    # Convert to 1-indexed for skip frequency check
    layer_number = layer_idx + 1
    # layer_number % freq == 0 means full attention
    return layer_number % window_attn_skip_freq != 0


def _attention_flops(
    seq_len: int,
    hidden_size: int,
    num_attention_heads: int,
    num_query_groups: int,
    kv_channels: int,
    is_swa: bool = False,
    swa_window_size: int = 128,
) -> int:
    """Calculate FLOPS for a single attention layer.

    This follows the standard transformer attention computation:
    - QKV linear projections
    - Attention score computation (Q @ K^T)
    - Attention output (scores @ V)
    - Output projection

    Args:
        seq_len: Sequence length.
        hidden_size: Model hidden size.
        num_attention_heads: Number of attention heads for queries.
        num_query_groups: Number of key-value groups (for GQA/MQA).
        kv_channels: Dimension per attention head.
        is_swa: Whether to use sliding window attention.
        swa_window_size: Window size for sliding window attention.

    Returns:
        Total FLOPS for the attention layer (forward + backward = 6x multiplier).
    """
    # QKV linear projection: hidden -> (q_heads + 2 * kv_groups) * kv_channels
    linear_qkv = seq_len * hidden_size * (
        kv_channels * (num_attention_heads + num_query_groups * 2)
    )

    # Output projection: q_heads * kv_channels -> hidden
    linear_proj = seq_len * hidden_size * (kv_channels * num_attention_heads)

    # Attention computation
    if is_swa:
        # Sliding window attention: reduced attention span
        # For causal SWA, non-zero elements = triangle + rectangle
        attention_mask_nz_elem = (
            swa_window_size * (swa_window_size + 1) / 2
            + (seq_len - swa_window_size) * swa_window_size
        )
        attention = num_attention_heads * (attention_mask_nz_elem * kv_channels) * 2
    else:
        # Full causal attention
        bmm_k = kv_channels
        bmm_b = num_attention_heads
        attention_mask_nz_elem = seq_len * (seq_len + 1) / 2
        attention = bmm_b * attention_mask_nz_elem * bmm_k * 2

    # 6x multiplier: 3x for forward/backward (fwd + wgrad + dgrad), 2x for GEMM operations
    return int((linear_qkv + linear_proj + attention) * 6)


def _moe_mlp_flops(
    seq_len: int,
    hidden_size: int,
    moe_ffn_hidden_size: int,
    moe_router_topk: int,
    gated_linear_unit: bool = True,
) -> int:
    """Calculate FLOPS for a single MoE MLP layer.

    Args:
        seq_len: Sequence length.
        hidden_size: Model hidden size.
        moe_ffn_hidden_size: Hidden size of each expert's FFN.
        moe_router_topk: Number of experts activated per token.
        gated_linear_unit: Whether using gated activation (e.g., SwiGLU).

    Returns:
        Total FLOPS for the MoE MLP layer (forward + backward = 6x multiplier).
    """
    # Total tokens processed = seq_len * topk (each token goes to topk experts)
    total_num_tokens = seq_len * moe_router_topk

    # FC1: hidden -> ffn_hidden (2x if gated for gate + value)
    glu_multiplier = 2 if gated_linear_unit else 1
    linear_fc1 = total_num_tokens * hidden_size * moe_ffn_hidden_size * glu_multiplier

    # FC2: ffn_hidden -> hidden
    linear_fc2 = total_num_tokens * moe_ffn_hidden_size * hidden_size

    return int((linear_fc1 + linear_fc2) * 6)


def _vocab_flops(
    seq_len: int,
    hidden_size: int,
    vocab_size: int,
) -> int:
    """Calculate FLOPS for the vocabulary/output projection layer.

    Args:
        seq_len: Sequence length.
        hidden_size: Model hidden size.
        vocab_size: Vocabulary size.

    Returns:
        Total FLOPS for the vocab layer (forward + backward = 6x multiplier).
    """
    return int(seq_len * hidden_size * vocab_size * 6)


def gpt_oss(config: FLOPSConfig) -> int:
    """Model FLOPS for GPT-OSS family.

    GPT-OSS is a Mixture-of-Experts model with sliding window attention:
    - Uses MoE (Mixture of Experts) for FFN layers
    - Some layers use full attention, others use sliding window attention
    - The pattern is controlled by window_attn_skip_freq

    Args:
        config: FLOPSConfig with the following required fields:
            - gbs: Global batch size
            - enc_seq_len: Sequence length
            - hs: Hidden size
            - layers: Number of transformer layers
            - attention_heads: Number of attention heads
            - query_groups: Number of KV groups (for GQA)
            - moe_ffn_hidden_size: Expert FFN hidden size
            - moe_router_topk: Number of experts per token
            - vocab_size: Vocabulary size
            And optional fields:
            - swa_window_size: Sliding window size (default: 128)
            - window_attn_skip_freq: Full attention frequency (default: 2)
            - kv_channels: KV dimension per head (default: hs / attention_heads)
            - gated_linear_unit: Whether using gated MLP (default: True)

    Returns:
        Total FLOPS for the model.
    """
    seq_len = config.enc_seq_len
    hidden_size = config.hs
    num_layers = config.layers

    # Get optional parameters with defaults
    swa_window_size = config.swa_window_size if config.swa_window_size is not None else 128
    window_attn_skip_freq = (
        config.window_attn_skip_freq if config.window_attn_skip_freq is not None else 2
    )
    kv_channels = _compute_kv_channels(config)
    gated_linear_unit = config.gated_linear_unit

    total_flops = 0

    # Per-layer FLOPS
    for layer_idx in range(num_layers):
        # Determine attention type for this layer
        is_swa = _is_layer_window_attention(window_attn_skip_freq, layer_idx)

        # Attention FLOPS
        attn_flops = _attention_flops(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_attention_heads=config.attention_heads,
            num_query_groups=config.query_groups,
            kv_channels=kv_channels,
            is_swa=is_swa,
            swa_window_size=swa_window_size,
        )

        # MoE MLP FLOPS
        mlp_flops = _moe_mlp_flops(
            seq_len=seq_len,
            hidden_size=hidden_size,
            moe_ffn_hidden_size=config.moe_ffn_hidden_size,
            moe_router_topk=config.moe_router_topk,
            gated_linear_unit=gated_linear_unit,
        )

        total_flops += attn_flops + mlp_flops

    # Vocabulary/output projection FLOPS
    total_flops += _vocab_flops(seq_len, hidden_size, config.vocab_size)

    # Multiply by global batch size
    return total_flops * config.gbs
