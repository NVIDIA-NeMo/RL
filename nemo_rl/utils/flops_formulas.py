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
from typing import List, Optional, TypeVar, Union

T = TypeVar("T")


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
    head_dim: Optional[int] = None
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
    dsa_indexer_n_heads: Optional[int] = None
    dsa_indexer_head_dim: Optional[int] = None
    dsa_indexer_topk: Optional[int] = None
    dsa_indexer_compute_layers: Optional[int] = None


def _require(value: Optional[T], name: str) -> T:
    """Return the value of a FLOPSConfig field that the calling formula needs.

    Args:
        value: The field value read from the config.
        name: The field name, used in the error message.

    Returns:
        The field value.

    Raises:
        ValueError: If the field is unset.
    """
    if value is None:
        raise ValueError(f"FLOPSConfig.{name} is required for this FLOPs calculation")
    return value


def gpt3(config: FLOPSConfig):
    """Model FLOPs for GPT3 family."""
    enc_seq_len = _require(config.enc_seq_len, "enc_seq_len")
    hs = _require(config.hs, "hs")
    layers = _require(config.layers, "layers")
    vocab_size = _require(config.vocab_size, "vocab_size")
    return (
        24 * config.gbs * enc_seq_len * hs * hs
        + 4 * config.gbs * enc_seq_len * enc_seq_len * hs
    ) * (3 * layers) + (6 * config.gbs * enc_seq_len * hs * vocab_size)


def llama(config: FLOPSConfig):
    """Model FLOPs for llama3 family."""
    enc_seq_len = _require(config.enc_seq_len, "enc_seq_len")
    hs = _require(config.hs, "hs")
    layers = _require(config.layers, "layers")
    ffn_hs = _require(config.ffn_hs, "ffn_hs")
    attention_heads = _require(config.attention_heads, "attention_heads")
    query_groups = _require(config.query_groups, "query_groups")
    vocab_size = _require(config.vocab_size, "vocab_size")
    return (
        config.gbs
        * enc_seq_len
        * layers
        * hs
        * hs
        * (
            12
            + (12 * query_groups / attention_heads)
            + (18 * ffn_hs / hs)
            + (6 * enc_seq_len / hs)
            + (6 * vocab_size / (layers * hs))
        )
    )


def nemotron(config: FLOPSConfig):
    """Model FLOPs for nemotron family."""
    enc_seq_len = _require(config.enc_seq_len, "enc_seq_len")
    hs = _require(config.hs, "hs")
    layers = _require(config.layers, "layers")
    ffn_hs = _require(config.ffn_hs, "ffn_hs")
    attention_heads = _require(config.attention_heads, "attention_heads")
    query_groups = _require(config.query_groups, "query_groups")
    vocab_size = _require(config.vocab_size, "vocab_size")
    return (
        config.gbs
        * enc_seq_len
        * layers
        * hs
        * hs
        * (
            12
            + (12 * query_groups / attention_heads)
            + (12 * ffn_hs / hs)
            + (12 * enc_seq_len / hs)
            + (6 * vocab_size / (layers * hs))
        )
    )


def mixtral(config: FLOPSConfig):
    """Model FLOPs for mixtral family."""
    enc_seq_len = _require(config.enc_seq_len, "enc_seq_len")
    hs = _require(config.hs, "hs")
    layers = _require(config.layers, "layers")
    ffn_hs = _require(config.ffn_hs, "ffn_hs")
    attention_heads = _require(config.attention_heads, "attention_heads")
    moe_router_topk = _require(config.moe_router_topk, "moe_router_topk")
    query_groups = _require(config.query_groups, "query_groups")
    vocab_size = _require(config.vocab_size, "vocab_size")
    return (
        config.gbs
        * enc_seq_len
        * layers
        * hs
        * hs
        * (
            12
            + (12 * query_groups / attention_heads)
            + (18 * moe_router_topk * ffn_hs / hs)
            + (12 * enc_seq_len / hs)
            + (6 * vocab_size / (layers * hs))
        )
    )


def qwen2(config: FLOPSConfig):
    """Model FLOPs for Qwen2 family."""
    causal_self_attn = True
    seq_len = _require(config.enc_seq_len, "enc_seq_len")
    hidden_size = _require(config.hs, "hs")
    layers = _require(config.layers, "layers")
    ffn_hs = _require(config.ffn_hs, "ffn_hs")
    vocab_size = _require(config.vocab_size, "vocab_size")
    gated_linear_multiplier = 2

    # attention flops for GQA
    attention_flops = (
        3
        * 2
        * config.gbs
        * layers
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
        * layers
        * seq_len
        * hidden_size
        * (1 + gated_linear_multiplier)
        * ffn_hs
    )

    # vocab flops
    vocab_flops = 3 * 2 * config.gbs * seq_len * hidden_size * vocab_size

    return attention_flops + mlp_flops + vocab_flops


def qwen3(config: FLOPSConfig):
    """Model FLOPs for Qwen3 family."""
    causal_self_attn = True
    seq_len = _require(config.enc_seq_len, "enc_seq_len")
    hidden_size = _require(config.hs, "hs")
    layers = _require(config.layers, "layers")
    attention_heads = _require(config.attention_heads, "attention_heads")
    moe_router_topk = _require(config.moe_router_topk, "moe_router_topk")
    query_groups = _require(config.query_groups, "query_groups")
    vocab_size = _require(config.vocab_size, "vocab_size")
    moe_ffn_hidden_size = _require(config.moe_ffn_hidden_size, "moe_ffn_hidden_size")
    gated_linear_multiplier = 2

    # head_dim defaults to hidden_size / num_heads, but Qwen3 sets it explicitly and for some
    # models (e.g. Qwen3-235B-A22B: head_dim=128, num_heads=64, hidden=4096) num_heads * head_dim
    # != hidden_size ("wide" attention). The QKV/output projections and the O(seq^2) attention
    # scores scale with num_heads * head_dim, NOT hidden_size, so use the real projection sizes.
    # This reduces exactly to the previous formula when num_heads * head_dim == hidden_size.
    head_dim = (
        config.head_dim
        if config.head_dim is not None
        else hidden_size // attention_heads
    )
    q_proj_size = attention_heads * head_dim
    kv_proj_size = query_groups * head_dim

    # attention flops for GQA
    attention_flops = (
        3
        * 2
        * config.gbs
        * layers
        * seq_len
        * (
            q_proj_size * hidden_size  # Q projection
            + 2 * kv_proj_size * hidden_size  # K, V projections (GQA)
            + q_proj_size * hidden_size  # output projection
            + seq_len
            * q_proj_size
            * (2 * (0.5 if causal_self_attn else 1))  # QK^T + A.V scores
        )
    )

    # mlp flops
    mlp_flops = (
        3
        * 2
        * config.gbs
        * layers
        * seq_len
        * hidden_size
        * (1 + gated_linear_multiplier)
        * (moe_ffn_hidden_size * moe_router_topk)  # MoE layers
    )

    # vocab flops
    vocab_flops = 3 * 2 * config.gbs * seq_len * hidden_size * vocab_size

    return attention_flops + mlp_flops + vocab_flops


def bert(config: FLOPSConfig):
    """Model FLOPs for BERT family."""
    enc_seq_len = _require(config.enc_seq_len, "enc_seq_len")
    hs = _require(config.hs, "hs")
    layers = _require(config.layers, "layers")
    vocab_size = _require(config.vocab_size, "vocab_size")
    return (
        72
        * config.gbs
        * layers
        * enc_seq_len
        * hs
        * hs
        * (1 + (enc_seq_len / (6 * hs)) + (vocab_size / (12 * hs * layers)))
    )


def transformer(config: FLOPSConfig):
    """Calculate FLOPs for a standard Transformer model.

    Note: This does not cover encoder-decoder models.
    """
    # Extract parameters from config
    batch_size = config.gbs
    vocab_size = config.vocab_size

    if vocab_size is None:
        raise ValueError("vocab_size is required for transformer FLOPs calculation")

    hidden_size = _require(config.hs, "hs")
    seq_length = _require(config.enc_seq_len, "enc_seq_len")
    num_layers = _require(config.layers, "layers")
    num_attention_heads = _require(config.attention_heads, "attention_heads")
    ffn_hidden_size = _require(config.ffn_hs, "ffn_hs")

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
    hs = _require(config.hs, "hs")
    layers = _require(config.layers, "layers")
    in_channels = _require(config.in_channels, "in_channels")
    inp_s = _require(config.inp_s, "inp_s")
    model_channels = _require(config.model_channels, "model_channels")
    vec_in_dim = _require(config.vec_in_dim, "vec_in_dim")
    seq_len = model_channels + inp_s
    base_factor = 6 * config.gbs  # common multiplier for most terms

    # Joint layer computations
    joint_layer_flops = (
        base_factor
        * layers[0]
        * (
            10 * hs * hs  # hidden size operations
            + 2
            * hs
            * (model_channels + inp_s)
            * (1 + hs * 7)  # channel and context joint attention
            + 2 * (model_channels + inp_s) * hs  # final projection
        )
    )

    # Single layer computations
    single_layer_flops = (
        base_factor
        * layers[1]
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
        inp_s * in_channels * hs  # image embedding
        + inp_s * hs * model_channels  # text embedding
        + vec_in_dim * hs
        + hs * hs  # vector embedding
        + 2 * (model_channels * hs + hs * hs)  # guidance + timestep embedding
        + (inp_s * in_channels * hs) / config.gbs  # final projection
    )

    return joint_layer_flops + single_layer_flops + other_flops


def _mla_projection_params(config: FLOPSConfig) -> int:
    """Return the per-layer MLA projection parameter count."""
    hs = _require(config.hs, "hs")
    attention_heads = _require(config.attention_heads, "attention_heads")
    kv_lora_rank = _require(config.kv_lora_rank, "kv_lora_rank")
    qk_head_dim = _require(config.qk_head_dim, "qk_head_dim")
    qk_pos_emb_head_dim = _require(config.qk_pos_emb_head_dim, "qk_pos_emb_head_dim")
    v_head_dim = _require(config.v_head_dim, "v_head_dim")
    if config.q_lora_rank is not None:
        per_layer_mla_params = hs * config.q_lora_rank + config.q_lora_rank * (
            (qk_head_dim + qk_pos_emb_head_dim) * attention_heads
        )
    else:
        per_layer_mla_params = hs * (
            (qk_head_dim + qk_pos_emb_head_dim) * attention_heads
        )

    per_layer_mla_params += hs * qk_pos_emb_head_dim
    per_layer_mla_params += hs * kv_lora_rank + kv_lora_rank * (
        (qk_head_dim + v_head_dim) * attention_heads
    )
    per_layer_mla_params += v_head_dim * attention_heads * hs
    return per_layer_mla_params


def _moe_ffn_params(config: FLOPSConfig) -> int:
    """Return activated dense/MoE FFN parameters for all model and MTP layers."""
    hs = _require(config.hs, "hs")
    ffn_hs = _require(config.ffn_hs, "ffn_hs")
    moe_shared_expert_intermediate_size = _require(
        config.moe_shared_expert_intermediate_size,
        "moe_shared_expert_intermediate_size",
    )
    moe_ffn_hidden_size = _require(config.moe_ffn_hidden_size, "moe_ffn_hidden_size")
    dense_layer_ffn_params = hs * ffn_hs * 3  # gated linear unit
    per_shared_expert_params = hs * moe_shared_expert_intermediate_size * 3
    per_selected_expert_params = hs * moe_ffn_hidden_size * 3
    ffn_params = 0

    moe_layer_freq = _require(config.moe_layer_freq, "moe_layer_freq")
    if isinstance(moe_layer_freq, int):
        layers = _require(config.layers, "layers")
        moe_layer_pattern = [
            1 if (i % moe_layer_freq == 0) else 0 for i in range(layers)
        ]
    else:
        moe_layer_pattern = moe_layer_freq

    # moe_router_topk is read only where a MoE layer needs it: a config whose
    # layers are all dense does not have to set it.
    for i in moe_layer_pattern:
        if i == 0:
            ffn_params += dense_layer_ffn_params
        else:
            ffn_params += per_shared_expert_params + (
                per_selected_expert_params
                * _require(config.moe_router_topk, "moe_router_topk")
            )
    for _ in range(config.mtp_num_layers or 0):
        ffn_params += per_shared_expert_params + (
            per_selected_expert_params
            * _require(config.moe_router_topk, "moe_router_topk")
        )
    return ffn_params


def _mla_moe_linear_flops(config: FLOPSConfig) -> int:
    """Return MLA projection and activated FFN training FLOPs per input."""
    enc_seq_len = _require(config.enc_seq_len, "enc_seq_len")
    layers = _require(config.layers, "layers")
    mla_layers = layers + (config.mtp_num_layers or 0)
    mla_params = _mla_projection_params(config) * mla_layers
    return 6 * (mla_params + _moe_ffn_params(config)) * enc_seq_len


def _vocab_and_mtp_flops(config: FLOPSConfig) -> int:
    """Return output projection and MTP embedding projection FLOPs per input."""
    enc_seq_len = _require(config.enc_seq_len, "enc_seq_len")
    hs = _require(config.hs, "hs")
    vocab_size = _require(config.vocab_size, "vocab_size")
    vocab_flops = 6 * vocab_size * hs * enc_seq_len
    for _ in range(config.mtp_num_layers or 0):
        vocab_flops += 6 * vocab_size * hs * enc_seq_len
        vocab_flops += 6 * hs * 2 * hs * enc_seq_len
    return vocab_flops


def deepseekv3(config: FLOPSConfig) -> float:
    """Model FLOPs for DeepSeek V3."""
    enc_seq_len = _require(config.enc_seq_len, "enc_seq_len")
    layers = _require(config.layers, "layers")
    attention_heads = _require(config.attention_heads, "attention_heads")
    qk_head_dim = _require(config.qk_head_dim, "qk_head_dim")
    qk_pos_emb_head_dim = _require(config.qk_pos_emb_head_dim, "qk_pos_emb_head_dim")
    v_head_dim = _require(config.v_head_dim, "v_head_dim")
    bmm1_flops = (
        0.5 * (qk_head_dim + qk_pos_emb_head_dim) * attention_heads * (enc_seq_len**2)
    )
    bmm2_flops = 0.5 * v_head_dim * attention_heads * (enc_seq_len**2)
    per_input_attention_flops = 6 * (bmm1_flops + bmm2_flops) * layers
    if config.mtp_num_layers is not None:
        per_input_attention_flops += (
            6 * (bmm1_flops + bmm2_flops) * config.mtp_num_layers
        )

    return (
        per_input_attention_flops
        + _mla_moe_linear_flops(config)
        + _vocab_and_mtp_flops(config)
    ) * config.gbs


def _causal_topk_pairs(seq_len: int, topk: int) -> float:
    """Approximate causal sparse-attention query/key pairs with dense formula conventions."""
    effective_topk = min(topk, seq_len)
    if effective_topk == seq_len:
        return 0.5 * seq_len * seq_len
    return effective_topk * seq_len - 0.5 * effective_topk**2


def glm_moe_dsa(config: FLOPSConfig) -> float:
    """Model FLOPs for GLM MoE DSA.

    GLM-MoE-DSA uses MLA projections and MoE FFNs like DeepSeek-style MoE models,
    but its core attention path uses a DSA indexer to select sparse top-k keys.
    The indexer is forward-only in the RL runs where indexer loss is disabled, so
    its matmuls are counted with a single 2x FMA factor rather than the 6x
    forward+backward training factor used for trainable model-path GEMMs.
    """
    seq_len = _require(config.enc_seq_len, "enc_seq_len")
    hs = _require(config.hs, "hs")
    layers = _require(config.layers, "layers")
    attention_heads = _require(config.attention_heads, "attention_heads")
    q_lora_rank = _require(config.q_lora_rank, "q_lora_rank")
    kv_lora_rank = _require(config.kv_lora_rank, "kv_lora_rank")
    qk_pos_emb_head_dim = _require(config.qk_pos_emb_head_dim, "qk_pos_emb_head_dim")
    dsa_indexer_n_heads = _require(config.dsa_indexer_n_heads, "dsa_indexer_n_heads")
    dsa_indexer_head_dim = _require(config.dsa_indexer_head_dim, "dsa_indexer_head_dim")
    dsa_indexer_topk = _require(config.dsa_indexer_topk, "dsa_indexer_topk")
    dsa_indexer_compute_layers = _require(
        config.dsa_indexer_compute_layers, "dsa_indexer_compute_layers"
    )

    # DSA sparse attention. In absorbed MLA DSA, scores use latent KV + RoPE
    # channels and value aggregation uses latent KV channels.
    sparse_pairs = _causal_topk_pairs(seq_len, dsa_indexer_topk)
    sparse_score_dim = kv_lora_rank + qk_pos_emb_head_dim
    sparse_value_dim = kv_lora_rank
    per_input_sparse_attention_flops = (
        6
        * layers
        * sparse_pairs
        * attention_heads
        * (sparse_score_dim + sparse_value_dim)
    )

    # DSA indexer: q_lora -> index heads, hidden -> index key, hidden -> head weights,
    # then dense index-score matmul before top-k selection. Top-k itself is not counted.
    indexer_projection_params = (
        q_lora_rank * dsa_indexer_n_heads * dsa_indexer_head_dim
        + hs * dsa_indexer_head_dim
        + hs * dsa_indexer_n_heads
    )
    # The indexer scores every causally valid query/key pair before selecting top-k.
    dense_causal_pairs = seq_len * (seq_len + 1) // 2
    # This formula assumes dsa_indexer_loss_coeff=0.0, for which MCore runs the
    # indexer under no_grad. Count only its forward matmuls (2 FLOPs per FMA);
    # enabling indexer loss also requires accounting for the backward pass.
    per_input_indexer_flops = (
        2
        * dsa_indexer_compute_layers
        * (
            seq_len * indexer_projection_params
            + dense_causal_pairs * dsa_indexer_n_heads * dsa_indexer_head_dim
        )
    )

    return (
        _mla_moe_linear_flops(config)
        + per_input_sparse_attention_flops
        + per_input_indexer_flops
        + _vocab_and_mtp_flops(config)
    ) * config.gbs


def _mlp_layer_flops(config: FLOPSConfig):
    """Model FLOPs for MLP layer."""
    enc_seq_len = _require(config.enc_seq_len, "enc_seq_len")
    hs = _require(config.hs, "hs")
    ffn_hs = _require(config.ffn_hs, "ffn_hs")
    return (
        6
        * config.gbs
        * enc_seq_len
        * hs
        * ffn_hs
        * (2 if config.gated_linear_unit else 1)
    )


def _non_mla_attn_layer_flops(config: FLOPSConfig):
    """Model FLOPs for attention layer."""
    enc_seq_len = _require(config.enc_seq_len, "enc_seq_len")
    hs = _require(config.hs, "hs")
    attention_heads = _require(config.attention_heads, "attention_heads")
    query_groups = _require(config.query_groups, "query_groups")
    return (
        6
        * config.gbs
        * enc_seq_len
        * hs
        * (
            hs  # Q
            + query_groups / attention_heads * hs * 2  # KV
            + enc_seq_len / 2 * 2
            + hs
        )
    )


def _mamba_layer_flops(config: FLOPSConfig):
    """Model FLOPs for Mamba layer. We ignore part of the flops of scan because the chunk size is not known from model config."""
    assert config.mamba_state_dim is not None
    assert config.mamba_head_dim is not None
    enc_seq_len = _require(config.enc_seq_len, "enc_seq_len")
    hs = _require(config.hs, "hs")
    mamba_num_groups = _require(config.mamba_num_groups, "mamba_num_groups")

    if config.mamba_num_heads:
        nheads = config.mamba_num_heads
    else:
        nheads = 2 * hs // config.mamba_head_dim  # default expand is 2
    d_in = nheads * config.mamba_head_dim
    return (
        (
            6
            * config.gbs
            * enc_seq_len
            * hs
            * (2 * d_in + 2 * mamba_num_groups * config.mamba_state_dim + nheads)
        )
        + (3 * 2 * config.gbs * enc_seq_len * d_in * config.mamba_state_dim)
        + (6 * config.gbs * enc_seq_len * d_in * hs)
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
        + 6
        * config.gbs
        * _require(config.enc_seq_len, "enc_seq_len")
        * _require(config.hs, "hs")
        * _require(config.vocab_size, "vocab_size")
    )


def nemotronh(config: FLOPSConfig):
    """Model FLOPs for NemotronH."""
    return _hybrid_model_flops(config)
