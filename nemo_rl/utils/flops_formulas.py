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

"""Theoretical training FLOPs per model family.

Each model family has one class that holds only the hyperparameters its formula reads
and computes the FLOPs of one training step in ``flops``. The formulas were lifted from
NeMo/nemo/utils/flops_formulas.py.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class _FlopsBase(BaseModel):
    """Common settings: instances are immutable and reject unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class LlamaFlops(_FlopsBase):
    """Model FLOPs for llama3 family.

    Attributes:
        hs: Hidden size.
        layers: Number of transformer layers.
        ffn_hs: Feed-forward hidden size.
        attention_heads: Number of attention heads.
        query_groups: Number of key/value heads (grouped-query attention).
        vocab_size: Vocabulary size.
    """

    kind: Literal["llama"] = "llama"
    hs: int
    layers: int
    ffn_hs: int
    attention_heads: int
    query_groups: int
    vocab_size: int

    def flops(self, batch_size: int, seq_len: int) -> float:
        """Return the FLOPs of one training step.

        Args:
            batch_size: Number of sequences in the step.
            seq_len: Padded sequence length.

        Returns:
            Forward plus backward FLOPs.
        """
        return (
            batch_size
            * seq_len
            * self.layers
            * self.hs
            * self.hs
            * (
                12
                + (12 * self.query_groups / self.attention_heads)
                + (18 * self.ffn_hs / self.hs)
                + (6 * seq_len / self.hs)
                + (6 * self.vocab_size / (self.layers * self.hs))
            )
        )


class Qwen2Flops(_FlopsBase):
    """Model FLOPs for Qwen2 family.

    Attributes:
        hs: Hidden size.
        layers: Number of transformer layers.
        ffn_hs: Feed-forward hidden size.
        vocab_size: Vocabulary size.
    """

    kind: Literal["qwen2"] = "qwen2"
    hs: int
    layers: int
    ffn_hs: int
    vocab_size: int

    def flops(self, batch_size: int, seq_len: int) -> float:
        """Return the FLOPs of one training step.

        Args:
            batch_size: Number of sequences in the step.
            seq_len: Padded sequence length.

        Returns:
            Forward plus backward FLOPs.
        """
        causal_self_attn = True
        gated_linear_multiplier = 2

        # attention flops for GQA
        attention_flops = (
            3
            * 2
            * batch_size
            * self.layers
            * seq_len
            * self.hs
            * self.hs
            * (
                (2 + 1)  # QKV gemm
                + (
                    seq_len / self.hs * 2 * (0.5 if causal_self_attn else 1)
                )  # attention
                + 1  # attention proj gemm
            )
        )

        # mlp flops
        mlp_flops = (
            3
            * 2
            * batch_size
            * self.layers
            * seq_len
            * self.hs
            * (1 + gated_linear_multiplier)
            * self.ffn_hs
        )

        # vocab flops
        vocab_flops = 3 * 2 * batch_size * seq_len * self.hs * self.vocab_size

        return attention_flops + mlp_flops + vocab_flops


class Qwen3Flops(_FlopsBase):
    """Model FLOPs for Qwen3 family.

    Attributes:
        hs: Hidden size.
        layers: Number of transformer layers.
        attention_heads: Number of attention heads.
        query_groups: Number of key/value heads (grouped-query attention).
        head_dim: Per-head dimension. ``None`` means ``hs // attention_heads``.
        moe_ffn_hidden_size: Hidden size of one feed-forward expert. For a dense
            model, the feed-forward hidden size.
        moe_router_topk: Experts activated per token. 1 for a dense model.
        vocab_size: Vocabulary size.
    """

    kind: Literal["qwen3"] = "qwen3"
    hs: int
    layers: int
    attention_heads: int
    query_groups: int
    head_dim: int | None = None
    moe_ffn_hidden_size: int
    moe_router_topk: int
    vocab_size: int

    def flops(self, batch_size: int, seq_len: int) -> float:
        """Return the FLOPs of one training step.

        Args:
            batch_size: Number of sequences in the step.
            seq_len: Padded sequence length.

        Returns:
            Forward plus backward FLOPs.
        """
        causal_self_attn = True
        gated_linear_multiplier = 2

        # head_dim defaults to hidden_size / num_heads, but Qwen3 sets it explicitly and for some
        # models (e.g. Qwen3-235B-A22B: head_dim=128, num_heads=64, hidden=4096) num_heads * head_dim
        # != hidden_size ("wide" attention). The QKV/output projections and the O(seq^2) attention
        # scores scale with num_heads * head_dim, NOT hidden_size, so use the real projection sizes.
        # This reduces exactly to the previous formula when num_heads * head_dim == hidden_size.
        head_dim = (
            self.head_dim
            if self.head_dim is not None
            else self.hs // self.attention_heads
        )
        q_proj_size = self.attention_heads * head_dim
        kv_proj_size = self.query_groups * head_dim

        # attention flops for GQA
        attention_flops = (
            3
            * 2
            * batch_size
            * self.layers
            * seq_len
            * (
                q_proj_size * self.hs  # Q projection
                + 2 * kv_proj_size * self.hs  # K, V projections (GQA)
                + q_proj_size * self.hs  # output projection
                + seq_len
                * q_proj_size
                * (2 * (0.5 if causal_self_attn else 1))  # QK^T + A.V scores
            )
        )

        # mlp flops
        mlp_flops = (
            3
            * 2
            * batch_size
            * self.layers
            * seq_len
            * self.hs
            * (1 + gated_linear_multiplier)
            * (self.moe_ffn_hidden_size * self.moe_router_topk)  # MoE layers
        )

        # vocab flops
        vocab_flops = 3 * 2 * batch_size * seq_len * self.hs * self.vocab_size

        return attention_flops + mlp_flops + vocab_flops


class MlaAttention(_FlopsBase):
    """Projection sizes of multi-head latent attention (MLA).

    Attributes:
        attention_heads: Number of attention heads.
        q_lora_rank: Rank of the low-rank query projection. ``None`` means the query
            projection is full rank.
        kv_lora_rank: Rank of the low-rank key/value projection.
        qk_head_dim: Per-head query/key dimension without positional embedding.
        qk_pos_emb_head_dim: Per-head query/key dimension of the positional embedding.
        v_head_dim: Per-head value dimension.
    """

    attention_heads: int
    q_lora_rank: int | None
    kv_lora_rank: int
    qk_head_dim: int
    qk_pos_emb_head_dim: int
    v_head_dim: int

    def projection_params(self, hs: int) -> int:
        """Return the per-layer MLA projection parameter count.

        Args:
            hs: Hidden size of the model.

        Returns:
            Number of projection parameters in one layer.
        """
        if self.q_lora_rank is not None:
            per_layer_mla_params = hs * self.q_lora_rank + self.q_lora_rank * (
                (self.qk_head_dim + self.qk_pos_emb_head_dim) * self.attention_heads
            )
        else:
            per_layer_mla_params = hs * (
                (self.qk_head_dim + self.qk_pos_emb_head_dim) * self.attention_heads
            )

        per_layer_mla_params += hs * self.qk_pos_emb_head_dim
        per_layer_mla_params += hs * self.kv_lora_rank + self.kv_lora_rank * (
            (self.qk_head_dim + self.v_head_dim) * self.attention_heads
        )
        per_layer_mla_params += self.v_head_dim * self.attention_heads * hs
        return per_layer_mla_params


class MoeFfn(_FlopsBase):
    """Feed-forward sizes of a model that mixes dense and mixture-of-experts layers.

    Attributes:
        ffn_hs: Feed-forward hidden size of a dense layer.
        moe_layer_pattern: One entry per layer: 0 for a dense feed-forward, 1 for a
            mixture-of-experts feed-forward.
        moe_shared_expert_intermediate_size: Total hidden size of the shared experts,
            which every token passes through.
        moe_ffn_hidden_size: Hidden size of one routed expert.
        moe_router_topk: Routed experts activated per token.
    """

    ffn_hs: int
    moe_layer_pattern: tuple[int, ...]
    moe_shared_expert_intermediate_size: int
    moe_ffn_hidden_size: int
    moe_router_topk: int

    def activated_params(self, hs: int, mtp_num_layers: int = 0) -> int:
        """Return activated dense/MoE FFN parameters for all model and MTP layers.

        Args:
            hs: Hidden size of the model.
            mtp_num_layers: Number of multi-token-prediction layers. Each one has a
                mixture-of-experts feed-forward.

        Returns:
            Number of feed-forward parameters a token passes through.
        """
        dense_layer_ffn_params = hs * self.ffn_hs * 3  # gated linear unit
        per_shared_expert_params = hs * self.moe_shared_expert_intermediate_size * 3
        per_selected_expert_params = hs * self.moe_ffn_hidden_size * 3
        moe_layer_ffn_params = per_shared_expert_params + (
            per_selected_expert_params * self.moe_router_topk
        )
        ffn_params = 0
        for i in self.moe_layer_pattern:
            if i == 0:
                ffn_params += dense_layer_ffn_params
            else:
                ffn_params += moe_layer_ffn_params
        ffn_params += mtp_num_layers * moe_layer_ffn_params
        return ffn_params


class _MlaMoeFlops(_FlopsBase):
    """Fields and shared terms of the models that use MLA and mixture-of-experts layers.

    Attributes:
        hs: Hidden size.
        layers: Number of transformer layers.
        vocab_size: Vocabulary size.
        mla: Attention projection sizes.
        ffn: Feed-forward sizes.
    """

    hs: int
    layers: int
    vocab_size: int
    mla: MlaAttention
    ffn: MoeFfn

    def _linear_flops(self, seq_len: int, mtp_num_layers: int = 0) -> int:
        """Return MLA projection and activated FFN training FLOPs per input."""
        mla_layers = self.layers + mtp_num_layers
        mla_params = self.mla.projection_params(self.hs) * mla_layers
        return (
            6
            * (mla_params + self.ffn.activated_params(self.hs, mtp_num_layers))
            * seq_len
        )

    def _vocab_and_mtp_flops(self, seq_len: int, mtp_num_layers: int = 0) -> int:
        """Return output projection and MTP embedding projection FLOPs per input."""
        vocab_flops = 6 * self.vocab_size * self.hs * seq_len
        for _ in range(mtp_num_layers):
            vocab_flops += 6 * self.vocab_size * self.hs * seq_len
            vocab_flops += 6 * self.hs * 2 * self.hs * seq_len
        return vocab_flops


class DeepSeekV3Flops(_MlaMoeFlops):
    """Model FLOPs for DeepSeek V3.

    Attributes:
        mtp_num_layers: Number of multi-token-prediction layers that are trained.
    """

    kind: Literal["deepseek_v3"] = "deepseek_v3"
    mtp_num_layers: int = 0

    def flops(self, batch_size: int, seq_len: int) -> float:
        """Return the FLOPs of one training step.

        Args:
            batch_size: Number of sequences in the step.
            seq_len: Padded sequence length.

        Returns:
            Forward plus backward FLOPs.
        """
        bmm1_flops = (
            0.5
            * (self.mla.qk_head_dim + self.mla.qk_pos_emb_head_dim)
            * self.mla.attention_heads
            * (seq_len**2)
        )
        bmm2_flops = 0.5 * self.mla.v_head_dim * self.mla.attention_heads * (seq_len**2)
        per_input_attention_flops = 6 * (bmm1_flops + bmm2_flops) * self.layers
        per_input_attention_flops += 6 * (bmm1_flops + bmm2_flops) * self.mtp_num_layers

        return (
            per_input_attention_flops
            + self._linear_flops(seq_len, self.mtp_num_layers)
            + self._vocab_and_mtp_flops(seq_len, self.mtp_num_layers)
        ) * batch_size


def _causal_topk_pairs(seq_len: int, topk: int) -> float:
    """Approximate causal sparse-attention query/key pairs with dense formula conventions."""
    effective_topk = min(topk, seq_len)
    if effective_topk == seq_len:
        return 0.5 * seq_len * seq_len
    return effective_topk * seq_len - 0.5 * effective_topk**2


class GlmMoeDsaFlops(_MlaMoeFlops):
    """Model FLOPs for GLM MoE DSA.

    GLM-MoE-DSA uses MLA projections and MoE FFNs like DeepSeek-style MoE models,
    but its core attention path uses a DSA indexer to select sparse top-k keys.
    The indexer is forward-only in the RL runs where indexer loss is disabled, so
    its matmuls are counted with a single 2x FMA factor rather than the 6x
    forward+backward training factor used for trainable model-path GEMMs.

    Multi-token-prediction layers are not counted: the MCore Bridge path used by
    these runs disables them.

    Attributes:
        dsa_indexer_n_heads: Number of heads in the DSA indexer.
        dsa_indexer_head_dim: Per-head dimension of the DSA indexer.
        dsa_indexer_topk: Keys the indexer selects per query.
        dsa_indexer_compute_layers: Number of layers that compute DSA indices
            instead of reusing those of an earlier layer.
    """

    kind: Literal["glm_moe_dsa"] = "glm_moe_dsa"
    dsa_indexer_n_heads: int
    dsa_indexer_head_dim: int
    dsa_indexer_topk: int
    dsa_indexer_compute_layers: int

    def flops(self, batch_size: int, seq_len: int) -> float:
        """Return the FLOPs of one training step.

        Args:
            batch_size: Number of sequences in the step.
            seq_len: Padded sequence length.

        Returns:
            Forward plus backward FLOPs.

        Raises:
            ValueError: If ``mla.q_lora_rank`` is unset. The DSA indexer projects
                from the low-rank query, so the formula needs its rank.
        """
        q_lora_rank = self.mla.q_lora_rank
        if q_lora_rank is None:
            raise ValueError("GlmMoeDsaFlops needs mla.q_lora_rank to be set")

        # DSA sparse attention. In absorbed MLA DSA, scores use latent KV + RoPE
        # channels and value aggregation uses latent KV channels.
        sparse_pairs = _causal_topk_pairs(seq_len, self.dsa_indexer_topk)
        sparse_score_dim = self.mla.kv_lora_rank + self.mla.qk_pos_emb_head_dim
        sparse_value_dim = self.mla.kv_lora_rank
        per_input_sparse_attention_flops = (
            6
            * self.layers
            * sparse_pairs
            * self.mla.attention_heads
            * (sparse_score_dim + sparse_value_dim)
        )

        # DSA indexer: q_lora -> index heads, hidden -> index key, hidden -> head weights,
        # then dense index-score matmul before top-k selection. Top-k itself is not counted.
        indexer_projection_params = (
            q_lora_rank * self.dsa_indexer_n_heads * self.dsa_indexer_head_dim
            + self.hs * self.dsa_indexer_head_dim
            + self.hs * self.dsa_indexer_n_heads
        )
        # The indexer scores every causally valid query/key pair before selecting top-k.
        dense_causal_pairs = seq_len * (seq_len + 1) // 2
        # This formula assumes dsa_indexer_loss_coeff=0.0, for which MCore runs the
        # indexer under no_grad. Count only its forward matmuls (2 FLOPs per FMA);
        # enabling indexer loss also requires accounting for the backward pass.
        per_input_indexer_flops = (
            2
            * self.dsa_indexer_compute_layers
            * (
                seq_len * indexer_projection_params
                + dense_causal_pairs
                * self.dsa_indexer_n_heads
                * self.dsa_indexer_head_dim
            )
        )

        return (
            self._linear_flops(seq_len)
            + per_input_sparse_attention_flops
            + per_input_indexer_flops
            + self._vocab_and_mtp_flops(seq_len)
        ) * batch_size


# Discriminated on ``kind`` so that each model family carries only the
# hyperparameters its formula reads, and a missing one fails at construction.
FlopsModel = Annotated[
    Union[
        LlamaFlops,
        Qwen2Flops,
        Qwen3Flops,
        DeepSeekV3Flops,
        GlmMoeDsaFlops,
    ],
    Field(discriminator="kind"),
]
