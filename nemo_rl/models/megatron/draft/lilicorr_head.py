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
"""LiLiCorr lattice reranker, native to NeMo-RL.

Ported from ModelOpt's ``modelopt.torch.speculative.plugins.modeling_lilicorr`` so the
Megatron draft path owns one definition rather than importing an HF-transformers module
into a Megatron provider. The other Megatron drafters here are either native (DFlash,
DSpark) or import ModelOpt's *Megatron* plugin (Eagle3 uses ``megatron_eagle``); ModelOpt
ships no ``megatron_lilicorr``, so importing its HF module was a pattern found nowhere
else in this tree.

The head is REPLICATED, not tensor-parallel: it is plain ``nn.Linear`` throughout, and
every TP rank holds an identical copy. Being native buys a single owner of the parameter
layout and no cross-repo version skew -- it does not make the head sharded.

The definitions below are a faithful copy, deliberately not a reimplementation. Parameter
names and module structure are the contract the released checkpoints are written against,
and an earlier hand-written version here was wrong in ways the names concealed:
``feature_norm`` is a ``[5]`` learned weighting across the five captured target layers,
not a projection of the 4096-dim hidden, and the factor heads emit ``factor_dim`` vectors
for cosine scoring rather than scalars.
``tests/unit/models/megatron/draft/test_lilicorr_head_equivalence.py`` pins this against
ModelOpt's head numerically, so drift is caught rather than argued about.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _LiLiCorrRMSNorm(nn.Module):
    """RMSNorm matching ``transformers`` Qwen3RMSNorm exactly.

    Same formula, same fp32 variance accumulation, same epsilon placement and the same
    ``weight`` parameter name, so checkpoints are interchangeable. Local rather than
    imported to keep the head free of a transformers dependency.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


_NORM_CLS = _LiLiCorrRMSNorm


_NUM_CANDIDATE_FEATURES = 5


class LiLiCorrLayer(nn.Module):
    """One bidirectional attention layer over the candidate lattice.

    Attention is unmasked across the whole ``slots x k`` lattice — a candidate may
    look at any other, in either direction — because the lattice is a scoring
    structure, not a sequence being generated. Position enters only through the
    slot/rank embeddings and the relative-slot bias.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float, rms_norm_eps: float
    ):
        """Build the pre-norm attention and MLP sublayers."""
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"LiLiCorr hidden_size={hidden_size} must be divisible by num_heads={num_heads}."
            )
        self.attn_norm = _NORM_CLS(hidden_size, eps=rms_norm_eps)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=0.0
        )
        self.mlp_norm = _NORM_CLS(hidden_size, eps=rms_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_bias: torch.Tensor
    ) -> torch.Tensor:
        """Pre-norm attention with an additive bias, then the MLP."""
        attn_input = self.attn_norm(hidden_states)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=attention_bias,
            need_weights=False,
        )
        hidden_states = hidden_states + attn_output
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class LiLiCorrHead(nn.Module):
    """Anchor-conditioned chain scorer over the drafter's per-slot top-k candidates.

    Produces the two factors a chain needs: a ``start`` score for slot 0 against
    the anchor, and a ``pair`` score for every transition between adjacent slots.
    Both are cosine similarities scaled by ``logit_scale``; the wrapper composes
    them into log-potentials via :meth:`compose_log_factors`.

    Every geometry and scaling argument is required. ``candidate_topk`` sets the
    lattice width and the shape of ``rank_embedding``, while ``logit_scale`` and
    ``vector_eps`` change the score without changing any shape — so a defaulted
    value would build a head that loads cleanly and scores a different function.
    """

    def __init__(
        self,
        *,
        model_hidden_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        block_size: int,
        candidate_topk: int,
        factor_dim: int,
        rms_norm_eps: float,
        vector_eps: float,
        logit_scale: float,
    ):
        """Validate the geometry, then build the lattice tower and the two factor heads."""
        super().__init__()
        if block_size < 2:
            raise ValueError(f"LiLiCorr block_size must be >= 2, got {block_size}.")
        if candidate_topk < 1:
            raise ValueError(
                f"LiLiCorr candidate_topk must be >= 1, got {candidate_topk}."
            )
        if num_layers <= 0:
            raise ValueError(f"LiLiCorr num_layers must be positive, got {num_layers}.")
        if factor_dim <= 0:
            raise ValueError(f"LiLiCorr factor_dim must be positive, got {factor_dim}.")
        if not 0.0 < vector_eps < 0.5:
            raise ValueError(
                f"LiLiCorr vector_eps must be in (0, 0.5), got {vector_eps}."
            )
        if logit_scale <= 0.0:
            raise ValueError(
                f"LiLiCorr logit_scale must be positive, got {logit_scale}."
            )

        self.block_size = int(block_size)
        self.num_candidate_slots = self.block_size - 1
        self.candidate_topk = int(candidate_topk)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.factor_dim = int(factor_dim)
        self.vector_eps = float(vector_eps)
        self.logit_scale = float(logit_scale)

        # Identity when the head is as wide as the draft, so no redundant matmul.
        self.token_proj = (
            nn.Identity()
            if model_hidden_size == hidden_size
            else nn.Linear(model_hidden_size, hidden_size)
        )
        self.pass_hidden_proj = nn.Linear(model_hidden_size, hidden_size)
        self.feature_mlp = nn.Sequential(
            nn.LayerNorm(_NUM_CANDIDATE_FEATURES),
            nn.Linear(_NUM_CANDIDATE_FEATURES, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.slot_embedding = nn.Parameter(
            torch.zeros(1, 1, self.num_candidate_slots, 1, hidden_size)
        )
        # Rank-ordered, so its first `topk` rows are exactly the rows that apply to
        # ranks 0..topk-1. That is what lets a head trained at K serve any k <= K.
        self.rank_embedding = nn.Parameter(
            torch.zeros(1, 1, 1, self.candidate_topk, hidden_size)
        )
        self.relative_slot_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * self.block_size - 1)
        )
        self.same_slot_bias = nn.Parameter(torch.zeros(num_heads))
        # The anchor is a row of the target hidden state, hence model_hidden_size
        # wide. One row has no sequence to position, so it carries no position
        # embedding and does not join the lattice attention: it feeds the factor
        # heads only.
        self.context_proj = nn.Linear(model_hidden_size, hidden_size)

        self.layers = nn.ModuleList(
            [
                LiLiCorrLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = _NORM_CLS(hidden_size, eps=rms_norm_eps)
        self.anchor_norm = _NORM_CLS(hidden_size, eps=rms_norm_eps)
        # The factor heads read [self, anchor, self * anchor].
        self.factor_input_proj = nn.Linear(hidden_size * 3, hidden_size)
        # Named by edge direction: a transition runs OUT of the earlier candidate
        # and IN to the later one, so pair = out[s] . in[s+1] and start = anchor_out . in[0].
        self.out_head = nn.Linear(hidden_size, factor_dim)
        self.in_head = nn.Linear(hidden_size, factor_dim)
        self.anchor_out_head = nn.Linear(hidden_size, factor_dim)
        self.reset_factor_heads()

    def reset_factor_heads(self) -> None:
        """Re-seed the three factor heads with a small non-zero std.

        Must run after any ``_init_weights`` sweep. The factors are dot products of
        the out/in embeddings, so zero-init on both sides makes the product's
        gradient vanish at init — a dead saddle. A small std keeps the initial
        scores near zero (near-uniform) while leaving gradient to break symmetry.
        """
        for head in (self.out_head, self.in_head, self.anchor_out_head):
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    def compose_log_factors(
        self, *, start_scores: torch.Tensor, pair_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn the cosine scores into the chain's log-potentials, ``logit_scale * cos``.

        ``logit_scale`` is a fixed, non-learnable constant that sets how sharp the chain
        may become, since a cosine in ``[-1, 1]`` is too flat to commit on. fp32 because
        the decode's argmax runs on these values.
        """
        return (
            self.logit_scale * start_scores.float(),
            self.logit_scale * pair_scores.float(),
        )

    def _attention_bias(
        self, batch_blocks: int, *, device: torch.device, dtype: torch.dtype, topk: int
    ) -> torch.Tensor:
        """Build the per-head additive attention bias over the lattice.

        Relative slot distance, plus a same-slot term. This is pure lattice geometry with
        no token content in it, so one bias serves every block and is simply broadcast.

        Shapes below use a running example of ``block_size`` 4 (hence 3 slots) and
        ``topk`` 2, giving ``S = slots * topk = 6`` lattice positions.
        """
        # Which slot each of the S flattened positions belongs to: [0, 0, 1, 1, 2, 2] —
        # slot 0's two candidates, then slot 1's, then slot 2's.
        slot_ids = torch.arange(
            self.num_candidate_slots, device=device, dtype=torch.long
        ).repeat_interleave(topk)
        # Signed slot distance from query i to key j, [S, S]. A slot-1 row reads
        # [1, 1, 0, 0, -1, -1]: slot 0 behind it, itself, then slot 2 ahead.
        # The clamp is defensive and never binds — slots span at most block_size-2.
        rel = (slot_ids.view(-1, 1) - slot_ids.view(1, -1)).clamp(
            min=-(self.block_size - 1), max=self.block_size - 1
        )
        # One learned scalar per (attention head, distance); num_heads is the lattice
        # attention's head count (lilicorr_num_heads), not the draft's and not the out/in
        # factor heads. A column index cannot be negative, so adding block_size-1 centres
        # the signed distance over the table's 2*block_size-1 columns, putting distance 0
        # in the middle: with the running example's offset of 3, rel -2 -> column 1,
        # 0 -> column 3, +2 -> column 5. Gives [num_heads, S, S].
        bias = self.relative_slot_bias[:, rel + self.block_size - 1]
        # The topk x topk diagonal blocks: candidates competing for the same slot. This
        # coincides with rel == 0, which relative_slot_bias already covers, but it carries
        # its own learned scalar per head.
        same_slot = slot_ids.view(-1, 1) == slot_ids.view(1, -1)
        bias = bias + same_slot.unsqueeze(0).to(
            dtype=bias.dtype
        ) * self.same_slot_bias.view(-1, 1, 1)
        bias = bias.to(device=device, dtype=dtype)
        # [batch_blocks * num_heads, S, S] is the layout nn.MultiheadAttention wants.
        return (
            bias.unsqueeze(0)
            .expand(batch_blocks, -1, -1, -1)
            .reshape(batch_blocks * self.num_heads, bias.shape[-2], bias.shape[-1])
        )

    def _project_anchor(
        self,
        *,
        anchor_hidden: torch.Tensor,
        anchor_valid: torch.Tensor | None,
        bsz: int,
        n_blocks: int,
    ) -> torch.Tensor:
        """Project one committed target row per block.

        Shape is checked rather than adapted: the anchor is the whole context the
        factor heads see, so a differently shaped one would load and score a
        different function of the same parameters.
        """
        expected = (bsz, n_blocks, 1)
        if tuple(anchor_hidden.shape[:3]) != expected:
            raise ValueError(
                f"anchor_hidden must be {expected} + [dim] — one row per block — "
                f"got {tuple(anchor_hidden.shape)}."
            )
        anchor_state = self.context_proj(
            anchor_hidden.reshape(bsz * n_blocks, anchor_hidden.shape[-1])
        )
        if anchor_valid is not None:
            # A block anchored at position 0 has no preceding committed token. Its input
            # row arrives already zeroed, but context_proj has a bias, so a zeroed input
            # still projects to that bias — zero the output to keep the anchor at zero.
            valid = anchor_valid.reshape(bsz * n_blocks).to(
                device=anchor_state.device, dtype=anchor_state.dtype
            )
            anchor_state = anchor_state * valid.unsqueeze(-1)
        return anchor_state

    def forward(
        self,
        *,
        candidate_token_ids: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        pass_hidden: torch.Tensor,
        embed_tokens: nn.Module,
        anchor_hidden: torch.Tensor,
        anchor_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score the lattice.

        Shapes below use ``B`` for the batch, ``N`` for the blocks per sequence, ``slots``
        for the ``block_size - 1`` predicted positions in a block, ``k`` for the candidates
        per slot and ``H`` for the draft's hidden size.

        Args:
            candidate_token_ids: Candidate ids ``[B, N, slots, k]``.
            candidate_log_probs: Their log-probs under the drafter ``[B, N, slots, k]``.
            pass_hidden: Draft hidden per slot ``[B, N, slots, H]``.
            embed_tokens: Embedding table for the candidate ids (target's).
            anchor_hidden: One committed target row per block ``[B, N, 1, H]``.
            anchor_valid: Whether each block's anchor is in range ``[B, N, 1]``.

        Returns:
            ``(start_scores, pair_scores)`` — ``[B, N, k]`` and
            ``[B, N, slots-1, k, k]``, both cosine similarities in ``[-1, 1]``.
        """
        if candidate_token_ids.ndim != 4:
            raise ValueError(
                "candidate_token_ids must be [batch, blocks, slots, k], got "
                f"{tuple(candidate_token_ids.shape)}."
            )
        bsz, n_blocks, n_slots, topk = candidate_token_ids.shape
        if n_slots != self.num_candidate_slots or topk > self.candidate_topk:
            raise ValueError(
                f"lattice slots={n_slots}, k={topk} is not compatible with "
                f"slots={self.num_candidate_slots}, trained k={self.candidate_topk} "
                "(need 1 <= k <= trained)."
            )
        if candidate_log_probs.shape != candidate_token_ids.shape:
            raise ValueError(
                "candidate_log_probs must match candidate_token_ids, got "
                f"{tuple(candidate_log_probs.shape)} vs {tuple(candidate_token_ids.shape)}."
            )
        if pass_hidden.shape[:3] != (bsz, n_blocks, n_slots):
            raise ValueError(
                f"pass_hidden must be [batch, blocks, slots, dim], got {tuple(pass_hidden.shape)}."
            )

        # Detached: the candidate embeddings are an input to the head, not a path
        # the head trains the target's embedding table through.
        token_states = self.token_proj(embed_tokens(candidate_token_ids).detach())
        pass_states = self.pass_hidden_proj(pass_hidden).unsqueeze(-2)

        log_probs = candidate_log_probs.float()
        probs = log_probs.exp()
        logprob_gap = log_probs - log_probs.max(dim=-1, keepdim=True).values
        # The candidate's rank as a fraction, top-1 at 0 and last at 1: [0, 1/3, 2/3, 1]
        # at topk 4. Identical for every slot and block, rank being a position within a
        # slot's own candidate list. At topk 1 there is no range to normalize over.
        if topk == 1:
            rank_frac = torch.zeros_like(log_probs)
        else:
            rank_frac = (
                torch.arange(topk, device=log_probs.device, dtype=log_probs.dtype).view(
                    1, 1, 1, topk
                )
                / float(topk - 1)
            ).expand_as(log_probs)
        is_top1 = torch.zeros_like(log_probs)
        is_top1[..., 0] = 1.0
        features = torch.stack(
            [log_probs, probs, logprob_gap, rank_frac, is_top1], dim=-1
        )

        hidden_states = token_states + pass_states
        hidden_states = hidden_states + self.feature_mlp(
            features.to(dtype=token_states.dtype)
        )
        hidden_states = hidden_states + self.slot_embedding.to(
            device=token_states.device, dtype=token_states.dtype
        )
        hidden_states = hidden_states + self.rank_embedding[:, :, :, :topk, :].to(
            device=token_states.device, dtype=token_states.dtype
        )
        hidden_states = hidden_states.reshape(
            bsz * n_blocks, n_slots * topk, token_states.shape[-1]
        )

        anchor_state = self._project_anchor(
            anchor_hidden=anchor_hidden,
            anchor_valid=anchor_valid,
            bsz=bsz,
            n_blocks=n_blocks,
        )
        attention_bias = self._attention_bias(
            bsz * n_blocks,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            topk=topk,
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_bias=attention_bias)
        hidden_states = self.output_norm(hidden_states).reshape(
            bsz, n_blocks, n_slots, topk, -1
        )
        anchor_state = self.anchor_norm(anchor_state).reshape(bsz, n_blocks, -1)

        anchor_for_candidate = anchor_state[:, :, None, None, :].expand_as(
            hidden_states
        )
        factor_hidden = F.silu(
            self.factor_input_proj(
                torch.cat(
                    [
                        hidden_states,
                        anchor_for_candidate,
                        hidden_states * anchor_for_candidate,
                    ],
                    dim=-1,
                )
            )
        )
        # L2-normalize every bilinear operand so each score is a cosine in [-1, 1]:
        # the factor's magnitude is then set solely by the fixed logit_scale, and
        # direction carries all the learned signal. Bounding the direction rather
        # than squashing the raw score keeps the gradient largest at the
        # near-uniform operating point instead of saturating there.
        out_vec = F.normalize(self.out_head(factor_hidden), dim=-1, eps=self.vector_eps)
        in_vec = F.normalize(self.in_head(factor_hidden), dim=-1, eps=self.vector_eps)
        anchor_out = F.normalize(
            self.anchor_out_head(anchor_state), dim=-1, eps=self.vector_eps
        )

        start_scores = (anchor_out[:, :, None, :] * in_vec[:, :, 0, :, :]).sum(dim=-1)
        pair_scores = (
            out_vec[:, :, :-1, :, None, :] * in_vec[:, :, 1:, None, :, :]
        ).sum(dim=-1)
        return start_scores, pair_scores
