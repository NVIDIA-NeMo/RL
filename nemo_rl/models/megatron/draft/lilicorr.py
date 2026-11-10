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

"""LiLiCorr reranker head for Megatron draft co-training.

LiLiCorr is a DFlash backbone plus a candidate-lattice reranker: the backbone proposes a
block of tokens, and the reranker rescores a small candidate set per slot before the
target verifies. That is the same relationship DSpark has to DFlash, so this module holds
only the head and reuses :class:`~nemo_rl.models.megatron.draft.dflash.DFlashBody` for the
backbone rather than reimplementing a tensor-parallel transformer.

Geometry is read from the draft checkpoint's ``config.json`` (``lilicorr_hidden_size``,
``lilicorr_num_layers``, ``lilicorr_num_heads``, ``lilicorr_mlp_ratio``,
``lilicorr_candidate_topk``) rather than configured here: a mismatch against trained
weights loses accuracy silently instead of raising.

Weight names follow the serving layout that NVIDIA's released drafters use
(``lilicorr.feature_mlp.up_proj``), not the ``nn.Sequential`` positional names
(``lilicorr.feature_mlp.1``) that ModelOpt's exporter emitted before
NVIDIA/Model-Optimizer#3061.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class LiLiCorrHeadConfig:
    """Every constructor argument ModelOpt's LiLiCorrHead takes, read from a checkpoint.

    Each field maps to a key the released drafters carry in ``dflash_config``. None of it
    is defaulted: a reranker built with guessed geometry loads cleanly and scores a
    different function, surfacing much later as a lower acceptance length rather than as
    an error.
    """

    model_hidden_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    mlp_ratio: float
    block_size: int
    candidate_topk: int
    factor_dim: int
    rms_norm_eps: float
    vector_eps: float
    logit_scale: float

    _KEYS = {
        "hidden_size": "lilicorr_hidden_size",
        "num_layers": "lilicorr_num_layers",
        "num_heads": "lilicorr_num_heads",
        "mlp_ratio": "lilicorr_mlp_ratio",
        "candidate_topk": "lilicorr_candidate_topk",
        "factor_dim": "lilicorr_factor_dim",
        "vector_eps": "lilicorr_vector_eps",
        "logit_scale": "lilicorr_logit_scale",
    }

    @classmethod
    def from_checkpoint_config(cls, cfg: dict) -> "LiLiCorrHeadConfig":
        """Build from a drafter ``config.json``."""
        d = cfg.get("dflash_config", cfg)
        missing = [src for src in cls._KEYS.values() if src not in d]
        if "block_size" not in d and "block_size" not in cfg:
            missing.append("block_size")
        if missing:
            raise ValueError(
                f"draft checkpoint config is missing LiLiCorr geometry {sorted(missing)}; "
                "refusing to default it, since a guessed reranker scores a different "
                "function without erroring"
            )
        kwargs = {dst: d[src] for dst, src in cls._KEYS.items()}
        return cls(
            model_hidden_size=int(cfg.get("hidden_size", 4096)),
            block_size=int(d.get("block_size", cfg.get("block_size"))),
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            hidden_size=int(kwargs["hidden_size"]),
            num_layers=int(kwargs["num_layers"]),
            num_heads=int(kwargs["num_heads"]),
            mlp_ratio=float(kwargs["mlp_ratio"]),
            candidate_topk=int(kwargs["candidate_topk"]),
            factor_dim=int(kwargs["factor_dim"]),
            vector_eps=float(kwargs["vector_eps"]),
            logit_scale=float(kwargs["logit_scale"]),
        )


def build_lilicorr_head(config: LiLiCorrHeadConfig) -> nn.Module:
    """Instantiate the native LiLiCorr reranker head.

    Native to NeMo-RL rather than imported from ModelOpt. ModelOpt ships Megatron
    plugins for eagle and medusa but none for lilicorr, so importing its HF module here
    was a pattern used nowhere else in this tree -- DFlash and DSpark define their heads
    natively, Eagle3 imports ModelOpt's *Megatron* plugin.

    The port is a faithful copy, and
    ``tests/unit/models/megatron/draft/test_lilicorr_head_equivalence.py`` pins it
    numerically against ModelOpt's head so divergence surfaces as a failing test rather
    than as a drafter that loads and serves noise.
    """
    from nemo_rl.models.megatron.draft.lilicorr_head import LiLiCorrHead

    return LiLiCorrHead(
        model_hidden_size=config.model_hidden_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        block_size=config.block_size,
        candidate_topk=config.candidate_topk,
        factor_dim=config.factor_dim,
        rms_norm_eps=config.rms_norm_eps,
        vector_eps=config.vector_eps,
        logit_scale=config.logit_scale,
    )


def lilicorr_lattice_nll(
    head: nn.Module,
    start_scores: Tensor,
    pair_scores: Tensor,
    gold_index: Tensor,
) -> Tensor:
    """Negative log-likelihood of the target path through the candidate lattice.

    LiLiCorr does not score slots independently -- that was a wrong guess in an earlier
    version here. It emits ``start_scores`` ``[B, N, k]`` and ``pair_scores``
    ``[B, N, slots-1, k, k]``, cosine similarities forming a lattice, and the quantity
    being trained is the score of the *path* the target actually took.

    ``head.compose_log_factors`` turns the raw cosines into log factors with the head's own
    normalization, so the temperature and epsilon the checkpoint was trained with are
    honoured rather than reinvented.

    Blocks whose target was not fully proposed carry ``-100`` somewhere along the path and
    contribute nothing: a partial path has no well-defined gold transition, and scoring it
    against an arbitrary candidate would train the lattice toward a wrong answer exactly
    where the backbone already failed.
    """
    log_start, log_pair = head.compose_log_factors(
        start_scores=start_scores, pair_scores=pair_scores
    )

    # A block counts only if every slot's target was proposed.
    complete = gold_index.ne(-100).all(dim=-1)
    if not bool(complete.any()):
        return (log_start.sum() + log_pair.sum()) * 0.0

    safe = gold_index.clamp_min(0)
    b, n, slots = safe.shape

    # log P(path) = log_start[gold_0] + sum_i log_pair[i, gold_i, gold_{i+1}],
    # normalized by the lattice's own partition function so this is a likelihood rather
    # than an unbounded score.
    path = log_start.gather(-1, safe[..., :1]).squeeze(-1)
    for i in range(slots - 1):
        step = log_pair[:, :, i]  # [B, N, k, k]
        frm = safe[..., i].view(b, n, 1, 1).expand(b, n, 1, step.size(-1))
        row = step.gather(2, frm).squeeze(2)  # [B, N, k]
        path = path + row.gather(-1, safe[..., i + 1 : i + 2]).squeeze(-1)

    # Partition: forward algorithm over the same factors.
    alpha = log_start
    for i in range(slots - 1):
        alpha = torch.logsumexp(alpha.unsqueeze(-1) + log_pair[:, :, i], dim=-2)
    log_z = torch.logsumexp(alpha, dim=-1)

    nll = (log_z - path)[complete]
    return nll.mean()


@dataclass
class LiLiCorrDraftOutput:
    """What a LiLiCorr draft forward must hand the loss.

    The backbone and the reranker are trained by different terms over different shapes, so
    both have to survive the forward rather than one replacing the other:

    * ``student_logits`` -- ``[B, S, V]`` next-token logits from the DFlash backbone, which
      the existing :class:`DraftCrossEntropyLossFn` consumes against the teacher.
    * ``reranker_logits`` -- ``[B, S, K]`` per-candidate scores.
    * ``gold_index`` -- ``[B, S]`` index of the target's token within each slot's
      candidates, ``-100`` where it is absent.
    """

    student_logits: Tensor
    start_scores: Tensor
    pair_scores: Tensor
    candidate_ids: Tensor

    def gold_index(self, target_ids: Tensor) -> Tensor:
        """Resolve the target's token within each slot's candidates."""
        return gold_index_from_candidates(self.candidate_ids, target_ids)


def gold_index_from_candidates(candidate_ids: Tensor, target_ids: Tensor) -> Tensor:
    """Locate the target's token among each slot's candidates.

    Returns ``-100`` where the target token was not proposed. That is a normal outcome --
    the backbone's topk missed -- and those slots must contribute nothing rather than be
    scored against an arbitrary candidate, which would train the reranker toward a wrong
    answer on exactly the positions it already failed.
    """
    match = candidate_ids.eq(target_ids.unsqueeze(-1))
    found = match.any(dim=-1)
    # argmax returns 0 for an all-false row; the `found` mask is what makes that safe.
    return torch.where(
        found, match.float().argmax(dim=-1), torch.full_like(target_ids, -100)
    )


def lilicorr_draft_loss(
    head: nn.Module,
    output: LiLiCorrDraftOutput,
    target_ids: Tensor,
    *,
    backbone_loss: Tensor,
    lattice_alpha: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Backbone draft loss plus the lattice NLL.

    ``backbone_loss`` comes from the existing draft cross-entropy path, so the DFlash
    objective is untouched and only the reranker term is added.
    """
    gold = output.gold_index(target_ids)
    lattice = lilicorr_lattice_nll(head, output.start_scores, output.pair_scores, gold)
    total = backbone_loss + lattice_alpha * lattice

    complete = gold.ne(-100).all(dim=-1)
    return total, {
        "draft_backbone_loss": float(backbone_loss.detach()),
        "draft_lattice_nll": float(lattice.detach()),
        # Share of blocks whose target path was fully proposed. If this collapses, the
        # lattice trains on almost nothing while the total loss keeps falling.
        "draft_complete_path_rate": float(complete.float().mean()),
    }
