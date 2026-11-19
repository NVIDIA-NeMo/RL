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
"""Malformed-think credit: an advantage cap on the turns whose think tags break the format.

``<algo>.malformed_think_credit`` (``grpo`` or ``ppo``): after the advantage stage has its advantages, every trained
token of a turn whose think tags break the format (experience/think_tags.VIOLATIONS: thinking closed twice, an
open tag in the generation, a turn that ended without closing thinking, ...) gets ``min(A, max_advantage)``: at
most ``max_advantage`` however its rollout scored, and never less negative than the rest of a rollout that failed in
a group that mostly passed. The rest of the rollout keeps the advantage of its own reward. A turn cut by a length limit before it
closed thinking is not marked.

This is the turn-level counterpart of ``reward_penalties.penalize_malformed_think_tag``, which zeroes the reward of
the whole rollout: that changes nothing for a rollout that already failed, and spreads the blame for one turn over
all of a successful one.

Only advantages change: the masks are untouched (the tokens stay in the policy loss's token normaliser) and PPO's
returns are untouched (the critic's targets do not change). The turns are found in the advantage stage from the
rows' ``input_ids`` / ``token_mask`` (``MalformedThinkTurnDetector``), so groups restored from a replay checkpoint
are treated like new ones. Only tag token ids are counted: tag text spelled with ordinary tokens is the model
writing about the tags (a file it is repairing, for instance), not a format error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from pydantic import BaseModel

from nemo_rl.experience.think_tags import (
    CUT_UNCLOSED,
    VIOLATIONS,
    ThinkTagIds,
    malformed_think_turns,
    resolve_think_tag_ids,
)


class MalformedThinkCreditConfig(BaseModel, extra="forbid"):
    enabled: bool = False
    # The highest advantage a trained token of a malformed turn keeps: min(A, max_advantage).
    max_advantage: float = -2.0


def malformed_think_credit_config(
    algo_cfg: object,
) -> Optional[MalformedThinkCreditConfig]:
    """The algorithm block's malformed_think_credit when enabled, else None (dicts from ``++`` overrides are parsed)."""
    raw = getattr(algo_cfg, "malformed_think_credit", None)
    if raw is None:
        return None
    cfg = (
        raw
        if isinstance(raw, MalformedThinkCreditConfig)
        else MalformedThinkCreditConfig.model_validate(dict(raw))
    )
    return cfg if cfg.enabled else None


@dataclass
class MalformedThinkCreditStats:
    """Additive per-step sums; ``to_metrics`` turns them into shares."""

    trained_tokens: float = 0.0
    # Trained tokens in malformed turns, those of them whose computed advantage was positive, and those lowered to
    # max_advantage.
    flagged_tokens: float = 0.0
    flagged_positive_tokens: float = 0.0
    lowered_tokens: float = 0.0
    trained_rows: int = 0
    flagged_rows: int = 0
    # Detection over the scanned rows: turns, malformed turns, and turns per kind (experience/think_tags.VIOLATIONS
    # and CUT_UNCLOSED, which is counted but not marked).
    turns: int = 0
    malformed_turns: int = 0
    kinds: dict[str, int] = field(default_factory=dict)

    def merge(self, other: "MalformedThinkCreditStats") -> None:
        self.trained_tokens += other.trained_tokens
        self.flagged_tokens += other.flagged_tokens
        self.flagged_positive_tokens += other.flagged_positive_tokens
        self.lowered_tokens += other.lowered_tokens
        self.trained_rows += other.trained_rows
        self.flagged_rows += other.flagged_rows
        self.turns += other.turns
        self.malformed_turns += other.malformed_turns
        for kind, count in other.kinds.items():
            self.kinds[kind] = self.kinds.get(kind, 0) + count

    def to_metrics(self) -> dict[str, float]:
        if not self.trained_tokens:
            return {}
        return {
            "malformed_think_credit/flagged_token_share": self.flagged_tokens
            / self.trained_tokens,
            "malformed_think_credit/flagged_positive_token_share": self.flagged_positive_tokens
            / self.trained_tokens,
            "malformed_think_credit/lowered_token_share": self.lowered_tokens
            / self.trained_tokens,
            "malformed_think_credit/flagged_row_share": (
                self.flagged_rows / self.trained_rows if self.trained_rows else 0.0
            ),
            "malformed_think_credit/malformed_turn_share": (
                self.malformed_turns / self.turns if self.turns else 0.0
            ),
            **{
                f"malformed_think_credit/turn_share/{kind}": (
                    self.kinds.get(kind, 0) / self.turns if self.turns else 0.0
                )
                for kind in (*VIOLATIONS, CUT_UNCLOSED)
            },
        }


class MalformedThinkTurnDetector:
    """Malformed-turn masks for advantage-stage batches, from their token rows.

    ``load`` (the tokenizer and the tag check) runs at controller setup, so a tokenizer without single-token think
    tags fails the run at startup rather than at the first advantage stage.
    """

    def __init__(self, tokenizer_config: dict[str, Any]) -> None:
        self._tokenizer_config = tokenizer_config
        self._ids: Optional[ThinkTagIds] = None

    def load(self) -> None:
        from nemo_rl.algorithms.utils import get_tokenizer

        self._ids = resolve_think_tag_ids(get_tokenizer(self._tokenizer_config))

    def mask(
        self,
        input_ids: torch.Tensor,
        token_mask: torch.Tensor,
        scan_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, MalformedThinkCreditStats]:
        """``(B, T)`` bool: the generated tokens of malformed turns; plus the detection counts.

        Args:
            input_ids: ``(B, ...)`` the rows' token ids, jagged (nested) or right-padded.
            token_mask: ``(B, T)`` the right-padded generated-token mask the advantages are aligned with.
            scan_rows: ``(B,)`` bool, the rows to scan (rows without trained tokens are left unmarked).
        """
        assert self._ids is not None, "MalformedThinkTurnDetector.load() was not called"
        rows = input_ids.unbind() if input_ids.is_nested else input_ids
        generated = token_mask.detach().cpu()
        width = int(generated.shape[-1])
        malformed = torch.zeros(generated.shape, dtype=torch.bool)
        stats = MalformedThinkCreditStats()
        for i in torch.nonzero(scan_rows.cpu(), as_tuple=False).reshape(-1).tolist():
            token_ids = rows[i].detach().cpu().reshape(-1)
            length = int(token_ids.shape[0])
            assert length <= width, (
                f"row {i}: {length} input ids but a token_mask of width {width}"
            )
            found = malformed_think_turns(
                token_ids[:length].numpy(), generated[i, :length].numpy(), ids=self._ids
            )
            malformed[i, :length] = torch.from_numpy(found.mask)
            stats.turns += found.turns
            stats.malformed_turns += found.malformed_turns
            for kind, count in found.kinds.items():
                stats.kinds[kind] = stats.kinds.get(kind, 0) + count
        return malformed, stats


def cap_advantage_on_malformed_turns(
    advantages: torch.Tensor,
    *,
    malformed_turn_mask: torch.Tensor,
    train_mask: torch.Tensor,
    max_advantage: float,
) -> tuple[torch.Tensor, MalformedThinkCreditStats]:
    """Lower the advantages of the trained tokens of malformed turns to at most ``max_advantage``.

    Args:
        advantages: ``(B, T)`` the advantage stage's advantages.
        malformed_turn_mask: ``(B, T)`` True on the generated tokens of malformed turns.
        train_mask: ``(B, T)`` > 0 on trained tokens.
        max_advantage: the highest advantage a flagged token keeps.

    Returns:
        The new advantages (same shape, dtype and device) and the step statistics.
    """
    if (
        malformed_turn_mask.shape != advantages.shape
        or train_mask.shape != advantages.shape
    ):
        raise ValueError(
            "malformed_think_credit needs advantages, malformed_turn_mask and train_mask of one shape, got "
            f"{tuple(advantages.shape)}, {tuple(malformed_turn_mask.shape)}, {tuple(train_mask.shape)}"
        )
    trained = train_mask.to(device=advantages.device) > 0
    flagged = malformed_turn_mask.to(device=advantages.device).bool() & trained
    lowered = flagged & (advantages > max_advantage)
    stats = MalformedThinkCreditStats(
        trained_tokens=float(trained.sum()),
        flagged_tokens=float(flagged.sum()),
        flagged_positive_tokens=float((flagged & (advantages > 0)).sum()),
        lowered_tokens=float(lowered.sum()),
        trained_rows=int(trained.any(dim=-1).sum()),
        flagged_rows=int(flagged.any(dim=-1).sum()),
    )
    return torch.where(
        lowered, torch.full_like(advantages, max_advantage), advantages
    ), stats
