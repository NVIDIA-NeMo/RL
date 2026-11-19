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
"""Tool-failure credit: no positive advantage on a turn whose tool call failed.

``<algo>.tool_failure_credit`` (``grpo`` or ``ppo``): after the advantage stage has its advantages, every trained
token of a turn with a failed tool call has its advantage clipped to ``min(A, 0)``. A rollout that failed a call,
recovered and earned reward 1 no longer reinforces the failed call; the turns after it keep their credit, and a
rollout with negative advantage keeps its full negative signal, failed turns included.

Only advantages change: the masks are untouched (the tokens stay in the policy loss's token normaliser) and PPO's
returns are untouched (the critic's targets do not change).

The failed turns are found in the advantage stage itself (``FailedToolTurnDetector``, running
experience/tool_failures.failed_tool_turns on the batch's ``input_ids`` / ``token_mask``) rather than stored as a data
plane column by the RolloutReassembler: the mask is a function of columns every row already carries, so groups
restored from a replay checkpoint and new groups are treated alike, and the flag can be turned on for a run that
resumes buffered rollouts. The price is detection on the trainer's path, a few ms per row (4-11 ms on 262k-token
rows: only tool-call headers and response heads are decoded).

Why: an audit of training rollouts found failed edits, reads of missing files and schema errors inside rollouts with
reward 1, trained with positive advantage like the rest of the rollout.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from pydantic import BaseModel, Field, model_validator

from nemo_rl.experience.tool_failures import (
    CATEGORIES,
    VALIDATED_CATEGORIES,
    ToolTagIds,
    failed_tool_turns,
    resolve_tool_tag_ids,
)


class ToolFailureCreditConfig(BaseModel, extra="forbid"):
    enabled: bool = False
    # Failure categories (experience/tool_failures.CATEGORIES) that mark a turn. The default is the ones validated
    # as the model's mistakes; permission_denied is mostly benign commands hitting over-broad deny rules.
    categories: list[str] = Field(default_factory=lambda: list(VALIDATED_CATEGORIES))

    @model_validator(mode="after")
    def _check(self) -> "ToolFailureCreditConfig":
        unknown = sorted(set(self.categories) - set(CATEGORIES))
        if unknown or not self.categories:
            raise ValueError(
                f"tool_failure_credit.categories must be a non-empty subset of {list(CATEGORIES)}, "
                f"got {self.categories}"
            )
        return self


def tool_failure_credit_config(algo_cfg: object) -> Optional[ToolFailureCreditConfig]:
    """The algorithm block's tool_failure_credit when enabled, else None (dicts from ``++`` overrides are parsed)."""
    raw = getattr(algo_cfg, "tool_failure_credit", None)
    if raw is None:
        return None
    cfg = (
        raw
        if isinstance(raw, ToolFailureCreditConfig)
        else ToolFailureCreditConfig.model_validate(dict(raw))
    )
    return cfg if cfg.enabled else None


@dataclass
class ToolFailureCreditStats:
    """Additive per-step sums; ``to_metrics`` turns them into shares."""

    trained_tokens: float = 0.0
    # Trained tokens in failed turns, and those of them whose advantage was positive (now 0).
    flagged_tokens: float = 0.0
    zeroed_tokens: float = 0.0
    # Sum of the positive advantages over trained tokens, and the part removed.
    positive_mass: float = 0.0
    removed_mass: float = 0.0
    trained_rows: int = 0
    flagged_rows: int = 0
    # Detection over the scanned rows: turns, turns with a failed call in an allowed category, tool responses, and
    # failed calls per category (every category, for monitoring).
    turns: int = 0
    failed_turns: int = 0
    tool_responses: int = 0
    failed_calls: dict[str, int] = field(default_factory=dict)

    def merge(self, other: "ToolFailureCreditStats") -> None:
        self.trained_tokens += other.trained_tokens
        self.flagged_tokens += other.flagged_tokens
        self.zeroed_tokens += other.zeroed_tokens
        self.positive_mass += other.positive_mass
        self.removed_mass += other.removed_mass
        self.trained_rows += other.trained_rows
        self.flagged_rows += other.flagged_rows
        self.turns += other.turns
        self.failed_turns += other.failed_turns
        self.tool_responses += other.tool_responses
        for category, count in other.failed_calls.items():
            self.failed_calls[category] = self.failed_calls.get(category, 0) + count

    def to_metrics(self) -> dict[str, float]:
        if not self.trained_tokens:
            return {}
        return {
            "tool_failure_credit/flagged_token_share": self.flagged_tokens
            / self.trained_tokens,
            "tool_failure_credit/zeroed_token_share": self.zeroed_tokens
            / self.trained_tokens,
            "tool_failure_credit/removed_positive_mass_share": (
                self.removed_mass / self.positive_mass if self.positive_mass else 0.0
            ),
            "tool_failure_credit/flagged_row_share": (
                self.flagged_rows / self.trained_rows if self.trained_rows else 0.0
            ),
            "tool_failure_credit/failed_turn_share": (
                self.failed_turns / self.turns if self.turns else 0.0
            ),
            **{
                f"tool_failure_credit/failed_calls_per_100_responses/{category}": (
                    100.0 * self.failed_calls.get(category, 0) / self.tool_responses
                    if self.tool_responses
                    else 0.0
                )
                for category in CATEGORIES
            },
        }


class FailedToolTurnDetector:
    """Failed-turn masks for advantage-stage batches, from their token rows, with the policy tokenizer.

    ``load`` (the tokenizer and the tag check) runs at controller setup, so a tokenizer without single-token tool
    tags fails the run at startup rather than at the first advantage stage.
    """

    def __init__(
        self, tokenizer_config: dict[str, Any], categories: Collection[str]
    ) -> None:
        self._tokenizer_config = tokenizer_config
        self._categories = tuple(categories)
        self._tokenizer: Any = None
        self._ids: Optional[ToolTagIds] = None

    def load(self) -> None:
        from nemo_rl.algorithms.utils import get_tokenizer

        self._tokenizer = get_tokenizer(self._tokenizer_config)
        self._ids = resolve_tool_tag_ids(self._tokenizer)

    def _decode_many(self, seqs: list[list[int]]) -> list[str]:
        return self._tokenizer.batch_decode(seqs, skip_special_tokens=False)

    def mask(
        self,
        input_ids: torch.Tensor,
        token_mask: torch.Tensor,
        scan_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, ToolFailureCreditStats]:
        """``(B, T)`` bool: the generated tokens of turns with a failed tool call; plus the detection counts.

        Args:
            input_ids: ``(B, ...)`` the rows' token ids, jagged (nested) or right-padded.
            token_mask: ``(B, T)`` the right-padded generated-token mask the advantages are aligned with.
            scan_rows: ``(B,)`` bool, the rows to scan (rows without trained tokens are left unmarked).
        """
        assert self._ids is not None, "FailedToolTurnDetector.load() was not called"
        rows = input_ids.unbind() if input_ids.is_nested else input_ids
        generated = token_mask.detach().cpu()
        width = int(generated.shape[-1])
        failed = torch.zeros(generated.shape, dtype=torch.bool)
        stats = ToolFailureCreditStats()
        for i in torch.nonzero(scan_rows.cpu(), as_tuple=False).reshape(-1).tolist():
            token_ids = rows[i].detach().cpu().reshape(-1)
            length = int(token_ids.shape[0])
            assert length <= width, (
                f"row {i}: {length} input ids but a token_mask of width {width}"
            )
            found = failed_tool_turns(
                token_ids[:length].numpy(),
                generated[i, :length].numpy(),
                ids=self._ids,
                decode_many=self._decode_many,
                categories=self._categories,
            )
            failed[i, :length] = torch.from_numpy(found.mask)
            stats.turns += found.turns
            stats.failed_turns += found.failed_turns
            stats.tool_responses += found.tool_responses
            for category, count in found.failed_calls.items():
                stats.failed_calls[category] = (
                    stats.failed_calls.get(category, 0) + count
                )
        return failed, stats


def zero_positive_credit_on_failed_turns(
    advantages: torch.Tensor,
    *,
    failed_turn_mask: torch.Tensor,
    train_mask: torch.Tensor,
) -> tuple[torch.Tensor, ToolFailureCreditStats]:
    """Clip the advantages of the trained tokens in failed turns to ``min(A, 0)``.

    Args:
        advantages: ``(B, T)`` the advantage stage's advantages.
        failed_turn_mask: ``(B, T)`` True on the generated tokens of turns with a failed tool call.
        train_mask: ``(B, T)`` > 0 on trained tokens.

    Returns:
        The new advantages (same shape, dtype and device) and the step statistics.
    """
    if (
        failed_turn_mask.shape != advantages.shape
        or train_mask.shape != advantages.shape
    ):
        raise ValueError(
            "tool_failure_credit needs advantages, failed_turn_mask and train_mask of one shape, got "
            f"{tuple(advantages.shape)}, {tuple(failed_turn_mask.shape)}, {tuple(train_mask.shape)}"
        )
    trained = train_mask.to(device=advantages.device) > 0
    flagged = failed_turn_mask.to(device=advantages.device).bool() & trained
    positive = advantages > 0
    zeroed = flagged & positive
    stats = ToolFailureCreditStats(
        trained_tokens=float(trained.sum()),
        flagged_tokens=float(flagged.sum()),
        zeroed_tokens=float(zeroed.sum()),
        positive_mass=float(advantages[trained & positive].sum()),
        removed_mass=float(advantages[zeroed].sum()),
        trained_rows=int(trained.any(dim=-1).sum()),
        flagged_rows=int(flagged.any(dim=-1).sum()),
    )
    return torch.where(zeroed, torch.zeros_like(advantages), advantages), stats
