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

"""Step-assembly accounting for the SingleController train pump.

Owns the cursor over the step the train pump is assembling and charges a
permanently lost prompt group to the step that will close short for it. Kept out
of ``SingleControllerActor`` so the assembling cursor, its step-assembly record
and the shortfall-crediting rules live in one place; the controller drives it
through ``begin_step``/``finish_step`` and the per-loss ``credit_shortfall``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nemo_rl.algorithms.async_utils.staleness_sampler import PromptGroupSampler
    from nemo_rl.algorithms.rollout_checkpointing.state import RolloutRuntimeState


@dataclass(frozen=True)
class ShortfallCredit:
    target_step: Optional[int]
    charged_step: int
    allow_short: bool


@dataclass
class StepAssembly:
    version: int
    selected_groups: int
    selection_credits: Optional[list[ShortfallCredit]]


class StepAssembler:
    """Track the step being assembled and charge lost groups to it.

    Holds the sampler, the per-step prompt-group target and the shared
    ``RolloutRuntimeState`` object directly. The runtime's ``batch_shortfall`` /
    ``steps_allowed_short`` / ``shortfall_reconciliation_pending`` maps and sets
    are read live off that object, so restore replacements stay visible. The
    cursor starts from the restored trainer version so step ``0`` stays valid
    after a restore.
    """

    def __init__(
        self,
        *,
        sampler: "PromptGroupSampler",
        num_prompts_per_step: int,
        runtime: "RolloutRuntimeState",
        initial_version: int,
    ) -> None:
        self._sampler = sampler
        self._num_prompts_per_step = num_prompts_per_step
        self._runtime = runtime
        self._assembling_version = initial_version
        self._step_assembly: Optional[StepAssembly] = None
        # Per-step count of allow-short reasons this assembler charged, so
        # reconciliation undoes only its own racing contribution (see
        # ``_allow_short_reasons``).
        self._allow_short_charges: dict[int, int] = {}

    @property
    def assembling_version(self) -> int:
        """The step the train pump is selecting groups for, or will select next."""
        return self._assembling_version

    @property
    def step_assembly(self) -> Optional[StepAssembly]:
        """The open step's assembly record, or ``None`` between steps."""
        return self._step_assembly

    def begin_step(self, version: int) -> None:
        """Open a fresh step's assembly record at ``version``."""
        self._step_assembly = StepAssembly(version, 0, None)

    def finish_step(self, version_during_step: int) -> None:
        """Close the step: a group lost from here on counts toward the next."""
        self._assembling_version = max(
            self._assembling_version, version_during_step + 1
        )
        self._step_assembly = None

    def accounting_step(self, target_step: Optional[int]) -> int:
        """The step a group counts toward: its own, or the one being assembled."""
        return self._assembling_version if target_step is None else target_step

    def _charge_step_for_lost_group(self, target_step: Optional[int]) -> Optional[int]:
        """The step a permanently lost group is charged to, or None for no step."""
        return self._sampler.charge_step_for_lost_group(
            target_step=target_step, assembling_version=self._assembling_version
        )

    def advance_assembling_version(self) -> None:
        """Move past steps whose selected groups and losses fill their batch."""
        assembly = self._step_assembly
        while True:
            selected = (
                assembly.selected_groups
                if assembly is not None and assembly.version == self._assembling_version
                else 0
            )
            lost = self._runtime.batch_shortfall.get(self._assembling_version, 0)
            if selected + lost < self._num_prompts_per_step:
                return
            self._assembling_version += 1

    def _mark_step_allowed_short(self, charged_step: int) -> None:
        """Record one allow-short reason the assembler charged to ``charged_step``.

        A step already permitted by another path -- a drained pooled step, an
        exhausted close, or a checkpoint restore -- has no tracked count, so its
        standing permission counts as one reason the assembler will not release
        (see ``_allow_short_reasons``). The assembler's own charge then stacks on
        top, so reconciliation can undo it without dropping the other path's.
        """
        self._allow_short_charges[charged_step] = (
            self._allow_short_reasons(charged_step) + 1
        )
        self._runtime.steps_allowed_short.add(charged_step)

    def _release_step_allowed_short(self, charged_step: int) -> None:
        """Undo one allow-short reason the assembler charged; drop at the last."""
        remaining = self._allow_short_reasons(charged_step) - 1
        if remaining > 0:
            self._allow_short_charges[charged_step] = remaining
        else:
            self._allow_short_charges.pop(charged_step, None)
            self._runtime.steps_allowed_short.discard(charged_step)

    def _allow_short_reasons(self, charged_step: int) -> int:
        """Reasons the step may close short: tracked charges, or 1 if premarked."""
        tracked = self._allow_short_charges.get(charged_step)
        if tracked is not None:
            return tracked
        return 1 if charged_step in self._runtime.steps_allowed_short else 0

    def reconcile_selection_credits(
        self, assembling_version: int, credits: list[ShortfallCredit]
    ) -> None:
        """Reapply only losses that raced an awaited selection, using its result."""
        for credit in credits:
            remaining = self._runtime.batch_shortfall[credit.charged_step] - 1
            if remaining:
                self._runtime.batch_shortfall[credit.charged_step] = remaining
            else:
                del self._runtime.batch_shortfall[credit.charged_step]
            # Undo this racing credit's allow-short permission; the reapplication
            # below re-adds it at the settled charged step (see _release...).
            if credit.allow_short:
                self._release_step_allowed_short(credit.charged_step)
        self._assembling_version = assembling_version
        self.advance_assembling_version()
        for credit in credits:
            self.credit_shortfall(credit.target_step, allow_short=credit.allow_short)
        self._runtime.shortfall_reconciliation_pending = False

    def credit_shortfall(
        self,
        target_step: Optional[int],
        *,
        allow_short: bool = False,
        prompt_idx: Optional[int] = None,
    ) -> None:
        """Record that a step will never receive a group it is waiting for.

        ``prompt_idx`` is the dataset index of the lost prompt, when known; it is
        logged so repeated losses can be traced to the prompts they cluster on.
        """
        self.advance_assembling_version()
        charged_step = self._charge_step_for_lost_group(target_step)
        if charged_step is None:
            return
        self._runtime.batch_shortfall[charged_step] = (
            self._runtime.batch_shortfall.get(charged_step, 0) + 1
        )
        if allow_short:
            self._mark_step_allowed_short(charged_step)
        assembly = self._step_assembly
        if (
            self._sampler.requires_shortfall_reconciliation
            and assembly is not None
            and assembly.selection_credits is not None
        ):
            assembly.selection_credits.append(
                ShortfallCredit(target_step, charged_step, allow_short)
            )
            self._runtime.shortfall_reconciliation_pending = True
        self.advance_assembling_version()
        lost = "" if prompt_idx is None else f" (lost prompt idx={prompt_idx})"
        print(
            f"  target_step={charged_step} is one group short{lost} "
            f"({self._runtime.batch_shortfall[charged_step]} total); the train pump "
            "will close that step early",
            flush=True,
        )
